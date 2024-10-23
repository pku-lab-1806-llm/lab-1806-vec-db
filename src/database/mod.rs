use anyhow::{anyhow, Result};
use fs2::FileExt;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Mutex, MutexGuard, RwLock},
};
use thread_save::ThreadSave;

use crate::{
    index_algorithm::{HNSWConfig, HNSWIndex},
    prelude::*,
};

pub mod thread_save;

pub fn acquire_lock(lock_file: impl AsRef<Path>) -> Result<File> {
    let file = File::options()
        .write(true)
        .create(true)
        .open(lock_file.as_ref())?;
    file.lock_exclusive()?;
    Ok(file)
}

pub fn sha256_hex(data: &[u8]) -> String {
    let result = Sha256::digest(data);
    base16ct::lower::encode_string(&result)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataIndex {
    metadata: Vec<BTreeMap<String, String>>,
    inner: HNSWIndex<f32>,
    #[serde(skip, default = "rand::SeedableRng::from_entropy")]
    rng: rand::rngs::StdRng,
}
impl MetadataIndex {
    /// Create a new index.
    pub fn new(dim: usize, dist: DistanceAlgorithm) -> Self {
        let config = HNSWConfig::default();
        let rng = rand::SeedableRng::from_entropy();
        Self {
            metadata: Vec::new(),
            inner: HNSWIndex::new(dim, dist, config),
            rng,
        }
    }
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
    pub fn dist(&self) -> DistanceAlgorithm {
        self.inner.config.dist
    }
    /// Load an index from a file.
    pub fn load(file: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(file)?;
        let buf_reader = BufReader::new(file);
        let index: Self = bincode::deserialize_from(buf_reader)?;
        Ok(index)
    }
    /// Save the index to a file.
    pub fn save(&self, file: impl AsRef<Path>) -> Result<()> {
        let file = File::create(file)?;
        let buf_writer = std::io::BufWriter::new(file);
        bincode::serialize_into(buf_writer, self)?;
        Ok(())
    }

    /// Add a vector with metadata to the index.
    pub fn add(&mut self, vec: Vec<f32>, metadata: BTreeMap<String, String>) {
        self.metadata.push(metadata);
        self.inner.add(&vec, &mut self.rng);
    }

    /// Add vectors with metadata to the index.
    pub fn batch_add(
        &mut self,
        vec_list: Vec<Vec<f32>>,
        metadata_list: Vec<BTreeMap<String, String>>,
    ) {
        assert_eq!(vec_list.len(), metadata_list.len());
        self.metadata.extend(metadata_list);
        let vec_list: Vec<_> = vec_list.iter().map(|v| v.as_slice()).collect();
        self.inner.batch_add(&vec_list, &mut self.rng);
    }

    /// Get the total number of vectors.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Search for the nearest vectors to query.
    ///
    /// Returns a list of (metadata, distance) pairs.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        upper_bound: Option<f32>,
    ) -> Vec<(BTreeMap<String, String>, f32)> {
        let results = match ef {
            Some(ef) => self.inner.knn_with_ef(&query, k, ef),
            None => self.inner.knn(&query, k),
        };
        let upper_bound = upper_bound.unwrap_or(std::f32::INFINITY);
        results
            .into_iter()
            .filter(|p| p.distance() <= upper_bound)
            .map(|p| (self.metadata[p.index].clone(), p.distance()))
            .collect()
    }
}

impl ThreadSave for RwLock<MetadataIndex> {
    fn save_to(&self, path: impl AsRef<Path>) {
        self.read().unwrap().save(path).unwrap();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecTableBrief {
    pub dim: usize,
    pub len: usize,
    pub dist: DistanceAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecDBBrief {
    #[serde(default)]
    tables: BTreeMap<String, VecTableBrief>,
}
impl VecDBBrief {
    pub fn new() -> Self {
        Self {
            tables: BTreeMap::new(),
        }
    }
    pub fn insert(&mut self, key: &str, dim: usize, len: usize, dist: DistanceAlgorithm) {
        self.tables
            .insert(key.to_string(), VecTableBrief { dim, len, dist });
    }
    pub fn remove(&mut self, key: &str) {
        self.tables.remove(key);
    }
    pub fn load(file: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(file)?;
        Ok(toml::from_str(&content)?)
    }
    pub fn save(&self, file: impl AsRef<Path>) -> Result<()> {
        let content = toml::to_string(self)?;
        std::fs::write(&file, content)?;
        Ok(())
    }
}
impl ThreadSave for Mutex<VecDBBrief> {
    fn save_to(&self, path: impl AsRef<Path>) {
        self.lock().unwrap().save(path).unwrap();
    }
}

/// A manager for a vector table.
///
/// Ensures:
/// - Auto-save the index to the file.
/// - Thread-safe. Read and write operations are protected by a RwLock.
/// - Unique. Only one manager for each table.
struct VecTableManager {
    manager: thread_save::ThreadSavingManager<RwLock<MetadataIndex>>,
    index: Arc<RwLock<MetadataIndex>>,
    drop_signal_sender: mpsc::Sender<()>,
}
impl VecTableManager {
    fn file_path_of(dir: impl AsRef<Path>, key: &str) -> PathBuf {
        let hex_key = sha256_hex(key.as_bytes());
        dir.as_ref().join(format!("{}.db", hex_key))
    }
    fn new_manager(
        file: PathBuf,
        index: &Arc<RwLock<MetadataIndex>>,
    ) -> thread_save::ThreadSavingManager<RwLock<MetadataIndex>> {
        thread_save::ThreadSavingManager::new(
            index.clone(),
            file,
            std::time::Duration::from_secs(30),
        )
    }
    pub fn new(
        dir: impl AsRef<Path>,
        key: &str,
        dim: usize,
        dist: DistanceAlgorithm,
        drop_signal_sender: std::sync::mpsc::Sender<()>,
    ) -> Self {
        let file = Self::file_path_of(&dir, &key);
        let index = MetadataIndex::new(dim, dist);
        let index = Arc::new(RwLock::new(index));
        let manager = Self::new_manager(file, &index);
        manager.mark_modified();
        Self {
            manager,
            index,
            drop_signal_sender,
        }
    }
    pub fn load(
        dir: impl AsRef<Path>,
        key: &str,
        drop_signal_sender: std::sync::mpsc::Sender<()>,
    ) -> Result<Self> {
        let file = Self::file_path_of(&dir, &key);
        let index = MetadataIndex::load(&file)?;
        let index = Arc::new(RwLock::new(index));
        let manager = Self::new_manager(file, &index);
        Ok(Self {
            manager,
            index,
            drop_signal_sender,
        })
    }
    pub fn info(&self) -> VecTableBrief {
        let index = self.index.read().unwrap();
        VecTableBrief {
            dim: index.dim(),
            len: index.len(),
            dist: index.dist(),
        }
    }
    /// Add a vector with metadata to the table.
    ///
    /// Signal the manager after the operation.
    pub fn add(&self, vec: Vec<f32>, metadata: BTreeMap<String, String>) {
        let mut index = self.index.write().unwrap();
        index.add(vec, metadata);
        self.manager.mark_modified();
    }
    /// Add vectors with metadata to the table.
    ///
    /// Signal the manager after the operation.
    pub fn batch_add(&self, vec_list: Vec<Vec<f32>>, metadata_list: Vec<BTreeMap<String, String>>) {
        let mut index = self.index.write().unwrap();
        index.batch_add(vec_list, metadata_list);
        self.manager.mark_modified();
    }
    /// Search vector in the table.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        upper_bound: Option<f32>,
    ) -> Vec<(BTreeMap<String, String>, f32)> {
        let index = self.index.read().unwrap();
        index.search(query, k, ef, upper_bound)
    }
}
impl Drop for VecTableManager {
    fn drop(&mut self) {
        // Sync the index to the file before dropping.
        self.manager.sync_save(true);
        // Send the signal to the drop signal receiver.
        self.drop_signal_sender.send(()).unwrap();
    }
}

/// A manager for a vector database.
///
/// Ensures:
/// - Auto-save the brief to the file. And tables are saved to files when necessary.
/// - Thread-safe. Read and write operations to the brief are protected by a Mutex. Operations to the tables are protected by thread-safe managers.
/// - Unique. Only one manager for each database.
///
/// Order of mutex lock: brief -> tables -> table_managers (by key)
pub struct VecDBManager {
    dir: PathBuf,
    brief: Arc<Mutex<VecDBBrief>>,
    brief_manager: thread_save::ThreadSavingManager<Mutex<VecDBBrief>>,
    tables: Mutex<BTreeMap<String, (mpsc::Receiver<()>, Arc<VecTableManager>)>>,
    /// The last field to be dropped.
    #[allow(unused)]
    lock_file: File,
}
impl VecDBManager {
    fn brief_file_path(dir: impl AsRef<Path>) -> PathBuf {
        dir.as_ref().join("brief.toml")
    }
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?;
        }
        let brief_file = Self::brief_file_path(&dir);
        let brief = if brief_file.exists() {
            VecDBBrief::load(&brief_file)?
        } else {
            VecDBBrief::new()
        };
        let brief = Arc::new(Mutex::new(brief));
        let brief_manager = thread_save::ThreadSavingManager::new(
            brief.clone(),
            brief_file,
            std::time::Duration::from_secs(5),
        );
        brief_manager.mark_modified();
        let lock_file = acquire_lock(dir.join("db.lock"))?;
        Ok(Self {
            dir,
            brief,
            brief_manager,
            lock_file,
            tables: Mutex::new(BTreeMap::new()),
        })
    }
    fn get_locks_by_order(
        &self,
    ) -> (
        MutexGuard<'_, VecDBBrief>,
        MutexGuard<'_, BTreeMap<String, (mpsc::Receiver<()>, Arc<VecTableManager>)>>,
    ) {
        let brief = self.brief.lock().unwrap();
        let tables = self.tables.lock().unwrap();
        (brief, tables)
    }
    /// Returns a list of table keys.
    pub fn get_all_keys(&self) -> Vec<String> {
        let (brief, _table) = self.get_locks_by_order();
        brief.tables.keys().cloned().collect()
    }
    /// Returns a list of table keys that are cached.
    pub fn get_cached_tables(&self) -> Vec<String> {
        let (_brief, tables) = self.get_locks_by_order();
        tables.keys().cloned().collect()
    }
    /// Remove a table from the cache, and wait for all operations to finish.
    pub fn remove_cached_table(&self, key: &str) -> Result<()> {
        let (_brief, mut tables) = self.get_locks_by_order();
        if let Some((receiver, table)) = tables.remove(key) {
            // Wait for other threads to finish.
            drop(table);
            receiver.recv().unwrap();
        }
        Ok(())
    }
    /// Returns bool indicating whether the table is newly created.
    pub fn create_table_if_not_exists(
        &self,
        key: &str,
        dim: usize,
        dist: DistanceAlgorithm,
    ) -> Result<bool> {
        let (mut brief, mut tables) = self.get_locks_by_order();
        if brief.tables.contains_key(key) {
            return self
                .get_table_with_lock(key, &brief, &mut tables)
                .map(|_| false);
        }
        brief.insert(&key, dim, 0, dist);
        self.brief_manager.mark_modified();

        let (sender, receiver) = mpsc::channel();
        let table = VecTableManager::new(&self.dir, &key, dim, dist, sender);
        let table = Arc::new(table);
        tables.insert(key.to_string(), (receiver, table));
        Ok(true)
    }
    /// Delete a table and waits for all operations to finish.
    ///
    /// Signal the brief manager after the operation.
    pub fn delete_table(&self, key: &str) -> Result<()> {
        let (mut brief, mut tables) = self.get_locks_by_order();
        if !brief.tables.contains_key(key) {
            return Err(anyhow!("Table {} not found", key));
        }
        brief.remove(key);
        self.brief_manager.mark_modified();
        if let Some((receiver, table)) = tables.remove(key) {
            // Wait for other threads to finish.
            drop(table);
            receiver.recv().unwrap();
        }
        // Remove the file.
        let table_file = VecTableManager::file_path_of(&self.dir, key);
        std::fs::remove_file(table_file)?;
        Ok(())
    }
    pub fn get_table_info(&self, key: &str) -> Option<VecTableBrief> {
        let (brief, _tables) = self.get_locks_by_order();
        brief.tables.get(key).cloned()
    }
    fn get_table_with_lock(
        &self,
        key: &str,
        brief: &MutexGuard<'_, VecDBBrief>,
        tables: &mut MutexGuard<'_, BTreeMap<String, (mpsc::Receiver<()>, Arc<VecTableManager>)>>,
    ) -> Result<Arc<VecTableManager>> {
        if !brief.tables.contains_key(key) {
            return Err(anyhow!("Table {} not found", key));
        }
        if !tables.contains_key(key) {
            let (sender, receiver) = mpsc::channel();
            let table = VecTableManager::load(&self.dir, key, sender)?;
            tables.insert(key.to_string(), (receiver, Arc::new(table)));
        }
        Ok(tables[key].1.clone())
    }

    /// Add a vector with metadata to a table.
    ///
    /// Signal the brief manager after the operation.
    pub fn add(&self, key: &str, vec: Vec<f32>, metadata: BTreeMap<String, String>) {
        let (mut brief, mut tables) = self.get_locks_by_order();
        let table = self.get_table_with_lock(key, &brief, &mut tables).unwrap();
        table.add(vec, metadata);
        brief.tables.insert(key.to_string(), table.info());
        self.brief_manager.mark_modified();
    }

    /// Add vectors with metadata to a table.
    /// Call it with a batch size around 64 to avoid long lock time.
    ///
    /// Signal the brief manager after the operation.
    pub fn batch_add(
        &self,
        key: &str,
        vec_list: Vec<Vec<f32>>,
        metadata_list: Vec<BTreeMap<String, String>>,
    ) {
        let (mut brief, mut tables) = self.get_locks_by_order();
        let table = self.get_table_with_lock(key, &brief, &mut tables).unwrap();
        table.batch_add(vec_list, metadata_list);
        brief.tables.insert(key.to_string(), table.info());
        self.brief_manager.mark_modified();
    }

    /// Search vector in a table.
    ///
    /// Returns the results as a tuple of (metadata, distance).
    pub fn search(
        &self,
        key: &str,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        upper_bound: Option<f32>,
    ) -> Result<Vec<(BTreeMap<String, String>, f32)>> {
        let table = {
            let (brief, mut tables) = self.get_locks_by_order();
            self.get_table_with_lock(key, &brief, &mut tables)?
        };

        Ok(table.search(query, k, ef, upper_bound))
    }

    /// Search vector in multiple tables.
    ///
    /// Returns the results as a tuple of (table_key, metadata, distance).
    pub fn join_search(
        &self,
        keys: &BTreeSet<String>,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        upper_bound: Option<f32>,
    ) -> Result<Vec<(String, BTreeMap<String, String>, f32)>> {
        let selected_tables = {
            let (brief, mut tables) = self.get_locks_by_order();
            let mut selected_tables = Vec::new();
            for key in keys {
                let table = self.get_table_with_lock(key, &brief, &mut tables)?;
                selected_tables.push((key.clone(), table));
            }
            selected_tables
        };

        let (sender, receiver) = std::sync::mpsc::channel();
        let idx_list = (0..selected_tables.len()).collect::<Vec<_>>();
        std::thread::scope(|s| {
            for idx in idx_list.iter() {
                let (key, table) = &selected_tables[*idx];
                s.spawn(|| {
                    let results = table.search(query, k, ef, upper_bound);
                    let results: Vec<(String, BTreeMap<String, String>, f32)> = results
                        .into_iter()
                        .map(|(metadata, dist)| (key.clone(), metadata, dist))
                        .collect();
                    sender.send(results).unwrap();
                });
            }
        });
        drop(sender);
        let mut results = Vec::new();
        for r in receiver {
            results.extend(r);
        }
        results.sort_by_key(|(_, _, dist)| OrderedFloat(*dist));
        results.truncate(k);
        Ok(results)
    }
}
impl Drop for VecDBManager {
    fn drop(&mut self) {
        // Sync the brief to the file before dropping.
        self.brief_manager.sync_save(true);
        // Wait for all the table managers to be dropped.
        let mut tables = self.tables.lock().unwrap();
        while let Some((_, (receiver, table))) = tables.pop_first() {
            drop(table);
            receiver.recv().unwrap();
        }
        // After all table managers are dropped, we can safely remove the lock file.
    }
}

#[cfg(test)]
mod test {
    use std::thread;

    use super::*;

    #[test]
    fn test_vec_db_manager() -> Result<()> {
        let dir = "./tmp/vec_db";
        let db = VecDBManager::new(dir)?;
        for key in db.get_all_keys() {
            db.delete_table(&key)?;
        }
        let dim = 4;
        let dist = DistanceAlgorithm::Cosine;

        fn metadata(name: &str) -> BTreeMap<String, String> {
            BTreeMap::from([("name".to_string(), name.to_string())])
        }
        let (s_a, r_a) = std::sync::mpsc::channel();
        let (s_c, r_c) = std::sync::mpsc::channel();
        thread::scope(|s| {
            s.spawn(|| {
                let key_a = "table_a";
                db.create_table_if_not_exists(key_a, dim, dist).unwrap();
                s_a.send(()).unwrap();
                db.add(key_a, vec![1.0, 0.0, 0.0, 0.0], metadata("a"));
                db.add(key_a, vec![0.0, 1.0, 0.0, 0.0], metadata("b"));
                db.add(key_a, vec![0.0, 0.0, 1.0, 0.0], metadata("c"));
                s_a.send(()).unwrap();
            });
            s.spawn(|| {
                let key_b = "table_b";
                db.create_table_if_not_exists(key_b, dim, dist).unwrap();
                db.batch_add(
                    key_b,
                    vec![
                        vec![1.0, 0.0, 0.0, 0.1],
                        vec![0.0, 1.0, 0.0, 0.1],
                        vec![0.0, 0.0, 1.0, 0.1],
                    ],
                    vec![metadata("a'"), metadata("b'"), metadata("c'")],
                );
            });
            let db_ref = &db;
            let s_c_ref = &s_c;
            s.spawn(move || {
                r_a.recv().unwrap();
                db_ref.get_table_info("table_a").unwrap();
                r_a.recv().unwrap();
                let results = db_ref
                    .search("table_a", &[0.0, 0.0, 1.0, 0.0], 3, None, Some(0.5))
                    .unwrap();
                let results: Vec<String> = results
                    .into_iter()
                    .map(|(m, _)| m["name"].clone())
                    .collect();
                s_c_ref.send(results).unwrap();
            });
        });
        let c_result = r_c.recv().unwrap();
        assert_eq!(c_result, vec!["c".to_string()]);

        let results = db.join_search(
            &BTreeSet::from(["table_a".to_string(), "table_b".to_string()]),
            &[1.0, 0.0, 0.0, 0.0],
            2,
            None,
            None,
        )?;
        let results = results
            .into_iter()
            .map(|(_, m, _)| m["name"].clone())
            .collect::<Vec<_>>();
        assert_eq!(results, vec![String::from("a"), String::from("a'")]);

        Ok(())
    }
}
