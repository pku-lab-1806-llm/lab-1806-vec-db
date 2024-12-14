use anyhow::{anyhow, bail, Result};
use fs2::FileExt;
use metadata_vec_table::MetadataVecTable;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Mutex, MutexGuard, RwLock},
    time::Duration,
};
use thread_save::{ThreadSave, ThreadSavingManager};

use crate::prelude::*;

pub mod dynamic_index;
pub mod metadata_vec_table;
pub mod thread_save;

/// Acquire an exclusive lock on a file.
pub fn acquire_lock(lock_file: impl AsRef<Path>) -> Result<File> {
    let file = File::options()
        .write(true)
        .create(true)
        .open(lock_file.as_ref())?;
    file.try_lock_exclusive()
        .map_err(|_| anyhow!("Failed to acquire lock for VecDBManager"))?;
    Ok(file)
}

/// Sanitize a key for use in file names.
/// Replace "<>:/\|?*" with "_".
pub fn sanitize_key(key: &str) -> String {
    key.chars()
        .map(|c| match c {
            '<' | '>' | ':' | '/' | '\\' | '|' | '?' | '*' => '_',
            _ => c,
        })
        .collect()
}

/// Compute the SHA-256 hash of the input data and return it as a hexadecimal string.
pub fn sha256_hex(data: &[u8]) -> String {
    let result = Sha256::digest(data);
    base16ct::lower::encode_string(&result)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecTableBrief {
    pub filename: String,
}
impl VecTableBrief {
    pub fn new(filename: String) -> Self {
        Self { filename }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecDBBrief {
    /// key -> filename
    #[serde(default)]
    tables: BTreeMap<String, VecTableBrief>,
    #[serde(skip, default)]
    filename_set: BTreeSet<String>,
}
impl VecDBBrief {
    pub fn new() -> Self {
        Self {
            tables: BTreeMap::new(),
            filename_set: BTreeSet::new(),
        }
    }
    pub fn contains(&self, key: &str) -> bool {
        self.tables.contains_key(key)
    }
    /// Choose a filename for a key. Ensure the filename is unique.

    pub fn insert(&mut self, key: &str) -> String {
        fn filename_with(base: &str, index: usize) -> String {
            if index == 0 {
                format!("{}.db", base)
            } else {
                format!("{}_{}.db", base, index)
            }
        }
        fn choose_filename(set: &mut BTreeSet<String>, key: &str) -> String {
            let base = sanitize_key(key);
            let mut index = 0;
            loop {
                let filename = filename_with(&base, index);
                if !set.insert(filename.clone()) {
                    return filename;
                }
                index += 1;
            }
        }
        let filename = choose_filename(&mut self.filename_set, key);
        self.tables
            .insert(key.to_string(), VecTableBrief::new(filename.clone()));
        filename
    }
    pub fn remove(&mut self, key: &str) {
        self.tables
            .remove(key)
            .map(|brief| self.filename_set.remove(&brief.filename));
    }
    pub fn load(file: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(file)?;
        let mut brief: Self = toml::from_str(&content)?;
        brief.filename_set = brief
            .tables
            .values()
            .map(|f| f.filename.to_string())
            .collect();
        if brief.tables.len() != brief.filename_set.len() {
            bail!("Duplicate filenames in the brief");
        }
        Ok(brief)
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
    index: ThreadSavingManager<RwLock<MetadataVecTable>>,
    drop_signal_sender: mpsc::Sender<()>,
}
impl VecTableManager {
    /// Get the file path of a table.
    fn file_path_of(dir: impl AsRef<Path>, key: &str) -> PathBuf {
        let hex_key = sha256_hex(key.as_bytes());
        dir.as_ref().join(format!("{}.db", hex_key))
    }
    fn thread_saving_duration() -> Duration {
        Duration::from_secs(60)
    }
    /// Create a new table manager.
    pub fn new(
        path: impl AsRef<Path>,
        dim: usize,
        dist: DistanceAlgorithm,
        drop_signal_sender: std::sync::mpsc::Sender<()>,
    ) -> Self {
        let index = MetadataVecTable::new(dim, dist);
        let index = ThreadSavingManager::new_rw(index, path, Self::thread_saving_duration(), true);
        Self {
            index,
            drop_signal_sender,
        }
    }
    /// Load a table manager from a file.
    pub fn load(
        path: impl AsRef<Path>,
        drop_signal_sender: std::sync::mpsc::Sender<()>,
    ) -> Result<Self> {
        let index = MetadataVecTable::load(&path)?;
        let index =
            ThreadSavingManager::new_rw(index, &path, Self::thread_saving_duration(), false);
        Ok(Self {
            index,
            drop_signal_sender,
        })
    }
    /// Get number of vectors in the table.
    pub fn len(&self) -> usize {
        self.index.read().len()
    }
    /// Get dimension of the vectors in the table.
    pub fn dim(&self) -> usize {
        self.index.read().dim()
    }
    /// Get the distance algorithm used by the table.
    pub fn dist(&self) -> DistanceAlgorithm {
        self.index.read().dist()
    }
    /// Add a vector with metadata to the table.
    pub fn add(&self, vec: Vec<f32>, metadata: BTreeMap<String, String>) {
        self.index.write().add(vec, metadata);
    }
    /// Add multiple vectors with metadata to the table.
    pub fn batch_add(&self, vec_list: Vec<Vec<f32>>, metadata_list: Vec<BTreeMap<String, String>>) {
        self.index.write().batch_add(vec_list, metadata_list);
    }
    /// Delete vectors with metadata that match the pattern.
    pub fn delete(&self, pattern: &BTreeMap<String, String>) -> usize {
        self.index.write().delete(pattern)
    }
    /// Build an HNSW index for the table.
    pub fn build_hnsw_index(&self, ef_construction: Option<usize>) {
        self.index.write().build_hnsw_index(ef_construction);
    }
    /// Clear the HNSW index for the table.
    pub fn clear_hnsw_index(&self) {
        self.index.write().clear_hnsw_index();
    }
    /// Check if the table has an HNSW index.
    pub fn has_hnsw_index(&self) -> bool {
        self.index.read().has_hnsw_index()
    }
    /// Build a PQ table for the table.
    pub fn build_pq_table(&self, m: usize, train_size: usize) -> Result<()> {
        self.index.write().build_pq_table(m, train_size)
    }
    /// Clear the PQ table for the table.
    pub fn clear_pq_table(&self) {
        self.index.write().clear_pq_table();
    }
    /// Check if the table has a PQ table.
    pub fn has_pq_table(&self) -> bool {
        self.index.read().has_pq_table()
    }
    /// Search for the nearest vectors to a query.
    /// Return a list of (metadata, distance) pairs.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        upper_bound: Option<f32>,
    ) -> Vec<(BTreeMap<String, String>, f32)> {
        self.index.read().search(query, k, ef, upper_bound)
    }
}
impl Drop for VecTableManager {
    fn drop(&mut self) {
        // Sync the index to the file before dropping.
        self.index.sync_save(true);
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
    brief: ThreadSavingManager<Mutex<VecDBBrief>>,
    tables: Mutex<BTreeMap<String, (mpsc::Receiver<()>, Arc<VecTableManager>)>>,
    /// The last field to be dropped.
    #[allow(unused)]
    lock_file: File,
}
impl VecDBManager {
    /// Get the file path of the brief.
    fn brief_file_path(dir: impl AsRef<Path>) -> PathBuf {
        dir.as_ref().join("brief.toml")
    }
    /// Create a new VecDBManager.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?;
        }
        let brief_file = Self::brief_file_path(&dir);
        let (brief, mark) = if brief_file.exists() {
            (VecDBBrief::load(&brief_file)?, false)
        } else {
            (VecDBBrief::new(), true)
        };
        let brief = ThreadSavingManager::new_mutex(
            brief,
            brief_file,
            std::time::Duration::from_secs(5),
            mark,
        );
        let lock_file = acquire_lock(dir.join("db.lock"))?;
        Ok(Self {
            dir,
            brief,
            lock_file,
            tables: Mutex::new(BTreeMap::new()),
        })
    }
    /// Get locks in the correct order.
    fn get_locks_by_order(
        &self,
    ) -> (
        MutexGuard<'_, VecDBBrief>,
        MutexGuard<'_, BTreeMap<String, (mpsc::Receiver<()>, Arc<VecTableManager>)>>,
    ) {
        let brief = self.brief.lock();
        let tables = self.tables.lock().unwrap();
        (brief, tables)
    }
    /// Get all table keys.
    pub fn get_all_keys(&self) -> Vec<String> {
        let (brief, _table) = self.get_locks_by_order();
        brief.tables.keys().cloned().collect()
    }
    /// Check if a table exists.
    pub fn contains_key(&self, key: &str) -> bool {
        let (brief, _table) = self.get_locks_by_order();
        brief.contains(key)
    }
    /// Get all cached table keys.
    pub fn get_cached_tables(&self) -> Vec<String> {
        let (_brief, tables) = self.get_locks_by_order();
        tables.keys().cloned().collect()
    }
    pub fn contains_cached(&self, key: &str) -> bool {
        let (_brief, tables) = self.get_locks_by_order();
        tables.contains_key(key)
    }
    /// Remove a cached table.
    pub fn remove_cached_table(&self, key: &str) -> Result<()> {
        let (_brief, mut tables) = self.get_locks_by_order();
        if let Some((receiver, table)) = tables.remove(key) {
            // Wait for other threads to finish.
            drop(table);
            receiver.recv().unwrap();
        }
        Ok(())
    }
    /// Create a table if it does not exist.
    pub fn create_table_if_not_exists(
        &self,
        key: &str,
        dim: usize,
        dist: DistanceAlgorithm,
    ) -> Result<bool> {
        let (mut brief, mut tables) = self.get_locks_by_order();
        if brief.tables.contains_key(key) {
            return Ok(false);
        }
        let filename = brief.insert(&key);
        let path = self.dir.join(&filename);

        let (sender, receiver) = mpsc::channel();
        let table = VecTableManager::new(path, dim, dist, sender);
        let table = Arc::new(table);
        tables.insert(key.to_string(), (receiver, table));
        Ok(true)
    }
    /// Delete a table.
    pub fn delete_table(&self, key: &str) -> Result<bool> {
        let (mut brief, mut tables) = self.get_locks_by_order();
        if !brief.tables.contains_key(key) {
            return Ok(false);
        }
        brief.remove(key);
        if let Some((receiver, table)) = tables.remove(key) {
            // Wait for other threads to finish.
            drop(table);
            receiver.recv().unwrap();
        }
        // Remove the file.
        let table_file = VecTableManager::file_path_of(&self.dir, key);
        std::fs::remove_file(table_file)?;
        Ok(true)
    }
    /// Get a table with the correct locks.
    fn table(&self, key: &str) -> Result<Arc<VecTableManager>> {
        let (brief, mut tables) = self.get_locks_by_order();
        if !brief.tables.contains_key(key) {
            return Err(anyhow!("Table {} not found", key));
        }
        if !tables.contains_key(key) {
            let (sender, receiver) = mpsc::channel();
            let filename = &brief.tables[key].filename;
            let path = self.dir.join(filename);
            let table = VecTableManager::load(&path, sender)?;
            tables.insert(key.to_string(), (receiver, Arc::new(table)));
        }
        Ok(tables[key].1.clone())
    }
    pub fn get_len(&self, key: &str) -> Result<usize> {
        Ok(self.table(key)?.len())
    }
    pub fn get_dim(&self, key: &str) -> Result<usize> {
        Ok(self.table(key)?.dim())
    }
    pub fn get_dist(&self, key: &str) -> Result<DistanceAlgorithm> {
        Ok(self.table(key)?.dist())
    }

    /// Add a vector with metadata to a table.
    pub fn add(&self, key: &str, vec: Vec<f32>, metadata: BTreeMap<String, String>) -> Result<()> {
        let table = self.table(key)?;
        if vec.len() != table.dim() {
            bail!("Dimension mismatch for vec");
        }
        table.add(vec, metadata);
        Ok(())
    }

    /// Add multiple vectors with metadata to a table.
    pub fn batch_add(
        &self,
        key: &str,
        vec_list: Vec<Vec<f32>>,
        metadata_list: Vec<BTreeMap<String, String>>,
    ) -> Result<()> {
        if vec_list.len() != metadata_list.len() {
            return Err(anyhow!("Length mismatch for vec_list and metadata_list"));
        }
        let table = self.table(key)?;
        if vec_list.iter().any(|v| v.len() != table.dim()) {
            bail!("Dimension mismatch for vec_list");
        }
        table.batch_add(vec_list, metadata_list);
        Ok(())
    }

    /// Build an HNSW index for a table.
    pub fn build_hnsw_index(&self, key: &str, ef_construction: Option<usize>) -> Result<()> {
        Ok(self.table(key)?.build_hnsw_index(ef_construction))
    }
    /// Clear the HNSW index for a table.
    pub fn clear_hnsw_index(&self, key: &str) -> Result<()> {
        Ok(self.table(key)?.clear_hnsw_index())
    }
    /// Check if a table has an HNSW index.
    pub fn has_hnsw_index(&self, key: &str) -> Result<bool> {
        Ok(self.table(key)?.has_hnsw_index())
    }

    pub fn build_pq_table(&self, key: &str, m: usize, train_size: usize) -> Result<()> {
        Ok(self.table(key)?.build_pq_table(m, train_size)?)
    }
    pub fn clear_pq_table(&self, key: &str) -> Result<()> {
        Ok(self.table(key)?.clear_pq_table())
    }
    pub fn has_pq_table(&self, key: &str) -> Result<bool> {
        Ok(self.table(key)?.has_pq_table())
    }

    /// Delete vectors with metadata that match the pattern.
    /// Return the number of vectors deleted.
    pub fn delete(&self, key: &str, pattern: &BTreeMap<String, String>) -> Result<usize> {
        Ok(self.table(key)?.delete(pattern))
    }

    /// Search for the nearest vectors to a query.
    /// Return a list of (metadata, distance) pairs.
    /// - When only Flat index is available, use the Flat index for search and ignore ef.
    /// - When HNSW index is available, use the HNSW index for search.
    /// - When PQ table is available, and ef is specified, use the PQ table for search.
    pub fn search(
        &self,
        key: &str,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        upper_bound: Option<f32>,
    ) -> Result<Vec<(BTreeMap<String, String>, f32)>> {
        Ok(self.table(key)?.search(query, k, ef, upper_bound))
    }
}
impl Drop for VecDBManager {
    fn drop(&mut self) {
        // Sync the brief to the file before dropping.
        self.brief.sync_save(true);
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
                db.add(key_a, vec![1.0, 0.0, 0.0, 0.0], metadata("a"))
                    .unwrap();
                db.build_hnsw_index(key_a, None).unwrap();
                db.add(key_a, vec![0.0, 1.0, 0.0, 0.0], metadata("b"))
                    .unwrap();
                db.add(key_a, vec![0.0, 0.0, 1.0, 0.0], metadata("c"))
                    .unwrap();
                s_a.send(()).unwrap();
            });
            s.spawn(|| {
                let key_b = "table_b";
                db.create_table_if_not_exists(key_b, dim, dist).unwrap();
                db.build_hnsw_index(key_b, None).unwrap();
                db.batch_add(
                    key_b,
                    vec![
                        vec![0.0, 0.0, 0.0, 0.1],
                        vec![0.0, 1.0, 0.0, 0.1],
                        vec![0.0, 0.0, 1.0, 0.1],
                    ],
                    vec![metadata("a'"), metadata("b'"), metadata("c'")],
                )
                .unwrap();
                db.delete(key_b, &metadata("a'")).unwrap();
                db.add(key_b, vec![1.0, 0.0, 0.0, 0.1], metadata("d"))
                    .unwrap();
            });
            let db_ref = &db;
            let s_c_ref = &s_c;
            s.spawn(move || {
                r_a.recv().unwrap();
                db_ref.get_len("table_a").unwrap();
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

        Ok(())
    }
}
