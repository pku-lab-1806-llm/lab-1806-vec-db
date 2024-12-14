use anyhow::{anyhow, bail, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Mutex, MutexGuard, RwLock},
};
use thread_save::ThreadSave;

use crate::{
    distance::pq_table::{PQConfig, PQTable},
    index_algorithm::{CandidatePair, FlatIndex, HNSWConfig, HNSWIndex},
    prelude::*,
    vec_set::VecSet,
};

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

/// Compute the SHA-256 hash of the input data and return it as a hexadecimal string.
pub fn sha256_hex(data: &[u8]) -> String {
    let result = Sha256::digest(data);
    base16ct::lower::encode_string(&result)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptHNSWIndex {
    Flat(FlatIndex<f32>),
    HNSW(HNSWIndex<f32>),
}
impl OptHNSWIndex {
    /// Create a new OptHNSWIndex with the specified dimension and distance algorithm.
    pub fn new(dim: usize, dist: DistanceAlgorithm) -> Self {
        Self::Flat(FlatIndex::new(dim, dist))
    }
    /// Get the VecSet of the index.
    pub fn vec_set(&self) -> &VecSet<f32> {
        match &self {
            OptHNSWIndex::Flat(flat) => &flat.vec_set,
            OptHNSWIndex::HNSW(hnsw) => &hnsw.vec_set,
        }
    }
    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vec_set().len()
    }
    /// Get the dimension of the vectors in the index.
    pub fn dim(&self) -> usize {
        self.vec_set().dim()
    }
    /// Get the distance algorithm used by the index.
    pub fn dist(&self) -> DistanceAlgorithm {
        match self {
            Self::Flat(index) => index.dist,
            Self::HNSW(index) => index.config.dist,
        }
    }
    /// Add a vector to the index.
    pub fn add(&mut self, vec: &[f32], rng: &mut impl rand::Rng) -> usize {
        match self {
            Self::Flat(index) => index.vec_set.push(vec),
            Self::HNSW(index) => index.add(vec, rng),
        }
    }
    /// Add multiple vectors to the index.
    pub fn batch_add(&mut self, vec_list: &[&[f32]], rng: &mut impl rand::Rng) -> Vec<usize> {
        match self {
            Self::Flat(index) => vec_list.iter().map(|vec| index.vec_set.push(vec)).collect(),
            Self::HNSW(index) => index.batch_add(vec_list, rng),
        }
    }
    /// Initialize the index after loading from a file.
    pub fn init_after_load(&mut self) {
        match self {
            Self::Flat(index) => index.init_after_load(),
            Self::HNSW(index) => index.init_after_load(),
        }
    }
    /// Perform k-nearest neighbors search.
    pub fn knn(&self, query: &[f32], k: usize) -> Vec<CandidatePair> {
        match self {
            Self::Flat(index) => index.knn(query, k),
            Self::HNSW(index) => index.knn(query, k),
        }
    }
    /// Perform k-nearest neighbors search with a specified ef parameter.
    pub fn knn_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<CandidatePair> {
        match self {
            Self::Flat(index) => index.knn(query, k),
            Self::HNSW(index) => index.knn_with_ef(query, k, ef),
        }
    }
    /// Perform k-nearest neighbors search with a PQ table.
    pub fn knn_pq(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        pq_table: &PQTable<f32>,
    ) -> Vec<CandidatePair> {
        match self {
            Self::Flat(index) => index.knn_pq(query, k, ef, pq_table),
            Self::HNSW(index) => index.knn_pq(query, k, ef, pq_table),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataVecTable {
    metadata: Vec<BTreeMap<String, String>>,
    inner: OptHNSWIndex,
    pq_table: Option<PQTable<f32>>,
    #[serde(skip, default = "rand::SeedableRng::from_entropy")]
    rng: rand::rngs::StdRng,
}
impl MetadataVecTable {
    /// Create a new MetadataVecTable with the specified dimension and distance algorithm.
    pub fn new(dim: usize, dist: DistanceAlgorithm) -> Self {
        Self {
            metadata: Vec::new(),
            inner: OptHNSWIndex::new(dim, dist),
            pq_table: None,
            rng: rand::SeedableRng::from_entropy(),
        }
    }
    /// Get the number of vectors in the table.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Get the dimension of the vectors in the table.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
    /// Get the distance algorithm used by the table.
    pub fn dist(&self) -> DistanceAlgorithm {
        self.inner.dist()
    }
    /// Load a MetadataVecTable from a file.
    pub fn load(file: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(file)?;
        let buf_reader = BufReader::new(file);
        let mut index: Self = bincode::deserialize_from(buf_reader)?;
        index.inner.init_after_load();
        Ok(index)
    }
    /// Save the MetadataVecTable to a file.
    pub fn save(&self, file: impl AsRef<Path>) -> Result<()> {
        let file = File::create(file)?;
        let buf_writer = std::io::BufWriter::new(file);
        bincode::serialize_into(buf_writer, self)?;
        Ok(())
    }

    /// Add a vector with metadata to the table.
    pub fn add(&mut self, vec: Vec<f32>, metadata: BTreeMap<String, String>) {
        self.clear_pq_table();
        self.metadata.push(metadata);
        self.inner.add(&vec, &mut self.rng);
    }

    /// Add multiple vectors with metadata to the table.
    pub fn batch_add(
        &mut self,
        vec_list: Vec<Vec<f32>>,
        metadata_list: Vec<BTreeMap<String, String>>,
    ) {
        assert_eq!(vec_list.len(), metadata_list.len());
        self.clear_pq_table();
        self.metadata.extend(metadata_list);
        let vec_list: Vec<_> = vec_list.iter().map(|v| v.as_slice()).collect();
        self.inner.batch_add(&vec_list, &mut self.rng);
    }

    /// Build an HNSW index for the table.
    pub fn build_hnsw_index(&mut self, ef_construction: Option<usize>) {
        if let OptHNSWIndex::Flat(flat) = &self.inner {
            let vec_set = &flat.vec_set;
            let dist = flat.dist;
            let mut config = HNSWConfig::default();
            config.max_elements = vec_set.len();
            if let Some(ef_construction) = ef_construction {
                config.ef_construction = ef_construction;
            }
            let hnsw = HNSWIndex::build_on_vec_set(vec_set, dist, config, false, &mut self.rng);
            self.inner = OptHNSWIndex::HNSW(hnsw);
        }
    }
    /// Clear the HNSW index for the table.
    pub fn clear_hnsw_index(&mut self) {
        if let OptHNSWIndex::HNSW(hnsw) = &self.inner {
            let mut flat = FlatIndex::new(self.inner.dim(), self.inner.dist());
            flat.vec_set = hnsw.vec_set.clone();
            self.inner = OptHNSWIndex::Flat(flat);
        }
    }
    /// Check if the table has an HNSW index.
    pub fn has_hnsw_index(&self) -> bool {
        matches!(self.inner, OptHNSWIndex::HNSW(_))
    }
    /// Build a PQ table for the table.
    pub fn build_pq_table(&mut self, m: usize, train_size: usize) -> Result<()> {
        if self.len() == 0 {
            bail!("Cannot build PQ table for an empty table");
        } else if self.len() < train_size {
            bail!("Train size is larger than the number of vectors in the table");
        } else if self.dim() % m != 0 {
            bail!("PQ table requires the dimension to be a multiple of m");
        }
        let config = PQConfig {
            dist: self.dist(),
            n_bits: 4,
            m,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
            k_means_size: Some(train_size),
        };
        self.pq_table = Some(PQTable::from_vec_set(
            self.inner.vec_set(),
            config,
            &mut self.rng,
        ));
        Ok(())
    }
    /// Clear the PQ table for the table.
    pub fn clear_pq_table(&mut self) {
        self.pq_table = None;
    }
    /// Check if the table has a PQ table.
    pub fn has_pq_table(&self) -> bool {
        self.pq_table.is_some()
    }

    /// Delete vectors with metadata that match the pattern.
    pub fn delete(&mut self, pattern: &BTreeMap<String, String>) {
        fn match_metadata(
            metadata: &BTreeMap<String, String>,
            pattern: &BTreeMap<String, String>,
        ) -> bool {
            pattern.iter().all(|(k, v)| metadata.get(k) == Some(v))
        }
        self.clear_hnsw_index();
        self.clear_pq_table();
        let matches = self
            .metadata
            .iter()
            .enumerate()
            .filter(|(_, m)| match_metadata(m, pattern))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if let OptHNSWIndex::Flat(flat) = &mut self.inner {
            for i in matches.into_iter().rev() {
                self.metadata.swap_remove(i);
                flat.vec_set.swap_remove(i);
            }
        }
    }

    /// Search for the nearest vectors to a query.
    /// Return a list of (metadata, distance) pairs.
    /// - When only Flat index is available, use the Flat index for search and ignore ef.
    /// - When HNSW index is available, use the HNSW index for search.
    /// - When PQ table is available, and ef is specified, use the PQ table for search.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        upper_bound: Option<f32>,
    ) -> Vec<(BTreeMap<String, String>, f32)> {
        let results = match (ef, &self.pq_table) {
            (Some(ef), Some(pq_table)) => self.inner.knn_pq(&query, k, ef, pq_table),
            (Some(ef), _) => self.inner.knn_with_ef(&query, k, ef),
            _ => self.inner.knn(&query, k),
        };
        let upper_bound = upper_bound.unwrap_or(std::f32::INFINITY);
        results
            .into_iter()
            .filter(|p| p.distance() <= upper_bound)
            .map(|p| (self.metadata[p.index].clone(), p.distance()))
            .collect()
    }
}

impl ThreadSave for RwLock<MetadataVecTable> {
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
    pub fn insert(&mut self, key: &str, brief: VecTableBrief) {
        self.tables.insert(key.to_string(), brief);
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
    manager: thread_save::ThreadSavingManager<RwLock<MetadataVecTable>>,
    index: Arc<RwLock<MetadataVecTable>>,
    drop_signal_sender: mpsc::Sender<()>,
}
impl VecTableManager {
    /// Get the file path of a table.
    fn file_path_of(dir: impl AsRef<Path>, key: &str) -> PathBuf {
        let hex_key = sha256_hex(key.as_bytes());
        dir.as_ref().join(format!("{}.db", hex_key))
    }
    /// Create a new manager for a table.
    fn new_manager(
        file: PathBuf,
        index: &Arc<RwLock<MetadataVecTable>>,
    ) -> thread_save::ThreadSavingManager<RwLock<MetadataVecTable>> {
        thread_save::ThreadSavingManager::new(
            index.clone(),
            file,
            std::time::Duration::from_secs(30),
        )
    }
    /// Create a new table manager.
    pub fn new(
        dir: impl AsRef<Path>,
        key: &str,
        dim: usize,
        dist: DistanceAlgorithm,
        drop_signal_sender: std::sync::mpsc::Sender<()>,
    ) -> Self {
        let file = Self::file_path_of(&dir, &key);
        let index = MetadataVecTable::new(dim, dist);
        let index = Arc::new(RwLock::new(index));
        let manager = Self::new_manager(file, &index);
        manager.mark_modified();
        Self {
            manager,
            index,
            drop_signal_sender,
        }
    }
    /// Load a table manager from a file.
    pub fn load(
        dir: impl AsRef<Path>,
        key: &str,
        drop_signal_sender: std::sync::mpsc::Sender<()>,
    ) -> Result<Self> {
        let file = Self::file_path_of(&dir, &key);
        let index = MetadataVecTable::load(&file)?;
        let index = Arc::new(RwLock::new(index));
        let manager = Self::new_manager(file, &index);
        Ok(Self {
            manager,
            index,
            drop_signal_sender,
        })
    }
    pub fn brief(&self) -> VecTableBrief {
        let index = self.index.read().unwrap();
        VecTableBrief {
            dim: index.dim(),
            len: index.len(),
            dist: index.dist(),
        }
    }
    /// Add a vector with metadata to the table.
    pub fn add(&self, vec: Vec<f32>, metadata: BTreeMap<String, String>) {
        let mut index = self.index.write().unwrap();
        index.add(vec, metadata);
        self.manager.mark_modified();
    }
    /// Add multiple vectors with metadata to the table.
    pub fn batch_add(&self, vec_list: Vec<Vec<f32>>, metadata_list: Vec<BTreeMap<String, String>>) {
        let mut index = self.index.write().unwrap();
        index.batch_add(vec_list, metadata_list);
        self.manager.mark_modified();
    }
    /// Delete vectors with metadata that match the pattern.
    pub fn delete(&self, pattern: &BTreeMap<String, String>) {
        let mut index = self.index.write().unwrap();
        index.delete(pattern);
        self.manager.mark_modified();
    }
    /// Build an HNSW index for the table.
    pub fn build_hnsw_index(&self, ef_construction: Option<usize>) {
        let mut index = self.index.write().unwrap();
        index.build_hnsw_index(ef_construction);
        self.manager.mark_modified();
    }
    /// Clear the HNSW index for the table.
    pub fn clear_hnsw_index(&self) {
        let mut index = self.index.write().unwrap();
        index.clear_hnsw_index();
        self.manager.mark_modified();
    }
    /// Check if the table has an HNSW index.
    pub fn has_hnsw_index(&self) -> bool {
        let index = self.index.read().unwrap();
        index.has_hnsw_index()
    }
    /// Build a PQ table for the table.
    pub fn build_pq_table(&self, m: usize, train_size: usize) -> Result<()> {
        let mut index = self.index.write().unwrap();
        index.build_pq_table(m, train_size)?;
        self.manager.mark_modified();
        Ok(())
    }
    /// Clear the PQ table for the table.
    pub fn clear_pq_table(&self) {
        let mut index = self.index.write().unwrap();
        index.clear_pq_table();
        self.manager.mark_modified();
    }
    /// Check if the table has a PQ table.
    pub fn has_pq_table(&self) -> bool {
        let index = self.index.read().unwrap();
        index.has_pq_table()
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
    /// Get locks in the correct order.
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
    /// Get all table keys.
    pub fn get_all_keys(&self) -> Vec<String> {
        let (brief, _table) = self.get_locks_by_order();
        brief.tables.keys().cloned().collect()
    }
    /// Get all cached table keys.
    pub fn get_cached_tables(&self) -> Vec<String> {
        let (_brief, tables) = self.get_locks_by_order();
        tables.keys().cloned().collect()
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
            return self
                .get_table_with_lock(key, &mut brief, &mut tables)
                .map(|_| false);
        }
        brief.insert(&key, VecTableBrief { dim, len: 0, dist });
        self.brief_manager.mark_modified();

        let (sender, receiver) = mpsc::channel();
        let table = VecTableManager::new(&self.dir, &key, dim, dist, sender);
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
        self.brief_manager.mark_modified();
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
    /// Get information about a table.
    pub fn get_table_info(&self, key: &str) -> Option<VecTableBrief> {
        let (brief, _tables) = self.get_locks_by_order();
        brief.tables.get(key).cloned()
    }
    /// Get a table with the correct locks.
    fn get_table_with_lock(
        &self,
        key: &str,
        brief: &mut MutexGuard<'_, VecDBBrief>,
        tables: &mut MutexGuard<'_, BTreeMap<String, (mpsc::Receiver<()>, Arc<VecTableManager>)>>,
    ) -> Result<Arc<VecTableManager>> {
        if !brief.tables.contains_key(key) {
            return Err(anyhow!("Table {} not found", key));
        }
        if !tables.contains_key(key) {
            let (sender, receiver) = mpsc::channel();
            let table = VecTableManager::load(&self.dir, key, sender)?;
            brief.insert(key, table.brief());
            self.brief_manager.mark_modified();
            tables.insert(key.to_string(), (receiver, Arc::new(table)));
        }
        Ok(tables[key].1.clone())
    }

    /// Add a vector with metadata to a table.
    pub fn add(&self, key: &str, vec: Vec<f32>, metadata: BTreeMap<String, String>) -> Result<()> {
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            if let Some(info) = brief.tables.get_mut(key) {
                info.len += 1;
                self.brief_manager.mark_modified();
                if vec.len() != info.dim {
                    bail!(
                        "Dimension mismatch for table {} ({} != {})",
                        key,
                        info.dim,
                        vec.len()
                    );
                }
            }
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };
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
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            if let Some(info) = brief.tables.get_mut(key) {
                info.len += vec_list.len();
                self.brief_manager.mark_modified();
                if let Some(v) = vec_list.iter().find(|v| v.len() != info.dim) {
                    bail!(
                        "Dimension mismatch for table {} ({} != {})",
                        key,
                        info.dim,
                        v.len()
                    );
                }
            }
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };
        table.batch_add(vec_list, metadata_list);
        Ok(())
    }

    /// Build an HNSW index for a table.
    pub fn build_hnsw_index(&self, key: &str, ef_construction: Option<usize>) -> Result<()> {
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };
        table.build_hnsw_index(ef_construction);
        Ok(())
    }
    /// Clear the HNSW index for a table.
    pub fn clear_hnsw_index(&self, key: &str) -> Result<()> {
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };
        table.clear_hnsw_index();
        Ok(())
    }
    /// Check if a table has an HNSW index.
    pub fn has_hnsw_index(&self, key: &str) -> Result<bool> {
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };
        Ok(table.has_hnsw_index())
    }

    pub fn build_pq_table(&self, key: &str, m: usize, train_size: usize) -> Result<()> {
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };
        table.build_pq_table(m, train_size)?;
        Ok(())
    }
    pub fn clear_pq_table(&self, key: &str) -> Result<()> {
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };
        table.clear_pq_table();
        Ok(())
    }
    pub fn has_pq_table(&self, key: &str) -> Result<bool> {
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };
        Ok(table.has_pq_table())
    }

    /// Delete vectors with metadata that match the pattern.
    pub fn delete(&self, key: &str, pattern: &BTreeMap<String, String>) -> Result<()> {
        let (mut brief, mut tables) = self.get_locks_by_order();
        let table = self.get_table_with_lock(key, &mut brief, &mut tables)?;
        table.delete(pattern);
        brief.insert(key, table.brief());
        self.brief_manager.mark_modified();
        Ok(())
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
        let table = {
            let (mut brief, mut tables) = self.get_locks_by_order();
            self.get_table_with_lock(key, &mut brief, &mut tables)?
        };

        Ok(table.search(query, k, ef, upper_bound))
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

        Ok(())
    }
}
