use super::dynamic_index::DynamicIndex;
use super::thread_save::ThreadSave;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path, sync::RwLock};

use crate::{
    distance::pq_table::{PQConfig, PQTable},
    index_algorithm::{FlatIndex, HNSWConfig, HNSWIndex},
    prelude::*,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataVecTable {
    metadata: Vec<BTreeMap<String, String>>,
    inner: DynamicIndex,
    pq_table: Option<PQTable<f32>>,
    #[serde(skip, default = "rand::SeedableRng::from_entropy")]
    rng: rand::rngs::StdRng,
}
impl MetadataVecTable {
    /// Create a new MetadataVecTable with the specified dimension and distance algorithm.
    pub fn new(dim: usize, dist: DistanceAlgorithm) -> Self {
        Self {
            metadata: Vec::new(),
            inner: DynamicIndex::new(dim, dist),
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
        if let DynamicIndex::Flat(flat) = &self.inner {
            let vec_set = &flat.vec_set;
            let dist = flat.dist;
            let mut config = HNSWConfig::default();
            config.max_elements = vec_set.len();
            if let Some(ef_construction) = ef_construction {
                config.ef_construction = ef_construction;
            }
            let hnsw = HNSWIndex::build_on_vec_set(vec_set, dist, config, false, &mut self.rng);
            self.inner = DynamicIndex::HNSW(hnsw);
        }
    }
    /// Clear the HNSW index for the table.
    pub fn clear_hnsw_index(&mut self) {
        if let DynamicIndex::HNSW(hnsw) = &self.inner {
            let mut flat = FlatIndex::new(self.inner.dim(), self.inner.dist());
            flat.vec_set = hnsw.vec_set.clone();
            self.inner = DynamicIndex::Flat(flat);
        }
    }
    /// Check if the table has an HNSW index.
    pub fn has_hnsw_index(&self) -> bool {
        matches!(self.inner, DynamicIndex::HNSW(_))
    }
    /// Build a PQ table for the table.
    pub fn build_pq_table(&mut self, m: usize, train_size: usize) -> Result<()> {
        if self.len() == 0 {
            bail!("Cannot build PQ table for an empty table");
        } else if self.len() < train_size {
            bail!("Train size is larger than the number of vectors in the table");
        } else if train_size == 0 {
            bail!("Train size cannot be zero");
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
    pub fn delete(&mut self, pattern: &BTreeMap<String, String>) -> usize {
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
        let n = matches.len();
        if let DynamicIndex::Flat(flat) = &mut self.inner {
            for i in matches.into_iter().rev() {
                self.metadata.swap_remove(i);
                flat.vec_set.swap_remove(i);
            }
        }
        n
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
