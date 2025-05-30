use serde::{Deserialize, Serialize};

use crate::{
    distance::pq_table::PQTable,
    index_algorithm::{CandidatePair, FlatIndex, HNSWIndex},
    prelude::*,
    vec_set::VecSet,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DynamicIndex {
    Flat(Box<FlatIndex<f32>>),
    HNSW(Box<HNSWIndex<f32>>),
}
impl DynamicIndex {
    /// Create a new OptHNSWIndex with the specified dimension and distance algorithm.
    pub fn new(dim: usize, dist: DistanceAlgorithm) -> Self {
        Self::Flat(Box::new(FlatIndex::new(dim, dist)))
    }
    /// Get the VecSet of the index.
    pub fn vec_set(&self) -> &VecSet<f32> {
        match &self {
            DynamicIndex::Flat(flat) => &flat.vec_set,
            DynamicIndex::HNSW(hnsw) => &hnsw.vec_set,
        }
    }
    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vec_set().len()
    }
    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vec_set().is_empty()
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
