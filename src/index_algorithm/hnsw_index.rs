use serde::{Deserialize, Serialize};

use crate::{distance::DistanceAlgorithm, vec_set::VecSet};

/// The configuration of the HNSW (Hierarchical Navigable Small World) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWConfig {
    pub max_elements: usize,
    pub ef_construction: usize,
    pub M: usize,
}

pub struct HNSWIndex<T> {
    pub dist: DistanceAlgorithm,
    pub config: HNSWConfig,
    pub len: usize,
    pub vec_set: VecSet<T>,
}
impl<T> HNSWIndex<T> {
    #[allow(non_snake_case)]
    pub fn M0(&self) -> usize {
        self.config.M * 2
    }
    pub fn ef(&self) -> usize {
        10
    }
}
