use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{distance::DistanceAlgorithm, scalar::Scalar, vec_set::VecSet};

use super::{IndexBuilder, IndexKNN, ResponsePair};

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

impl<T: Scalar> IndexBuilder<T> for HNSWIndex<T> {
    type Config = HNSWConfig;
    fn new(_dist: DistanceAlgorithm, _config: Self::Config) -> Self {
        unimplemented!("HNSWIndex::new")
    }
    fn add(&mut self, _vec: &[T], _label: usize, _rng: &mut impl Rng) {
        unimplemented!("HNSWIndex::add")
    }
}

impl<T: Scalar> IndexKNN<T> for HNSWIndex<T> {
    fn knn(&self, _query: &[T], _k: usize) -> Vec<ResponsePair> {
        unimplemented!("HNSWIndex::knn")
    }
}
