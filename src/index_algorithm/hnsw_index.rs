use std::rc::Rc;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{distance::DistanceAlgorithm, scalar::Scalar, vec_set::VecSet};

use super::{IndexAlgorithmTrait, ResponsePair};

/// The configuration of the HNSW (Hierarchical Navigable Small World) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWConfig {
    pub num_elements: usize,
    pub max_elements: usize,
    pub ef_construction: usize,
    pub M: usize,
}

pub struct HNSWIndex<T> {
    pub dist: DistanceAlgorithm,
    pub config: Rc<HNSWConfig>,
    pub vec_set: Rc<VecSet<T>>,
}

impl<T: Scalar> IndexAlgorithmTrait<T> for HNSWIndex<T> {
    type Config = HNSWConfig;

    fn from_vec_set(
        vec_set: Rc<VecSet<T>>,
        dist: DistanceAlgorithm,
        config: Rc<Self::Config>,
        rng: &mut impl Rng,
    ) -> Self {
        let _ = (vec_set, dist, config, rng);
        unimplemented!("HNSWIndex::from_vec_set")
    }
    fn knn(&self, query: &[T], k: usize) -> Vec<ResponsePair> {
        let _ = (query, k);
        unimplemented!()
    }
}
