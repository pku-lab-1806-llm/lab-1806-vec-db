use crate::distance::DistanceAlgorithm;

/// Different from `HNSWConfig` that is used in the configuration file.
/// This is the configuration to be directly passed to the HNSW index creation.
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct HNSWIndexConfig {
    pub dist: DistanceAlgorithm,
    pub num_elements: usize,
    pub max_elements: usize,
    pub ef_construction: usize,
    pub M: usize,
}
