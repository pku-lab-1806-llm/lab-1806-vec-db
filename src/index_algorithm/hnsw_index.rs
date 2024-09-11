use std::ops::Index;

use serde::{Deserialize, Serialize};

use crate::{distance::DistanceAlgorithm, scalar::Scalar, vec_set::VecSet};

use super::{IndexBuilder, IndexIter};

/// The configuration of the HNSW (Hierarchical Navigable Small World) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWConfig {
    pub max_elements: usize,
    pub ef_construction: usize,
    pub M: usize,
}

/// The inner configuration of the HNSW algorithm.
/// Contains more computed values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWInnerConfig {
    pub dim: usize,
    pub dist: DistanceAlgorithm,
    pub max_elements: usize,
    pub ef_construction: usize,
    pub M: usize,
    pub max_M0: usize,
    pub ef: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWIndex<T> {
    /// The configuration of the HNSW algorithm.
    ///
    /// Inner configuration contains more computed values.
    pub config: HNSWInnerConfig,
    /// The vector set. (capacity = max_elements)
    pub vec_set: VecSet<T>,
    /// The level 0 links.
    /// Dims: (vec_idx, neighbor_idx) flattened.
    /// Size: len * max_M0.
    /// Capacity: max_elements * max_M0.
    pub level0_links: Vec<u32>,
    /// The other links.
    /// Dims: (vec_idx, (level_idx - 1, neighbor_idx) flattened).
    /// Size: (len, level * M).
    /// Capacity: (max_elements, _).
    pub other_links: Vec<Vec<u32>>,
    /// The length of links.
    /// Dims: (vec_idx, level_idx).
    /// Size: (len, level).
    /// Capacity: (max_elements, _).
    pub links_len: Vec<Vec<u32>>,
    /// The level of each vector. level is 0-indexed.
    pub vec_level: Vec<u32>,
    pub max_level: Option<usize>,
    pub enter_point: Option<usize>,
}
impl<T: Scalar> HNSWIndex<T> {
    fn _get_links(&self, vec_idx: usize, level_idx: usize) -> &[u32] {
        let len = self.links_len[vec_idx][level_idx] as usize;
        if level_idx == 0 {
            let start = vec_idx * self.config.max_M0;
            &self.level0_links[start..start + len]
        } else {
            let vec_links = &self.other_links[vec_idx];
            let start = self.config.M * (level_idx - 1);
            &vec_links[start..start + len]
        }
    }
    fn _put_links(&mut self, vec_idx: usize, level_idx: usize, links: &[u32]) {
        let len = links.len() as u32;
        self.links_len[vec_idx][level_idx] = len;
        if level_idx == 0 {
            let start = vec_idx * self.config.max_M0;
            self.level0_links[start..start + len as usize].clone_from_slice(links);
        } else {
            let vec_links = &mut self.other_links[vec_idx];
            let start = self.config.M * (level_idx - 1);
            vec_links[start..start + len as usize].clone_from_slice(links);
        }
    }
    fn _push_only(&mut self, vec: &[T], level: usize) -> usize {
        let idx = self.vec_set.push(vec);
        self.level0_links
            .extend_from_slice(&vec![0; self.config.max_M0]);
        let links = vec![0; self.config.M * level];
        self.other_links.push(links);
        self.vec_level.push(level as u32);

        idx
    }
}
impl<T: Scalar> Index<usize> for HNSWIndex<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec_set[index]
    }
}
impl<T: Scalar> IndexIter<T> for HNSWIndex<T> {
    fn len(&self) -> usize {
        self.vec_set.len()
    }
}

impl<T: Scalar> IndexBuilder<T> for HNSWIndex<T> {
    type Config = HNSWConfig;
    fn dim(&self) -> usize {
        self.config.dim
    }
    fn new(dim: usize, dist: DistanceAlgorithm, config: Self::Config) -> Self {
        #[allow(non_snake_case)]
        let M = config.M;
        assert!(
            M <= 10_000,
            " M parameter exceeds 10_000 which may lead to adverse effects."
        );
        #[allow(non_snake_case)]
        let max_M0 = 2 * M;
        let max_elements = config.max_elements;

        let ef = 10;
        let ef_construction = config.ef_construction.max(config.M);

        let vec_set = VecSet::<T>::with_capacity(max_elements, dim);
        let level0_links = Vec::with_capacity(max_elements);
        let vec_level = Vec::with_capacity(max_elements);
        let other_links = Vec::with_capacity(max_elements);
        let links_len = Vec::with_capacity(max_elements);

        Self {
            config: HNSWInnerConfig {
                dim,
                dist,
                max_elements,
                ef_construction,
                M,
                max_M0,
                ef,
            },
            vec_set,
            level0_links,
            other_links,
            links_len,
            vec_level,
            max_level: None,
            enter_point: None,
        }
    }
    fn add(&mut self, _vec: &[T], _rng: &mut impl rand::Rng) -> usize {
        unimplemented!("HNSWIndex::add")
    }
}
