use std::ops::Index;

use rand::Rng;
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
pub struct HNSWInnerConfig {
    pub dim: usize,
    pub dist: DistanceAlgorithm,
    pub max_elements: usize,
    pub m: usize,
    pub max_m0: usize,
    pub ef_construction: usize,
    pub ef: usize,
    /// inv_log_m = 1 / ln(M)$.
    ///
    /// rand_level = floor(-ln(rand_uniform(0.0,1.0)) * inv_log_m)
    pub inv_log_m: f32,
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
    /// The maximum level of the index.
    pub max_level: Option<usize>,
    /// The enter point of the search.
    pub enter_point: Option<usize>,
}
impl<T: Scalar> HNSWIndex<T> {
    fn rand_level(&self, rng: &mut impl Rng) -> usize {
        let rand_uniform: f32 = rng.gen_range(0.0..1.0);
        (-rand_uniform.ln() * self.config.inv_log_m).floor() as usize
    }
    fn _get_links(&self, vec_idx: usize, level_idx: usize) -> &[u32] {
        let len = self.links_len[vec_idx][level_idx] as usize;
        if level_idx == 0 {
            let start = vec_idx * self.config.max_m0;
            &self.level0_links[start..start + len]
        } else {
            let vec_links = &self.other_links[vec_idx];
            let start = self.config.m * (level_idx - 1);
            &vec_links[start..start + len]
        }
    }
    fn _put_links(&mut self, vec_idx: usize, level_idx: usize, links: &[u32]) {
        self.max_level = self
            .max_level
            .map_or(Some(level_idx), |level| Some(level.max(level_idx)));
        let len = links.len() as u32;
        self.links_len[vec_idx][level_idx] = len;
        if level_idx == 0 {
            let start = vec_idx * self.config.max_m0;
            self.level0_links[start..start + len as usize].clone_from_slice(links);
        } else {
            let vec_links = &mut self.other_links[vec_idx];
            let start = self.config.m * (level_idx - 1);
            vec_links[start..start + len as usize].clone_from_slice(links);
        }
    }
    /// Push a vector to the index and initialize:
    ///
    /// vec_set, level0_links, other_links, links_len,
    /// vec_level, max_level, enter_point.
    ///
    /// Returns the index of the vector. The links are initialized to empty.
    fn push_init(&mut self, vec: &[T], level: usize) -> usize {
        let idx = self.vec_set.push(vec);
        self.level0_links
            .extend_from_slice(&vec![0; self.config.max_m0]);
        self.other_links.push(vec![0; self.config.m * level]);
        self.links_len.push(vec![0; level]);
        self.vec_level.push(level as u32);

        if self.max_level.map_or(true, |max_level| level > max_level) {
            self.max_level = Some(level);
            self.enter_point = Some(idx);
        }

        idx
    }
    /// Add a vector to the index with a specific level.
    pub fn add_to_level(&mut self, vec: &[T], level: usize) -> usize {
        let idx = self.push_init(vec, level);
        if self.enter_point != Some(idx) {
            unimplemented!("HNSWIndex::add_to_level")
        }
        unimplemented!("HNSWIndex::add_to_level")
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
        let max_elements = config.max_elements;
        let m = config.M;
        assert!(
            m <= 10_000,
            " M parameter exceeds 10_000 may lead to adverse effects."
        );
        let max_m0 = m * 2;
        let ef_construction = config.ef_construction.max(m);
        let ef = 10;
        let inv_log_m = 1.0 / (m as f32).ln();

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
                m,
                max_m0,
                ef,
                inv_log_m,
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
    fn add(&mut self, vec: &[T], rng: &mut impl Rng) -> usize {
        let level = self.rand_level(rng);
        let _idx = self.add_to_level(vec, level);
        unimplemented!("HNSWIndex::add")
    }
}
