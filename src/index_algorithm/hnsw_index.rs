use std::ops::Index;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{DistanceAdapter, DistanceAlgorithm},
    scalar::Scalar,
    vec_set::VecSet,
};

use super::{CandidatePair, IndexBuilder, IndexIter, IndexKNN};

/// The configuration of the HNSW (Hierarchical Navigable Small World) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWConfig {
    /// The dimension of the vectors.
    pub max_elements: usize,
    /// The number of neighbors to search during construction.
    pub ef_construction: usize,
    /// The number of neighbors to keep for each vector. (M)
    pub M: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResultSet {}

/// The inner configuration of the HNSW algorithm.
/// Contains more computed values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWInnerConfig {
    /// The dimension of the vectors.
    pub dim: usize,
    /// The distance algorithm.
    pub dist: DistanceAlgorithm,
    /// The maximum number of elements in the index.
    pub max_elements: usize,
    /// The number of neighbors to keep for each vector. (M)
    pub m: usize,
    /// The number of neighbors to keep for each vector at level 0. (2 * M)
    pub max_m0: usize,
    /// The number of neighbors to search during construction.
    pub ef_construction: usize,
    /// The number of neighbors to check during search.
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
    ///
    /// Capacity: max_elements.
    pub vec_level: Vec<u32>,
    /// The deleted mark of each vector.
    ///
    /// Capacity: max_elements.
    pub deleted_mark: Vec<bool>,
    /// The number of vectors marked as deleted.
    pub num_deleted: usize,
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
    fn get_links(&self, vec_idx: usize, level_idx: usize) -> &[u32] {
        assert!(
            level_idx <= self.vec_level[vec_idx] as usize,
            "Index out of bounds."
        );
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
        self.deleted_mark.push(false);

        if self.max_level.map_or(true, |max_level| level > max_level) {
            self.max_level = Some(level);
            self.enter_point = Some(idx);
        }

        idx
    }
    /// Mark a vector as deleted.
    pub fn mark_deleted(&mut self, idx: usize) {
        if self.is_marked_deleted(idx) {
            return;
        }
        self.deleted_mark[idx] = true;
        self.num_deleted += 1;
    }
    /// Check if a vector is marked as deleted.
    pub fn is_marked_deleted(&self, idx: usize) -> bool {
        self.deleted_mark[idx]
    }
    /// Search the base layer (level 0).
    fn _search_level0(&self, _enter_point: usize, _query: &[T], _ef: usize) -> Vec<usize> {
        unimplemented!("HNSWIndex::search_base_layer")
    }
    /// Greedy search on a specific level.
    fn greedy_search_on_level(&self, level_idx: usize, enter_point: usize, query: &[T]) -> usize {
        let dist = self.config.dist;
        let mut cur_p = enter_point;
        let mut cur_d = dist.d(&self.vec_set[cur_p], query);
        loop {
            let mut flag = false;
            for &neighbor in self.get_links(cur_p, level_idx) {
                let new_p = neighbor as usize;
                let new_d = dist.d(&self.vec_set[new_p], query);
                if new_d < cur_d {
                    cur_d = new_d;
                    cur_p = new_p;
                    flag = true;
                }
            }
            if !flag {
                break;
            }
        }
        cur_p
    }
    /// Greedy search until reaching the base layer.
    ///
    /// Note: This does *NOT* search on level 0. So the result is *NOT* the final result.
    pub fn greedy_search_until_level0(&self, query: &[T]) -> usize {
        let (mut level_idx, mut cur_p) = match (self.max_level, self.enter_point) {
            (Some(max_level), Some(enter_point)) => (max_level, enter_point),
            _ => panic!("The index is empty."),
        };
        while level_idx > 0 {
            cur_p = self.greedy_search_on_level(level_idx, cur_p, query);
            level_idx -= 1;
        }
        cur_p
    }
    /// Add a vector to the index with a specific level.
    pub fn add_to_level(&mut self, vec: &[T], level: usize) -> usize {
        let idx = self.push_init(vec, level);
        if self.enter_point != Some(idx) {
            unimplemented!("HNSWIndex::add_to_level")
        }
        unimplemented!("HNSWIndex::add_to_level")
    }
    /// Reset the capacity of the index.
    /// `exact` is true if the capacity should be exactly `new_max_elements`.
    ///
    /// This should reset vec_set, level0_links, vec_level, other_links, links_len, deleted_mark.
    /// Commonly used after loading the index from a file.
    pub fn reset_max_elements(&mut self, new_max_elements: usize, exact: bool) {
        assert!(
            new_max_elements >= self.config.max_elements,
            "The new capacity should be larger than the current capacity."
        );
        self.config.max_elements = new_max_elements;
        let num_reserve = new_max_elements - self.vec_set.len();
        if exact {
            self.vec_set.reserve_exact(num_reserve);
            self.level0_links
                .reserve_exact(num_reserve * self.config.max_m0);
            self.vec_level.reserve_exact(num_reserve);
            self.other_links.reserve_exact(num_reserve);
            self.links_len.reserve_exact(num_reserve);
            self.deleted_mark.reserve_exact(num_reserve);
        } else {
            self.vec_set.reserve(num_reserve);
            self.level0_links.reserve(num_reserve * self.config.max_m0);
            self.vec_level.reserve(num_reserve);
            self.other_links.reserve(num_reserve);
            self.links_len.reserve(num_reserve);
            self.deleted_mark.reserve(num_reserve);
        }
    }
}
impl<T: Scalar> Index<usize> for HNSWIndex<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec_set[index]
    }
}
impl<T: Scalar> IndexIter<T> for HNSWIndex<T> {
    fn dim(&self) -> usize {
        self.config.dim
    }
    fn len(&self) -> usize {
        self.vec_set.len()
    }
}

impl<T: Scalar> IndexBuilder<T> for HNSWIndex<T> {
    type Config = HNSWConfig;
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
        let deleted_mark = Vec::with_capacity(max_elements);

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
            deleted_mark,
            num_deleted: 0,
            max_level: None,
            enter_point: None,
        }
    }
    fn add(&mut self, vec: &[T], rng: &mut impl Rng) -> usize {
        if self.vec_set.len() >= self.config.max_elements {
            panic!("The index is full.");
        }
        let level = self.rand_level(rng);
        let _idx = self.add_to_level(vec, level);
        unimplemented!("HNSWIndex::add")
    }
}

impl<T: Scalar> IndexKNN<T> for HNSWIndex<T> {
    fn knn(&self, _query: &[T], _k: usize) -> Vec<CandidatePair> {
        unimplemented!("HNSWIndex::knn")
    }
}
