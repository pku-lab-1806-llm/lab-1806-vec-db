use std::{
    borrow::Borrow,
    collections::{BTreeSet, HashSet},
    ops::Index,
    path::Path,
    thread, vec,
};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{DistanceAdapter, DistanceAlgorithm},
    index_algorithm::ResultSet,
    scalar::Scalar,
    vec_set::VecSet,
};

use super::{prelude::*, CandidatePair};

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
    pub default_ef: usize,
    /// inv_log_m = 1 / ln(M)$.
    ///
    /// rand_level = floor(-ln(rand_uniform(0.0,1.0)) * inv_log_m)
    pub inv_log_m: f32,
    /// The threshold to start batch operations.
    pub start_batch_since: usize,
    /// The inner batch size for batch operations.
    pub inner_batch_size: usize,
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
    ///
    /// `u32`: This is expensive. We can use a smaller integer type.
    pub level0_links: Vec<u32>,
    /// The other links.
    /// Dims: (vec_idx, (level_idx - 1, neighbor_idx) flattened).
    /// Size: (len, vec_level * M).
    /// Capacity: (max_elements, _).
    ///
    /// `u32`: This is expensive. We can use a smaller integer type.
    pub other_links: Vec<Vec<u32>>,
    /// The length of links.
    /// Dims: (vec_idx, level_idx).
    /// Size: (len, level).
    /// Capacity: (max_elements, _).
    pub links_len: Vec<Vec<usize>>,
    /// The level of each vector. level is 0-indexed.
    ///
    /// Capacity: max_elements.
    pub vec_level: Vec<usize>,
    /// The deleted mark of each vector.
    ///
    /// Capacity: max_elements.
    pub deleted_mark: Vec<bool>,
    /// The number of vectors marked as deleted.
    pub num_deleted: usize,
    /// The maximum level of the index.
    pub enter_level: Option<usize>,
    /// The enter point of the search.
    pub enter_point: Option<usize>,
}
impl<T: Scalar> HNSWIndex<T> {
    /// Generate a random level.
    fn rand_level(&self, rng: &mut impl Rng) -> usize {
        let rand_uniform: f32 = rng.gen_range(0.0..1.0);
        (-rand_uniform.ln() * self.config.inv_log_m).floor() as usize
    }
    /// Get the limit of the number of links at a specific level.
    fn get_links_limit(&self, level: usize) -> usize {
        if level == 0 {
            self.config.max_m0
        } else {
            self.config.m
        }
    }
    /// Get the length of the links of a vector at a specific level.
    /// - Check if the level index is within the bounds.
    /// - Check if the length does not exceed the limit.
    fn get_links_len_checked(&self, vec_idx: usize, level: usize) -> usize {
        assert!(level <= self.vec_level[vec_idx], "Index out of bounds.");
        let len = self.links_len[vec_idx][level];
        let limit = self.get_links_limit(level);
        assert!(
            len <= limit,
            "links_len[{}][{}] exceeds limit {}.",
            vec_idx,
            level,
            limit
        );
        len
    }
    /// Get the links of a vector at a specific level.
    fn get_links(&self, vec_idx: usize, level: usize) -> &[u32] {
        let len = self.get_links_len_checked(vec_idx, level);
        if level == 0 {
            let start = vec_idx * self.config.max_m0;
            &self.level0_links[start..start + len]
        } else {
            let vec_links = &self.other_links[vec_idx];
            let start = self.config.m * (level - 1);
            &vec_links[start..start + len]
        }
    }
    /// Get the links of a vector at a specific level mutably.
    fn get_links_mut(&mut self, vec_idx: usize, level: usize) -> &mut [u32] {
        let len = self.get_links_len_checked(vec_idx, level);
        if level == 0 {
            let start = vec_idx * self.config.max_m0;
            &mut self.level0_links[start..start + len]
        } else {
            let vec_links = &mut self.other_links[vec_idx];
            let start = self.config.m * (level - 1);
            &mut vec_links[start..start + len]
        }
    }
    /// Put the links of a vector at a specific level.
    fn put_links(&mut self, vec_idx: usize, level: usize, links: &[u32]) {
        assert!(level <= self.vec_level[vec_idx], "Index out of bounds.");
        self.links_len[vec_idx][level] = links.len();
        self.get_links_mut(vec_idx, level).clone_from_slice(links);
    }
    /// Try to push a link to a vector at a specific level.
    ///
    /// Or re-arrange the links heuristically if the links are full.
    fn push_link_or_heuristic(&mut self, vec_idx: usize, level: usize, new_vec_idx: usize) {
        let len = self.get_links_len_checked(vec_idx, level);
        let limit = self.get_links_limit(level);
        if len < limit {
            // Enough space to push the new link.
            self.links_len[vec_idx][level] += 1;
            *self.get_links_mut(vec_idx, level).last_mut().unwrap() = new_vec_idx as u32;
            return;
        }
        let v = &self[vec_idx];
        let dist = self.config.dist;
        let mut set = ResultSet::new(limit + 1);
        let mut add = |idx: usize| {
            let d = dist.d(v, &self[idx]);
            set.add(CandidatePair::new(idx, d));
        };
        add(new_vec_idx);
        for &neighbor in self.get_links(vec_idx, level) {
            add(neighbor as usize);
        }
        let links = set.heuristic(limit, &self.vec_set, dist);
        let links = links.iter().map(|p| p.index as u32).collect::<Vec<_>>();
        self.put_links(vec_idx, level, &links);
    }
    /// Connect new links to a vector at a specific level. (Used during adding a vector)
    fn connect_new_links(&mut self, vec_idx: usize, level: usize, candidates: ResultSet) {
        assert!(
            self.get_links_len_checked(vec_idx, level) == 0,
            "Links are not empty."
        );
        let dist = self.config.dist;
        // Initial number of neighbors is limited to M, not max_M0 even at level 0.
        let m = self.config.m;
        let neighbors = candidates.heuristic(m, &self.vec_set, dist);
        let neighbors = neighbors.iter().map(|p| p.index as u32).collect::<Vec<_>>();
        self.put_links(vec_idx, level, &neighbors);
        for neighbor in neighbors {
            self.push_link_or_heuristic(neighbor as usize, level, vec_idx);
        }
    }
    /// Push a vector to the index and initialize:
    /// vec_set, level0_links, other_links, links_len, vec_level.
    ///
    /// Returns the index of the vector. The links are initialized to empty.
    fn push_init(&mut self, vec: &[T], level: usize) -> usize {
        let idx = self.vec_set.push(vec);
        self.level0_links
            .extend_from_slice(&vec![0; self.config.max_m0]);
        self.other_links.push(vec![0; self.config.m * level]);
        self.links_len.push(vec![0; level + 1]);
        self.vec_level.push(level);
        self.deleted_mark.push(false);

        idx
    }
    /// Delete a vector from the index by setting a deleted mark.
    pub fn soft_delete(&mut self, idx: usize) {
        if self.is_soft_deleted(idx) {
            return;
        }
        self.deleted_mark[idx] = true;
        self.num_deleted += 1;
    }
    /// Restore a vector from deleted by clearing the deleted mark.
    pub fn restore_soft_deleted(&mut self, idx: usize) {
        if !self.is_soft_deleted(idx) {
            return;
        }
        self.deleted_mark[idx] = false;
        self.num_deleted -= 1;
    }
    /// Check if a vector has deleted mark.
    pub fn is_soft_deleted(&self, idx: usize) -> bool {
        self.deleted_mark[idx]
    }
    /// Search on specific level for adding a vector.
    fn search_on_level(
        &self,
        enter_point: usize,
        level: usize,
        ef: usize,
        query: &[T],
    ) -> ResultSet {
        let dist = self.config.dist;
        let mut visited = HashSet::new();
        let mut queue = BTreeSet::new();
        let mut result = ResultSet::new(ef);

        // Insert the enter point.
        visited.insert(enter_point);
        let enter_pair = CandidatePair::new(enter_point, dist.d(&self[enter_point], query));
        if !self.is_soft_deleted(enter_point) {
            result.add(enter_pair.clone());
        }
        queue.insert(enter_pair);
        while let Some(pair) = queue.pop_first() {
            if !result.check_candidate(&pair) {
                break;
            }
            for &neighbor in self.get_links(pair.index, level) {
                let new_p = neighbor as usize;
                if visited.contains(&new_p) {
                    continue;
                }
                visited.insert(new_p);
                let new_d = dist.d(&self[new_p], query);
                let new_pair = CandidatePair::new(new_p, new_d);
                if !self.is_soft_deleted(new_p) {
                    result.add(new_pair.clone());
                }
                queue.insert(new_pair);
            }
        }
        result
    }
    /// Greedy search on a specific level.
    fn greedy_search_on_level(&self, level: usize, enter_point: usize, query: &[T]) -> usize {
        let dist = self.config.dist;
        let mut cur_p = enter_point;
        let mut cur_d = dist.d(&self.vec_set[cur_p], query);
        loop {
            let mut flag = false;
            for &neighbor in self.get_links(cur_p, level) {
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
    /// - This does *NOT* search nearest vector on `target_level`.
    /// - Starts at enter_point. NEVER call this when adding a vector higher than enter_level.
    fn greedy_search_until_level(&self, target_level: usize, query: &[T]) -> usize {
        let (mut level, mut cur_p) = match (self.enter_level, self.enter_point) {
            (Some(enter_level), Some(enter_point)) => (enter_level, enter_point),
            _ => panic!("The index is empty."),
        };
        while level > target_level {
            cur_p = self.greedy_search_on_level(level, cur_p, query);
            level -= 1;
        }
        cur_p
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
    /// Batch add vectors to the index.
    fn inner_batch_add(&mut self, vec_list: &[&[T]], rng: &mut impl Rng) -> Vec<usize> {
        let n = vec_list.len();
        if self.len() < self.config.start_batch_since || n < 3 {
            // At the beginning, we choose to add vectors one by one.
            // Or if vec_list is too small, batch add is not efficient.
            return vec_list.iter().map(|vec| self.add(vec, rng)).collect();
        }
        let mut indices = Vec::with_capacity(n);
        let dist = self.config.dist;
        for &vec in vec_list.iter() {
            let level = self.rand_level(rng);
            let idx = self.push_init(vec, level);
            indices.push(idx);
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        thread::scope(|s| {
            for idx in indices.iter() {
                s.spawn(|| {
                    let idx = *idx;
                    let level = self.vec_level[idx];
                    let vec = &self[idx];
                    // Index is not null, so unwrap is safe.
                    let enter_point = self.enter_point.unwrap();
                    let enter_level = self.enter_level.unwrap();
                    let mut cached_dist = Vec::new();
                    for (&rhs_idx, &rhs_vec) in indices.iter().zip(vec_list.iter()) {
                        if rhs_idx < idx {
                            cached_dist.push((rhs_idx, dist.d(vec, rhs_vec)));
                        }
                    }

                    let mut cur_p = if level < enter_level {
                        self.greedy_search_until_level(level, vec)
                    } else {
                        enter_point
                    };
                    let ef = self.config.ef_construction;
                    for level in (0..=level.min(enter_level)).rev() {
                        let mut candidates = self.search_on_level(cur_p, level, ef, vec);
                        // Choose the nearest neighbor as the enter point of the next level.
                        cur_p = candidates.results.first().unwrap().index;
                        for &(rhs_idx, rhs_dist) in cached_dist.iter() {
                            if self.vec_level[rhs_idx] < level {
                                continue;
                            }
                            // Consider the new vector as a candidate.
                            let new_pair = CandidatePair::new(rhs_idx, rhs_dist);
                            candidates.add(new_pair);
                        }
                        sender.send((idx, level, candidates)).unwrap();
                    }
                });
            }
        });
        // Drop the sender to close the channel.
        drop(sender);
        let mut candidates = receiver.iter().collect::<Vec<_>>();
        candidates.sort_by_key(|&(idx, level, _)| (idx, level));
        for (idx, level, candidates) in candidates {
            self.connect_new_links(idx, level, candidates);
        }
        for &idx in indices.iter() {
            let level = self.vec_level[idx];
            // Index is not null, so unwrap is safe.
            if level > self.enter_level.unwrap() {
                self.enter_level = Some(level);
                self.enter_point = Some(idx);
            }
        }
        indices
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
        let default_ef = 10;
        let inv_log_m = 1.0 / (m as f32).ln();
        let start_batch_since = 1000;
        let inner_batch_size = 64;

        let vec_set = VecSet::<T>::with_capacity(dim, max_elements);
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
                default_ef,
                inv_log_m,
                start_batch_since,
                inner_batch_size,
            },
            vec_set,
            level0_links,
            other_links,
            links_len,
            vec_level,
            deleted_mark,
            num_deleted: 0,
            enter_level: None,
            enter_point: None,
        }
    }
    fn add(&mut self, vec: &[T], rng: &mut impl Rng) -> usize {
        if self.vec_set.len() >= self.config.max_elements {
            panic!("The index is full.");
        }
        let level = self.rand_level(rng);

        let idx = self.push_init(vec, level);
        let (enter_level, enter_point) = match (self.enter_level, self.enter_point) {
            (Some(enter_level), Some(enter_point)) => (enter_level, enter_point),
            _ => {
                // Initialize the enter point if the index is empty.
                self.enter_level = Some(level);
                self.enter_point = Some(idx);
                // No links to connect, so we can return early.
                return idx;
            }
        };

        let mut cur_p = if level < enter_level {
            self.greedy_search_until_level(level, vec)
        } else {
            enter_point
        };
        let ef = self.config.ef_construction;
        for level in (0..=level.min(enter_level)).rev() {
            let candidates = self.search_on_level(cur_p, level, ef, vec);
            // Choose the nearest neighbor as the enter point of the next level.
            cur_p = candidates.results.first().unwrap().index;
            self.connect_new_links(idx, level, candidates);
        }

        // Update the enter point if the new vector is closer to the query.
        if level > enter_level {
            self.enter_level = Some(level);
            self.enter_point = Some(idx);
        }
        idx
    }
    fn batch_add(&mut self, vec_list: &[&[T]], rng: &mut impl Rng) -> Vec<usize> {
        vec_list
            .chunks(self.config.inner_batch_size)
            .flat_map(|chunk| self.inner_batch_add(chunk, rng))
            .collect()
    }
    fn batch_add_process(&mut self, vec_list: &[&[T]], rng: &mut impl Rng) -> Vec<usize> {
        use indicatif::{ProgressIterator, ProgressStyle};

        let style = ProgressStyle::default_bar()
                    .template("[{elapsed}|ETA {eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len} batches, {per_sec:2}")
                    .unwrap()
                    .progress_chars("##-");

        vec_list
            .chunks(self.config.inner_batch_size)
            .progress_with_style(style)
            .flat_map(|chunk| self.inner_batch_add(chunk, rng))
            .collect()
    }
    fn build_on_vec_set(
        vec_set: impl Borrow<VecSet<T>>,
        dist: DistanceAlgorithm,
        config: Self::Config,
        process_bar: bool,
        rng: &mut impl Rng,
    ) -> Self {
        let vec_set = vec_set.borrow();
        let mut index = Self::new(vec_set.dim(), dist, config);
        let vec_refs: Vec<&[T]> = vec_set.iter().collect();
        if process_bar {
            index.batch_add_process(&vec_refs, rng);
        } else {
            index.batch_add(&vec_refs, rng);
        }
        index
    }
}

impl<T: Scalar> IndexKNN<T> for HNSWIndex<T> {
    fn knn(&self, query: &[T], k: usize) -> Vec<CandidatePair> {
        self.knn_with_ef(query, k, self.config.default_ef)
    }
}
impl<T: Scalar> IndexKNNWithEf<T> for HNSWIndex<T> {
    fn knn_with_ef(&self, query: &[T], k: usize, ef: usize) -> Vec<CandidatePair> {
        if self.len() == 0 {
            return Vec::new();
        }
        let ef = ef.max(k);
        let level = 0;
        let enter_point = self.greedy_search_until_level(level, query);
        let result = self.search_on_level(enter_point, level, ef, query);
        result.into_sorted_vec_limit(k)
    }
}
impl<T: Scalar> IndexSerde for HNSWIndex<T> {
    /// Save the index to the file.
    ///
    /// For HNSWIndex, the capacity should be reset after loading.
    fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut index: Self = bincode::deserialize_from(reader)?;

        let all_len = [
            index.vec_set.len(),
            index.level0_links.len(),
            index.vec_level.len(),
            index.other_links.len(),
            index.links_len.len(),
            index.deleted_mark.len(),
        ];
        if !all_len.windows(2).all(|w| w[0] == w[1]) {
            anyhow::bail!("The lengths of the index data are not consistent. See `load_with_external_vec_set` for loading with external vec_set, if you have saved the index without vec_set.");
        }

        // The capacity is not as expected after deserialization.
        let new_max_elements = index.config.max_elements;
        index.config.max_elements = index.len();
        // Reset the capacity.
        index.reset_max_elements(new_max_elements, true);

        Ok(index)
    }
}
impl<T: Scalar> IndexSerdeExternalVecSet<T> for HNSWIndex<T> {
    fn save_without_vec_set(mut self, path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // Move the vec_set out of the index.
        let vec_set = self.vec_set;
        self.vec_set = VecSet::new(vec_set.dim(), vec![]);

        // Call the original save method.
        self.save(path)?;

        // Restore the vec_set.
        self.vec_set = vec_set;
        Ok(self)
    }
    fn load_with_external_vec_set(
        path: impl AsRef<Path>,
        vec_set: VecSet<T>,
    ) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut index: Self = bincode::deserialize_from(reader)?;

        // Restore the vec_set at first.
        index.vec_set = vec_set;
        // The capacity is not as expected after deserialization.
        let new_max_elements = index.config.max_elements;
        index.config.max_elements = index.len();
        // Reset the capacity.
        index.reset_max_elements(new_max_elements, true);

        Ok(index)
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use anyhow::Result;
    use rand::SeedableRng;

    use crate::{
        config::{DBConfig, IndexAlgorithmConfig},
        index_algorithm::{linear_index, IndexFromVecSet},
    };

    use super::*;

    #[test]
    pub fn hnsw_index_test() -> Result<()> {
        fn clip_msg(s: &str) -> String {
            if s.len() > 100 {
                format!("{}...", &s[..100])
            } else {
                s.to_string()
            }
        }
        let file_path = "config/db_config.toml";
        let config = DBConfig::load_from_toml_file(file_path)?;
        println!("Loaded config: {:#?}", config);
        let raw_vec_set = VecSet::<f32>::load_with(&config.vec_data)?;
        let dist = DistanceAlgorithm::L2Sqr;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let config = match config.algorithm {
            IndexAlgorithmConfig::HNSW(config) => config,
            _ => panic!("Testing HNSWIndex with non-HNSW config."),
        };

        // Limit the dimension for testing.
        let clipped_dim = raw_vec_set.dim().min(12);

        // Clipped vector set.
        let mut vec_set = VecSet::with_capacity(clipped_dim, raw_vec_set.len());
        for vec in raw_vec_set.iter() {
            vec_set.push(&vec[..clipped_dim]);
        }

        // Test the HNSWIndex by comparing with LinearIndex.
        let index = HNSWIndex::<f32>::build_on_vec_set(&vec_set, dist, config, false, &mut rng);
        let linear_index = linear_index::LinearIndex::from_vec_set(vec_set, dist, (), &mut rng);

        // Save and load the index. >>>>
        println!("Saving the index...");
        let path = "data/hnsw_index.tmp.bin";
        let vec_set = index.save_without_vec_set(path)?.vec_set;

        let index = HNSWIndex::<f32>::load_with_external_vec_set(path, vec_set)?;
        println!("Loaded the index.");
        // <<<< Save and load the index.

        let k = 6;
        let query_index = 200;

        println!("Query Index: {}", query_index);
        println!(
            "Query Vector: {}",
            clip_msg(&format!("{:?}", &index[query_index]))
        );

        let result = index.knn(&index[query_index], k);
        let linear_result = linear_index.knn(&linear_index[query_index], k);

        for (res, l_res) in result.iter().zip(linear_result.iter()) {
            println!("Index: {}, Distance: {}", res.index, res.distance);
            println!("Vector: {}", clip_msg(&format!("{:?}", &index[res.index])));
            assert_eq!(res.index, l_res.index, "Index mismatch");
        }
        assert_eq!(result.len(), k.min(index.len()));

        assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));

        fs::remove_file(path)?;
        Ok(())
    }
}
