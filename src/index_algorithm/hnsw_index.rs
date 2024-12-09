use std::{
    borrow::Borrow,
    collections::{BTreeSet, HashSet},
    ops::Index,
    path::Path,
    vec,
};

use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    distance::{pq_table::PQTable, DistanceAdapter, DistanceAlgorithm},
    index_algorithm::ResultSet,
    scalar::Scalar,
    vec_set::VecSet,
};

use super::{prelude::*, CandidatePair};

/// The default maximum number of elements in the index.
///
/// 0 means auto reallocation.
pub fn default_max_elements() -> usize {
    0
}

/// The default number of neighbors to search during construction.
pub fn default_ef_construction() -> usize {
    200
}

/// The default number of neighbors to keep for each vector.
#[allow(non_snake_case)]
pub fn default_M() -> usize {
    16
}

/// The configuration of the HNSW (Hierarchical Navigable Small World) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWConfig {
    /// The initial capacity of the index.
    ///
    /// More vectors can be added safely with auto re-allocation.
    /// Set it to the maximum expected number of vectors for better performance.
    /// But if you are not sure, you can set it to a smaller number, even 0.
    ///
    /// For loaded indexes, this does not affect the capacity.
    /// If you really want to ensure the capacity, call [HNSWIndex::init_capacity_after_load].
    #[serde(default = "default_max_elements")]
    pub max_elements: usize,
    /// The number of neighbors to search during construction.
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,
    /// The number of neighbors to keep for each vector.
    #[serde(default = "default_M")]
    pub M: usize,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            max_elements: default_max_elements(),
            ef_construction: default_ef_construction(),
            M: default_M(),
        }
    }
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWIndex<T> {
    /// The configuration of the HNSW algorithm.
    ///
    /// Inner configuration contains more computed values.
    pub(crate) config: HNSWInnerConfig,
    /// The vector set. (capacity = max_elements)
    pub(crate) vec_set: VecSet<T>,
    /// The level 0 links.
    /// Dims: (vec_idx, neighbor_idx) flattened.
    /// Size: len * max_M0.
    /// Capacity: max_elements * max_M0.
    ///
    /// `u32`: This is expensive. We can use a smaller integer type.
    pub(crate) level0_links: Vec<u32>,
    /// The other links.
    /// Dims: (vec_idx, (level_idx - 1, neighbor_idx) flattened).
    /// Size: (len, vec_level * M).
    /// Capacity: (max_elements, _).
    ///
    /// `u32`: This is expensive. We can use a smaller integer type.
    pub(crate) other_links: Vec<Vec<u32>>,
    /// The length of links.
    /// Dims: (vec_idx, level_idx).
    /// Size: (len, level).
    /// Capacity: (max_elements, _).
    pub(crate) links_len: Vec<Vec<usize>>,
    /// The level of each vector. level is 0-indexed.
    ///
    /// Capacity: max_elements.
    pub(crate) vec_level: Vec<usize>,
    /// The number of vectors marked as deleted.
    pub(crate) num_deleted: usize,
    /// The maximum level of the index.
    pub(crate) enter_level: Option<usize>,
    /// The enter point of the search.
    pub(crate) enter_point: Option<usize>,
    /// The norm cache of the vectors.
    ///
    /// - For L2Sqr, cache dot_product(a, a).
    /// - For Cosine, cache vec_norm(a).
    #[serde(skip)]
    pub(crate) dist_cache: Vec<f32>,
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
    /// Or re-arrange the links heuristically if the links are full.
    fn arrange_links(&self, vec_idx: usize, level: usize, new_vec_idx: usize) -> Vec<u32> {
        let limit = self.get_links_limit(level);
        let mut links = self.get_links(vec_idx, level).to_vec();
        links.push(new_vec_idx as u32);
        if links.len() <= limit {
            return links;
        }
        let dist_fn = |a, b| self.inner_dist_fn(a, b);
        let mut set = ResultSet::new(limit + 1);
        for index in links.iter().map(|&idx| idx as usize) {
            let distance = dist_fn(vec_idx, index);
            set.add(CandidatePair::new(index, distance));
        }
        let links = set.heuristic(limit, dist_fn);
        links.iter().map(|p| p.index as u32).collect()
    }
    /// Arrange links in parallel.
    /// arranges: (vec_idx, level, new_vec_idx) need to be re-arranged.
    fn arrange_parallel(&mut self, arranges: &Vec<(usize, usize, usize)>) {
        let links = arranges
            .par_iter()
            .map(|&(vec_idx, level, new_vec_idx)| {
                let links = self.arrange_links(vec_idx, level, new_vec_idx);
                (vec_idx, level, links)
            })
            .collect::<Vec<_>>();

        for (vec_idx, level, links) in links {
            self.put_links(vec_idx, level, &links);
        }
    }
    /// Connect new links to a vector at a specific level.
    /// Returns (vec_idx, level, new_vec_idx) need to be re-arranged.
    fn heuristic_for_new(
        &mut self,
        vec_idx: usize,
        level: usize,
        candidates: ResultSet,
    ) -> Vec<(usize, usize, usize)> {
        assert!(
            self.get_links_len_checked(vec_idx, level) == 0,
            "Links are not empty."
        );
        // Initial number of neighbors is limited to M, not max_M0 even at level 0.
        let m = self.config.m;
        let neighbors = candidates.heuristic(m, |a, b| self.inner_dist_fn(a, b));
        let neighbors = neighbors.iter().map(|p| p.index as u32).collect::<Vec<_>>();
        self.put_links(vec_idx, level, &neighbors);
        return neighbors
            .iter()
            .map(|&neighbor| (neighbor as usize, level, vec_idx))
            .collect();
    }
    /// Push a vector to the index and initialize:
    /// vec_set, level0_links, other_links, links_len, vec_level, deleted_mark, norm_cache.
    ///
    /// Returns the index of the vector. The links are initialized to empty.
    fn push_init(&mut self, vec: &[T], level: usize) -> usize {
        let idx = self.vec_set.push(vec);
        self.level0_links
            .extend_from_slice(&vec![0; self.config.max_m0]);
        self.other_links.push(vec![0; self.config.m * level]);
        self.links_len.push(vec![0; level + 1]);
        self.vec_level.push(level);
        self.dist_cache.push(match self.config.dist {
            DistanceAlgorithm::L2Sqr => T::dot_product(vec, vec),
            DistanceAlgorithm::Cosine => T::vec_norm(vec),
        });
        idx
    }

    fn search_on_level_fn(
        &self,
        enter_point: usize,
        level: usize,
        ef: usize,
        dist_fn: &impl Fn(usize) -> f32,
    ) -> ResultSet {
        let mut visited = HashSet::new();
        let mut queue = BTreeSet::new();
        let mut result = ResultSet::new(ef);

        // Insert the enter point.
        visited.insert(enter_point);
        let enter_pair = CandidatePair::new(enter_point, dist_fn(enter_point));
        result.add(enter_pair.clone());
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
                let new_d = dist_fn(new_p);
                let new_pair = CandidatePair::new(new_p, new_d);
                result.add(new_pair.clone());
                queue.insert(new_pair);
            }
        }
        result
    }
    /// Search on specific level for adding a vector.
    fn search_on_level(
        &self,
        enter_point: usize,
        level: usize,
        ef: usize,
        query: &[T],
    ) -> ResultSet {
        let query_cache = self.config.dist.dist_cache(query);
        let dist_fn = |idx| self.dist_with_cache(idx, query, query_cache);

        self.search_on_level_fn(enter_point, level, ef, &dist_fn)
    }
    /// Greedy search on a specific level.
    fn greedy_search_on_level_fn(
        &self,
        level: usize,
        enter_point: usize,
        dist_fn: &impl Fn(usize) -> f32,
    ) -> usize {
        let mut cur_p = enter_point;
        let mut cur_d = dist_fn(cur_p);
        loop {
            let mut flag = false;
            for &neighbor in self.get_links(cur_p, level) {
                let new_p = neighbor as usize;
                let new_d = dist_fn(new_p);
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
    fn greedy_search_until_level_fn(
        &self,
        target_level: usize,
        dist_fn: &impl Fn(usize) -> f32,
    ) -> usize {
        let (mut level, mut cur_p) = match (self.enter_level, self.enter_point) {
            (Some(enter_level), Some(enter_point)) => (enter_level, enter_point),
            _ => panic!("The index is empty."),
        };
        while level > target_level {
            cur_p = self.greedy_search_on_level_fn(level, cur_p, dist_fn);
            level -= 1;
        }
        cur_p
    }
    fn dist_with_cache(&self, idx: usize, query: &[T], query_cache: f32) -> f32 {
        self.config
            .dist
            .d(&(&self[idx], self.dist_cache[idx]), &(query, query_cache))
    }
    fn inner_dist_fn(&self, idx0: usize, idx1: usize) -> f32 {
        self.dist_with_cache(idx0, &self[idx1], self.dist_cache[idx1])
    }
    /// Greedy search until reaching the base layer.
    ///
    /// - This does *NOT* search nearest vector on `target_level`.
    /// - Starts at enter_point. NEVER call this when adding a vector higher than enter_level.
    fn greedy_search_until_level(&self, target_level: usize, query: &[T]) -> usize {
        let query_cache = self.config.dist.dist_cache(query);
        let dist_fn = |idx| self.dist_with_cache(idx, query, query_cache);
        self.greedy_search_until_level_fn(target_level, &dist_fn)
    }
    /// Ensure the norm cache for cosine distance.
    ///
    /// Call this after loading the index.
    pub fn init_dist_cache_after_load(&mut self) {
        if self.dist_cache.is_empty() {
            self.dist_cache = self
                .vec_set
                .iter()
                .map(|v| self.config.dist.dist_cache(v))
                .collect();
        }
    }
    pub fn capacity(&self) -> usize {
        self.vec_set.capacity()
    }
    /// Reserve additional capacity for the index.
    pub fn reserve(&mut self, additional: usize) {
        self.vec_set.reserve(additional);
        self.level0_links.reserve(additional * self.config.max_m0);
        self.vec_level.reserve(additional);
        self.other_links.reserve(additional);
        self.links_len.reserve(additional);
    }
    fn next_batch_size(&self) -> usize {
        let m = self.config.start_batch_since;
        if self.len() < m {
            1
        } else {
            (self.len() / 10).min(m)
        }
    }
    /// Add vectors in a chunk in parallel.
    fn add_parallel(&mut self, vec_list: &[&[T]], rng: &mut impl Rng) -> Vec<usize> {
        let n = vec_list.len();
        if self.len() < self.config.start_batch_since || n == 1 {
            // At the beginning, we choose to add vectors one by one.
            // Or if vec_list is too small, batch add is not efficient.
            return vec_list.iter().map(|vec| self.add(vec, rng)).collect();
        }
        self.reserve(n);
        let mut indices = Vec::with_capacity(n);
        for &vec in vec_list.iter() {
            let level = self.rand_level(rng);
            let idx = self.push_init(vec, level);
            indices.push(idx);
        }
        let candidates = indices
            .par_iter()
            .map(|&idx| {
                let vec = &self[idx];
                let level = self.vec_level[idx];
                let enter_point = self.enter_point.unwrap();
                let enter_level = self.enter_level.unwrap();
                let mut cur_p = if level < enter_level {
                    self.greedy_search_until_level(level, vec)
                } else {
                    enter_point
                };
                let ef = self.config.ef_construction;
                let mut result = Vec::new();
                for level in (0..=level.min(enter_level)).rev() {
                    let mut candidates = self.search_on_level(cur_p, level, ef, vec);
                    // Choose the nearest neighbor as the enter point of the next level.
                    cur_p = candidates.results.first().unwrap().index;
                    indices
                        .iter()
                        .filter(|&&rhs_idx| rhs_idx < idx && self.vec_level[rhs_idx] >= level)
                        .for_each(|&rhs_idx| {
                            let d = self.inner_dist_fn(idx, rhs_idx);
                            candidates.add(CandidatePair::new(rhs_idx, d));
                        });
                    result.push((level, candidates));
                }
                (idx, result)
            })
            .collect::<Vec<_>>();
        for (idx, levels) in candidates {
            let arranges = levels
                .into_iter()
                .flat_map(|(level, candidates)| self.heuristic_for_new(idx, level, candidates))
                .collect::<Vec<_>>();

            self.arrange_parallel(&arranges);
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
    /// Split the batch add into smaller chunks to add in parallel.
    fn inner_batch_add(
        &mut self,
        vec_list: &[&[T]],
        rng: &mut impl Rng,
        pos_callback: impl Fn(usize) -> (),
    ) -> Vec<usize> {
        let mut cur = 0;
        let n = vec_list.len();
        let mut result = Vec::with_capacity(n);
        while cur < vec_list.len() {
            let next = (cur + self.next_batch_size()).min(n);
            result.extend(self.add_parallel(&vec_list[cur..next], rng));
            cur = next;
            pos_callback(cur);
        }
        result
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
        let max_m = 10_000;
        let m = config.M.min(max_m);
        if config.M > max_m {
            eprintln!("M parameter exceeds 10_000 may lead to adverse effects.");
        }
        let max_m0 = m * 2;
        let ef_construction = config.ef_construction.max(max_m0);
        let default_ef = ef_construction / 2;
        let inv_log_m = 1.0 / (m as f32).ln();
        let start_batch_since = 1000;

        let vec_set: VecSet<T> = VecSet::<T>::with_capacity(dim, max_elements);
        let level0_links = Vec::with_capacity(max_elements * max_m0);
        let vec_level = Vec::with_capacity(max_elements);
        let other_links = Vec::with_capacity(max_elements);
        let links_len = Vec::with_capacity(max_elements);
        let dist_cache = Vec::with_capacity(max_elements);

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
            },
            vec_set,
            level0_links,
            other_links,
            links_len,
            vec_level,
            num_deleted: 0,
            enter_level: None,
            enter_point: None,
            dist_cache,
        }
    }
    fn add(&mut self, vec: &[T], rng: &mut impl Rng) -> usize {
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
            let arranges = self.heuristic_for_new(idx, level, candidates);
            self.arrange_parallel(&arranges);
        }

        // Update the enter point if the new vector is closer to the query.
        if level > enter_level {
            self.enter_level = Some(level);
            self.enter_point = Some(idx);
        }
        idx
    }
    fn batch_add(&mut self, vec_list: &[&[T]], rng: &mut impl Rng) -> Vec<usize> {
        self.inner_batch_add(vec_list, rng, |_| {})
    }
    fn batch_add_process(&mut self, vec_list: &[&[T]], rng: &mut impl Rng) -> Vec<usize> {
        use indicatif::ProgressStyle;

        let style = ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}|ETA {eta}] {bar:40.cyan/blue} {pos:<8}/{len}|{per_sec:>15}",
            )
            .unwrap()
            .progress_chars("##-");

        let n = vec_list.len();
        let progress_bar = indicatif::ProgressBar::new(n as u64);
        progress_bar.set_style(style);
        let pos_callback = |cur| {
            progress_bar.set_position(cur as u64);
        };

        self.inner_batch_add(vec_list, rng, pos_callback)
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
    fn set_default_ef(&mut self, ef: usize) {
        assert!(ef > 0, "The search radius should be positive.");
        self.config.default_ef = ef;
    }
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
    fn init_after_load(&mut self) {
        // Do not init the capacity here.
        // In most case we load the index from a file for searching.

        // Init the norm cache for cosine distance.
        self.init_dist_cache_after_load();
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

        // Restore the vec_set
        index.vec_set = vec_set;
        index.init_after_load();
        Ok(index)
    }
}

impl<T: Scalar> IndexPQ<T> for HNSWIndex<T> {
    fn knn_pq(
        &self,
        query: &[T],
        k: usize,
        ef: usize,
        pq_table: &PQTable<T>,
    ) -> Vec<CandidatePair> {
        if self.len() == 0 {
            return Vec::new();
        }
        let dist = self.config.dist;
        assert_eq!(dist, pq_table.config.dist, "Distance algorithm mismatch.");
        let lookup = pq_table.create_lookup(query);
        let es = &pq_table.encoded_vec_set;
        let dist_fn = |idx| dist.d(&es[idx], &lookup);
        let ef = ef.max(k);
        let level = 0;
        let enter_point = self.greedy_search_until_level_fn(level, &dist_fn);
        let result = self.search_on_level_fn(enter_point, level, ef, &dist_fn);

        let query_cache = self.config.dist.dist_cache(query);
        let index_dist = |idx| self.dist_with_cache(idx, query, query_cache);
        result.pq_resort(k, index_dist)
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use anyhow::Result;
    use rand::SeedableRng;

    use crate::{
        config::VecDataConfig,
        index_algorithm::{linear_index, IndexFromVecSet},
    };

    use super::*;

    #[test]
    pub fn hnsw_index_test() -> Result<()> {
        hnsw_index_test_with_dist(DistanceAlgorithm::L2Sqr)?;
        hnsw_index_test_with_dist(DistanceAlgorithm::Cosine)?;

        Ok(())
    }
    fn hnsw_index_test_with_dist(dist: DistanceAlgorithm) -> Result<()> {
        fn clip_msg(s: &str) -> String {
            if s.len() > 100 {
                format!("{}...", &s[..100])
            } else {
                s.to_string()
            }
        }
        println!("Distance Algorithm: {:?}", dist);
        let file_path = "config/gist_1000.toml";
        let config = VecDataConfig::load_from_toml_file(file_path)?;
        println!("Loaded config: {:#?}", config);
        let raw_vec_set = VecSet::<f32>::load_with(&config)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let config = HNSWConfig::default();

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
        index.save(path)?;
        let index = HNSWIndex::<f32>::load(path)?;
        println!("Loaded the index.");
        // <<<< Save and load the index.

        // Save and load the index without vec_set. >>>>
        println!("Saving the index...");
        let path = "data/hnsw_index.tmp.bin";
        let vec_set = index.save_without_vec_set(path)?.vec_set;

        let index = HNSWIndex::<f32>::load_with_external_vec_set(path, vec_set)?;
        println!("Loaded the index.");
        // <<<< Save and load the index without vec_set.

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
