use std::{collections::BTreeSet, fs::File, io::BufWriter, ops::Index};

use anyhow::Result;
use ordered_float::OrderedFloat;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{DistanceAdapter, DistanceAlgorithm},
    scalar::Scalar,
    vec_set::VecSet,
};
pub mod prelude {
    // All Index Traits
    pub use super::{IndexBuilder, IndexFromVecSet, IndexIter, IndexKNN, IndexSerde};
}

pub mod hnsw_index;
pub mod ivf_index;
pub mod linear_index;

/// A pair of the index and the distance.
/// For the response of the k-nearest neighbors search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidatePair {
    /// The distance to the query vector.
    ///
    /// The distance is wrapped in `OrderedFloat` to implement `Ord`.
    pub distance: OrderedFloat<f32>,
    /// The index of the vector in the index.
    pub index: usize,
}
impl CandidatePair {
    /// Get the distance. (unwrapped)
    pub fn distance(&self) -> f32 {
        self.distance.into_inner()
    }
    /// Create a new candidate pair.
    pub fn new(index: usize, distance: f32) -> Self {
        Self {
            index,
            distance: distance.into(),
        }
    }
}
impl PartialOrd for CandidatePair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CandidatePair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.distance).cmp(&OrderedFloat(other.distance))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultSet {
    pub k: usize,
    pub results: BTreeSet<CandidatePair>,
}
impl ResultSet {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            results: BTreeSet::new(),
        }
    }
    /// Check if a candidate with distance `d` is worth searching its neighbors.
    pub fn check_candidate(&self, pair: &CandidatePair) -> bool {
        self.results.len() < self.k || pair < self.results.iter().last().unwrap()
    }
    /// Add a candidate pair to the result set.
    ///
    /// Returns true if the pair is added.
    pub fn add(&mut self, pair: CandidatePair) -> bool {
        if self.results.len() < self.k {
            self.results.insert(pair);
            return true;
        }
        if let Some(last) = self.results.iter().last() {
            if pair.distance < last.distance {
                self.results.pop_last();
                self.results.insert(pair);
                return true;
            }
        }
        false
    }
    /// Convert the result set into a sorted vector.
    pub fn into_sorted_vec(self) -> Vec<CandidatePair> {
        self.results.into_iter().collect()
    }
    /// Convert the result set into a sorted vector with a limit.
    pub fn into_sorted_vec_limit(self, limit: usize) -> Vec<CandidatePair> {
        self.results.into_iter().take(limit).collect()
    }

    /// Pick at most m candidates as new neighbors heuristically.
    pub fn heuristic<T: Scalar>(
        self,
        m: usize,
        vec_set: &VecSet<T>,
        dist: DistanceAlgorithm,
    ) -> Vec<CandidatePair> {
        let mut neighbors: Vec<CandidatePair> = Vec::with_capacity(m);

        for pair in self.results {
            if neighbors.len() >= m {
                break;
            }
            let d = pair.distance();
            let v = &vec_set[pair.index];
            if neighbors.iter().all(|p| dist.d(v, &vec_set[p.index]) >= d) {
                neighbors.push(pair);
            }
        }
        neighbors
    }
}

/// The reusable data for the visited list.
///
/// Remember to call `setup` before each time you use it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReusableVisited {
    /// The turn of the visited list.
    turn: usize,
    /// The visited list. `true` if `visited[i] == turn`.
    visited: Vec<usize>,
}
impl ReusableVisited {
    pub fn new() -> Self {
        Self {
            turn: 0,
            visited: Vec::new(),
        }
    }
    pub fn is_visited(&self, index: usize) -> bool {
        self.visited[index] == self.turn
    }
    pub fn visit(&mut self, index: usize) {
        self.visited[index] = self.turn;
    }
    /// Setup the visited list for the new search.
    ///
    /// Exactly O(1) when `n` is the same as the last time.
    pub fn setup(&mut self, n: usize) {
        self.turn += 1;
        self.visited.resize(n, 0);
    }
}

/// The trait for index that can be accessed by the index.
///
/// Automatically implement `iter` method.
pub trait IndexIter<T: Scalar>: Index<usize, Output = [T]> {
    /// Get the number of vectors in the index.
    fn len(&self) -> usize;
    /// Get the dimension of the vectors.
    fn dim(&self) -> usize;

    /// Get an iterator of the vectors.
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a [T]>
    where
        T: 'a,
    {
        (0..self.len()).map(move |i| &self[i])
    }
}

/// The trait for index that can be built by adding vectors one by one.
pub trait IndexBuilder<T: Scalar>: IndexIter<T> {
    /// The Builder configuration of the index.
    type Config;
    /// Create a new index.
    fn new(dim: usize, dist: DistanceAlgorithm, config: Self::Config) -> Self;
    /// Add a vector to the index.
    ///
    /// Returns the index of the vector.
    fn add(&mut self, vec: &[T], rng: &mut impl Rng) -> usize;
}

/// The trait for index that can search the k-nearest neighbors.
pub trait IndexKNN<T: Scalar>: IndexIter<T> {
    /// Get the precise k-nearest neighbors.
    /// Returns a vector of pairs of the index and the distance.
    /// The vector is sorted by the distance in ascending order.
    fn knn(&self, query: &[T], k: usize) -> Vec<CandidatePair>;
}

/// The trait for index that can be built from a `VecSet`.
pub trait IndexFromVecSet<T: Scalar>: IndexIter<T> {
    /// The `from_vec_set` configuration of the index.
    type Config;

    /// Create an index from a vector set.
    fn from_vec_set(
        vec_set: VecSet<T>,
        dist: DistanceAlgorithm,
        config: Self::Config,
        rng: &mut impl Rng,
    ) -> Self;
}

pub trait IndexSerde: Serialize + for<'de> Deserialize<'de> {
    /// Save the index to the file.
    ///
    /// Some Index may have different Implementations.
    fn save(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }
    /// Load the index from the file.
    ///
    /// Some Index may have different Implementations.
    fn load(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let index = bincode::deserialize_from(reader)?;
        Ok(index)
    }
}
