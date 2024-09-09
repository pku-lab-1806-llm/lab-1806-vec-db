use std::rc::Rc;

use rand::Rng;

use crate::{distance::DistanceAlgorithm, scalar::Scalar, vec_set::VecSet};

pub mod hnsw_index;
pub mod ivf_index;
pub mod linear_index;

/// A pair of the index and the distance.
/// For the response of the k-nearest neighbors search.
///
/// This should implement `Ord` to be used in `BTreeSet`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponsePair {
    pub index: usize,
    pub distance: f32,
}
impl ResponsePair {
    pub fn new(index: usize, distance: f32) -> Self {
        Self { index, distance }
    }
}
impl Eq for ResponsePair {}
impl PartialOrd for ResponsePair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
impl Ord for ResponsePair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .expect("Failed to compare f32 distance in response pair.")
    }
}

pub trait IndexBuilder<T: Scalar> {
    /// The configuration of the index.
    type Config;
    /// Create a new index.
    fn new(dist: DistanceAlgorithm, config: Self::Config) -> Self;
    /// Add a vector to the index.
    fn add(&mut self, vec: &[T], label: usize, rng: &mut impl Rng);
}

pub trait IndexKNN<T: Scalar> {
    /// Get the precise k-nearest neighbors.
    /// Returns a vector of pairs of the index and the distance.
    /// The vector is sorted by the distance in ascending order.
    fn knn(&self, query: &[T], k: usize) -> Vec<ResponsePair>;
}

pub trait IndexFromVecSet<T: Scalar> {
    /// The configuration of the index.
    type Config;

    /// Create an index from a vector set.
    fn from_vec_set(
        vec_set: Rc<VecSet<T>>,
        dist: DistanceAlgorithm,
        config: Self::Config,
        rng: &mut impl Rng,
    ) -> Self;
}
