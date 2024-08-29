use std::collections::BTreeSet;

use crate::binary_scalar::BinaryScalar;
use crate::config::DistanceAlgorithm;
use crate::distance::Distance;
use crate::vec_set::{DynamicVecRef, DynamicVecSet, VecSet};

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

pub mod ivf_index;
pub mod linear_index;
