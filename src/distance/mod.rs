pub mod k_means;
pub mod pq_table;
use crate::scalar::Scalar;

use serde::{Deserialize, Serialize};

/// Distance algorithm to be used in the vector database.
///
/// See also `DistanceAlgorithm::d()`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceAlgorithm {
    /// L2 squared distance, AKA squared Euclidean distance.
    ///
    /// Range: `[0.0, +inf]`
    L2Sqr,
    /// L2 distance, AKA Euclidean distance.
    ///
    /// Range: `[0.0, +inf]`
    L2,
    /// Cosine distance.
    /// `cosine_distance = 1 - dot_product / (norm_self * norm_other)`
    ///
    /// Range: `[0.0, 2.0]`
    Cosine,
}
use DistanceAlgorithm::*;
/// Trait for calculating distances between two vectors.
///
/// `[T] where T: Scalar` implements this trait.
pub trait SliceDistance {
    /// The *square* of the L2 distance.
    ///
    /// Range: `[0.0, +inf]`
    fn l2_sqr_distance(&self, other: &Self) -> f32;
    /// L2 distance.
    ///
    /// Range: `[0.0, +inf]`
    fn l2_distance(&self, other: &Self) -> f32 {
        self.l2_sqr_distance(other).sqrt()
    }
    /// The dot product of two vectors. (For internal use)
    fn dot_product(&self, other: &Self) -> f32;
    /// Cosine distance.
    /// `cosine_distance = 1 - dot_product / (norm_self * norm_other)`
    ///
    /// Range: `[0.0, 2.0]`
    fn cosine_distance(&self, other: &Self) -> f32;
}
impl<T: Scalar> SliceDistance for [T] {
    fn l2_sqr_distance(&self, other: &Self) -> f32 {
        assert_eq!(
            self.len(),
            other.len(),
            "Vectors must have the same length to calculate distance."
        );
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a.cast_to_f32() - b.cast_to_f32()).powi(2))
            .sum()
    }
    fn dot_product(&self, other: &Self) -> f32 {
        assert_eq!(
            self.len(),
            other.len(),
            "Vectors must have the same length to calculate distance."
        );
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.cast_to_f32() * b.cast_to_f32())
            .sum()
    }
    fn cosine_distance(&self, other: &Self) -> f32 {
        let dot_product_sqr = self.dot_product(other);
        let norm_self = self.dot_product(self).sqrt();
        let norm_other = other.dot_product(other).sqrt();
        1.0 - dot_product_sqr / (norm_self * norm_other)
    }
}

pub trait DistanceAdapter<Lhs: ?Sized, Rhs: ?Sized> {
    /// Calculate distance using the specified algorithm.
    fn distance(&self, a: &Lhs, b: &Rhs) -> f32;
    /// Calculate distance using the specified algorithm.
    /// Alias of `distance()`.
    fn d(&self, a: &Lhs, b: &Rhs) -> f32 {
        self.distance(a, b)
    }
}

impl<T: Scalar> DistanceAdapter<[T], [T]> for DistanceAlgorithm {
    fn distance(&self, a: &[T], b: &[T]) -> f32 {
        match self {
            L2Sqr => a.l2_sqr_distance(b),
            L2 => a.l2_distance(b),
            Cosine => a.cosine_distance(b),
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_l2_sqr_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((L2Sqr.d(a.as_slice(), b.as_slice()) - 27.0_f32).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1, 2, 3];
        let b = [2, 4, 6];
        assert!((Cosine.d(a.as_slice(), b.as_slice()) - 0.0_f32).abs() < EPSILON);
    }
}