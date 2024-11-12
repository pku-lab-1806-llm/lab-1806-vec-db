pub mod k_means;
pub mod pq_table;

use serde::{Deserialize, Serialize};

pub mod prelude {
    // All Distance Traits & Algorithms
    pub use super::{DistanceAdapter, DistanceAlgorithm};
}

/// Distance algorithm to be used in the vector database.
///
/// SIMD is expected to be used automatically in most cases.
/// If you can strictly ensure that your program always runs on AVX supported CPUs,
/// re-compiling the library on your machine with RUSTFLAGS="-C target-cpu=native".
///
/// See also [DistanceAlgorithm::d].
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
    /// `cosine_distance(a, b) = 1 - dot_product(a, b) / (norm(a) * norm(b))`
    ///
    /// Range: `[0.0, 2.0]`
    Cosine,
}
use DistanceAlgorithm::*;

use crate::scalar::BaseScalar;

pub trait DistanceScalar: BaseScalar {
    /// The *square* of the L2 distance.
    fn l2_sqr_distance(a: &[Self], b: &[Self]) -> f32;

    /// L2 distance.
    fn l2_distance(a: &[Self], b: &[Self]) -> f32 {
        Self::l2_sqr_distance(a, b).sqrt()
    }

    /// The dot product of two vectors.
    fn dot_product(a: &[Self], b: &[Self]) -> f32;

    /// The norm of a vector.
    fn vec_norm(a: &[Self]) -> f32 {
        Self::dot_product(a, a).sqrt()
    }

    /// Cosine distance.
    /// `cosine_distance = 1 - dot_product / (norm_lhs * norm_rhs)`
    /// Range: `[0.0, 2.0]`
    fn cosine_distance(a: &[Self], b: &[Self]) -> f32 {
        let norm_lhs = Self::vec_norm(a);
        let norm_rhs = Self::vec_norm(b);
        Self::cosine_distance_cached(a, b, norm_lhs, norm_rhs)
    }

    /// Cosine distance with pre-calculated norms.
    fn cosine_distance_cached(a: &[Self], b: &[Self], norm_a: f32, norm_b: f32) -> f32 {
        let dot_product_sqr = Self::dot_product(a, b);
        1.0 - dot_product_sqr / (norm_a * norm_b)
    }
}
impl DistanceScalar for f32 {
    fn l2_sqr_distance(a: &[Self], b: &[Self]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
    fn dot_product(a: &[Self], b: &[Self]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}
impl DistanceScalar for u8 {
    fn l2_sqr_distance(a: &[Self], b: &[Self]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f32 - *y as f32).powi(2))
            .sum()
    }
    fn dot_product(a: &[Self], b: &[Self]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f32) * (*y as f32))
            .sum()
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

impl<T: DistanceScalar> DistanceAdapter<[T], [T]> for DistanceAlgorithm {
    fn distance(&self, a: &[T], b: &[T]) -> f32 {
        match self {
            L2Sqr => T::l2_sqr_distance(a, b),
            L2 => T::l2_distance(a, b),
            Cosine => T::cosine_distance(a, b),
        }
    }
}

/// Distance adapter for vectors with pre-calculated norms.
impl<T: DistanceScalar> DistanceAdapter<(&[T], f32), (&[T], f32)> for DistanceAlgorithm {
    fn distance(&self, a: &(&[T], f32), b: &(&[T], f32)) -> f32 {
        let (a, norm_a) = a;
        let (b, norm_b) = b;
        match self {
            L2Sqr => T::l2_sqr_distance(a, b),
            L2 => T::l2_distance(a, b),
            Cosine => T::cosine_distance_cached(a, b, *norm_a, *norm_b),
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
