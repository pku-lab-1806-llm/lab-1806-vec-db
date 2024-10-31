pub mod k_means;
pub mod pq_table;
use std::simd::num::SimdFloat;

use crate::scalar::Scalar;

use serde::{Deserialize, Serialize};

pub mod prelude {
    // All Distance Traits & Algorithms
    pub use super::{DistanceAdapter, DistanceAlgorithm, SliceDistance};
}

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
    /// SIMD-accelerated L2 squared distance.
    /// Range: `[0.0, +inf]`
    SimdL2Sqr,
    /// SIMD-accelerated L2 distance.
    /// Range: `[0.0, +inf]`
    SimdL2,
}
use DistanceAlgorithm::*;
/// Trait for calculating distances between two vectors.
///
/// `[T] where T: Scalar` implements this trait.
pub trait SliceDistance {
    /// The *square* of the L2 distance.
    fn l2_sqr_distance(&self, rhs: &Self) -> f32;

    /// L2 distance.
    fn l2_distance(&self, rhs: &Self) -> f32 {
        self.l2_sqr_distance(rhs).sqrt()
    }

    /// The dot product of two vectors. (For internal use)
    fn dot_product(&self, rhs: &Self) -> f32;

    /// Cosine distance.
    /// `cosine_distance = 1 - dot_product / (norm_self * norm_other)`
    ///
    /// Range: `[0.0, 2.0]`
    fn cosine_distance(&self, rhs: &Self) -> f32 {
        let dot_product_sqr = self.dot_product(rhs);
        let norm_self = self.dot_product(self).sqrt();
        let norm_other = rhs.dot_product(rhs).sqrt();
        1.0 - dot_product_sqr / (norm_self * norm_other)
    }

    /// Check if SIMD is supported for the current architecture.
    fn is_simd_supported(&self) -> bool;

    /// SIMD-accelerated L2 squared distance.
    fn simd_l2_sqr_distance(&self, rhs: &Self) -> f32;

    /// SIMD-accelerated L2 distance.
    fn simd_l2_distance(&self, rhs: &Self) -> f32 {
        self.simd_l2_sqr_distance(rhs).sqrt()
    }
}

impl<T: Scalar> SliceDistance for [T] {
    default fn l2_sqr_distance(&self, rhs: &Self) -> f32 {
        assert_eq!(
            self.len(),
            rhs.len(),
            "Vectors must have the same length to calculate distance."
        );
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| {
                let diff = a.cast_to_f32() - b.cast_to_f32();
                diff * diff
            })
            .sum()
    }
    default fn dot_product(&self, rhs: &Self) -> f32 {
        assert_eq!(
            self.len(),
            rhs.len(),
            "Vectors must have the same length to calculate distance."
        );
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| a.cast_to_f32() * b.cast_to_f32())
            .sum()
    }

    default fn is_simd_supported(&self) -> bool {
        false
    }

    default fn simd_l2_sqr_distance(&self, _rhs: &Self) -> f32 {
        panic!(
            "SIMD is not supported on (arch: {}, type: {})",
            std::env::consts::ARCH,
            std::any::type_name::<Self>()
        );
    }

    default fn simd_l2_distance(&self, rhs: &Self) -> f32 {
        self.simd_l2_sqr_distance(rhs).sqrt()
    }
}

impl SliceDistance for [f32] {
    fn l2_sqr_distance(&self, rhs: &Self) -> f32 {
        assert_eq!(
            self.len(),
            rhs.len(),
            "Vectors must have the same length to calculate distance."
        );
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum()
    }
    fn dot_product(&self, other: &Self) -> f32 {
        assert_eq!(
            self.len(),
            other.len(),
            "Vectors must have the same length to calculate distance."
        );
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }

    fn is_simd_supported(&self) -> bool {
        true
    }

    fn simd_l2_sqr_distance(&self, rhs: &Self) -> f32 {
        assert_eq!(
            self.len(),
            rhs.len(),
            "Vectors must have the same length to calculate distance."
        );
        use std::simd::Simd;
        const N: usize = 4;

        let self_iter = self.chunks_exact(N);
        let rhs_iter = rhs.chunks_exact(N);
        let rest_sum = self_iter.remainder().l2_sqr_distance(rhs_iter.remainder());

        let self_simd = self_iter.map(Simd::<f32, N>::from_slice);
        let rhs_simd = rhs_iter.map(Simd::<f32, N>::from_slice);
        let simd_sum = self_simd
            .zip(rhs_simd)
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<Simd<f32, N>>();

        simd_sum.reduce_sum() + rest_sum
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
            SimdL2Sqr => a.simd_l2_sqr_distance(b),
            SimdL2 => a.simd_l2_distance(b),
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
