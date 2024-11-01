pub mod k_means;
pub mod pq_table;

use serde::{Deserialize, Serialize};
use wide::f32x8 as Simd;
const SIMD_N: usize = 8;

pub mod prelude {
    // All Distance Traits & Algorithms
    pub use super::{DistanceAdapter, DistanceAlgorithm};
}

/// Distance algorithm to be used in the vector database.
/// SIMD is expected to be used automatically in most cases, but you can force it with the `Simd` variants.
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
    /// Dot product.
    DotProduct,
    /// Cosine distance.
    /// `cosine_distance = 1 - dot_product / (norm_lhs * norm_rhs)`
    ///
    /// Range: `[0.0, 2.0]`
    Cosine,
    /// SIMD-accelerated L2 squared distance.
    /// Range: `[0.0, +inf]`
    SimdL2Sqr,
    /// SIMD-accelerated L2 distance.
    /// Range: `[0.0, +inf]`
    SimdL2,
    /// SIMD-accelerated dot product.
    SimdDotProduct,
    /// SIMD-accelerated Cosine distance.
    /// `cosine_distance = 1 - dot_product / (norm_lhs * norm_rhs)`
    ///
    /// Range: `[0.0, 2.0]`
    SimdCosine,
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

    /// Cosine distance.
    /// `cosine_distance = 1 - dot_product / (norm_lhs * norm_rhs)`
    /// Range: `[0.0, 2.0]`
    fn cosine_distance(a: &[Self], b: &[Self]) -> f32 {
        let dot_product_sqr = Self::dot_product(a, b);
        let norm_lhs = Self::dot_product(a, a).sqrt();
        let norm_rhs = Self::dot_product(b, b).sqrt();
        1.0 - dot_product_sqr / (norm_lhs * norm_rhs)
    }

    /// SIMD-accelerated L2 squared distance.
    fn simd_l2_sqr_distance(a: &[Self], b: &[Self]) -> f32;
    /// SIMD-accelerated L2 distance.
    fn simd_l2_distance(a: &[Self], b: &[Self]) -> f32 {
        Self::simd_l2_sqr_distance(a, b).sqrt()
    }
    /// SIMD-accelerated dot product.
    fn simd_dot_product(a: &[Self], b: &[Self]) -> f32;
    /// SIMD-accelerated Cosine distance.
    fn simd_cosine_distance(a: &[Self], b: &[Self]) -> f32 {
        let dot_product_sqr = Self::simd_dot_product(a, b);
        let norm_lhs = Self::simd_dot_product(a, a).sqrt();
        let norm_rhs = Self::simd_dot_product(b, b).sqrt();
        1.0 - dot_product_sqr / (norm_lhs * norm_rhs)
    }
}
impl DistanceScalar for f32 {
    fn l2_sqr_distance(a: &[Self], b: &[Self]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
    fn dot_product(a: &[Self], b: &[Self]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    fn simd_l2_sqr_distance(a: &[Self], b: &[Self]) -> f32 {
        let a = a.chunks(SIMD_N).map(Simd::from);
        let b = b.chunks(SIMD_N).map(Simd::from);

        a.zip(b)
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .reduce(|x, y| x + y)
            .unwrap_or(Simd::ZERO)
            .reduce_add()
    }
    fn simd_dot_product(a: &[Self], b: &[Self]) -> f32 {
        let a = a.chunks(SIMD_N).map(Simd::from);
        let b = b.chunks(SIMD_N).map(Simd::from);

        a.zip(b)
            .map(|(a, b)| a * b)
            .reduce(|x, y| x + y)
            .unwrap_or(Simd::ZERO)
            .reduce_add()
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
    fn simd_l2_sqr_distance(a: &[Self], b: &[Self]) -> f32 {
        eprintln!("SIMD-accelerated L2 squared distance is not implemented for u8");
        Self::l2_sqr_distance(a, b)
    }
    fn simd_dot_product(a: &[Self], b: &[Self]) -> f32 {
        eprintln!("SIMD-accelerated dot product is not implemented for u8");
        Self::dot_product(a, b)
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
            DotProduct => T::dot_product(a, b),
            Cosine => T::cosine_distance(a, b),
            SimdL2Sqr => T::simd_l2_sqr_distance(a, b),
            SimdL2 => T::simd_l2_distance(a, b),
            SimdDotProduct => T::simd_dot_product(a, b),
            SimdCosine => T::simd_cosine_distance(a, b),
        }
    }
}

#[cfg(test)]
mod test {

    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_l2_sqr_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((L2Sqr.d(a.as_slice(), b.as_slice()) - 27.0_f32).abs() < EPSILON);
    }

    #[test]
    fn test_simd_distance() {
        let mut rng = StdRng::seed_from_u64(42);
        let a: Vec<f32> = (0..100).map(|_| rng.gen_range(0.0..1.0)).collect();
        let b: Vec<f32> = (0..100).map(|_| rng.gen_range(0.0..1.0)).collect();

        let simd_l2_sqr = SimdL2Sqr.d(a.as_slice(), b.as_slice());
        let l2_sqr = L2Sqr.d(a.as_slice(), b.as_slice());

        println!("L2Sqr: {:.9}, SimdL2Sqr: {:.9}", l2_sqr, simd_l2_sqr);

        assert!(
            (simd_l2_sqr - l2_sqr).abs() < EPSILON * l2_sqr.max(1.0),
            "L2Sqr: {:.9}, SimdL2Sqr: {:.9}",
            l2_sqr,
            simd_l2_sqr
        );

        let simd_cosine = SimdCosine.d(a.as_slice(), b.as_slice());
        let cosine = Cosine.d(a.as_slice(), b.as_slice());

        println!("Cosine: {:.9}, SimdCosine: {:.9}", cosine, simd_cosine);

        assert!(
            (simd_cosine - cosine).abs() < EPSILON,
            "Cosine: {:.9}, SimdCosine: {:.9}",
            cosine,
            simd_cosine
        );
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1, 2, 3];
        let b = [2, 4, 6];
        assert!((Cosine.d(a.as_slice(), b.as_slice()) - 0.0_f32).abs() < EPSILON);
    }
}
