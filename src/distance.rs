use crate::config::DistanceAlgorithm;

use crate::vec_set::DynamicVecRef;

/// Trait for calculating distances between two vectors.
///
/// `[T] where T: BinaryScalar` and `DynamicVecRef` should implement this trait.
pub trait Distance {
    /// The *square* of the L2 distance.
    ///
    /// Range: `[0.0, +inf]`
    fn l2_sqr_distance(&self, other: &Self) -> f32;
    /// L2 distance. *Automatically implemented.*
    ///
    /// Range: `[0.0, +inf]`
    fn l2_distance(&self, other: &Self) -> f32 {
        self.l2_sqr_distance(other).sqrt()
    }
    /// The dot product of two vectors. (For internal use)
    fn dot_product_sqr(&self, other: &Self) -> f32;
    /// Cosine distance.
    /// `cosine_distance = 1 - dot_product / (norm_self * norm_other)`
    ///
    /// Range: `[0.0, 2.0]`
    fn cosine_distance(&self, other: &Self) -> f32;
    /// Calculate distance using the specified algorithm.
    ///
    /// Internally calls `DistanceAlgorithm::distance`.
    fn dynamic_distance(&self, other: &Self, algorithm: DistanceAlgorithm) -> f32 {
        algorithm.distance(self, other)
    }
}
impl Distance for [f32] {
    fn l2_sqr_distance(&self, other: &Self) -> f32 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
    }
    fn dot_product_sqr(&self, other: &Self) -> f32 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
    }
    fn cosine_distance(&self, other: &Self) -> f32 {
        let dot_product_sqr = self.dot_product_sqr(other);
        let norm_self = self.dot_product_sqr(self).sqrt();
        let norm_other = other.dot_product_sqr(other).sqrt();
        1.0 - dot_product_sqr / (norm_self * norm_other)
    }
}
impl Distance for [u8] {
    fn l2_sqr_distance(&self, other: &Self) -> f32 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (*a as f32 - *b as f32).powi(2))
            .sum::<f32>()
    }
    fn dot_product_sqr(&self, other: &Self) -> f32 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (*a as f32 * *b as f32))
            .sum::<f32>()
    }
    fn cosine_distance(&self, other: &Self) -> f32 {
        let dot_product_sqr = self.dot_product_sqr(other);
        let norm_self = self.dot_product_sqr(self).sqrt();
        let norm_other = other.dot_product_sqr(other).sqrt();
        1.0 - dot_product_sqr / (norm_self * norm_other)
    }
}

impl Distance for DynamicVecRef<'_> {
    fn l2_sqr_distance(&self, other: &Self) -> f32 {
        match (self, other) {
            (Self::Float32(a), Self::Float32(b)) => a.l2_sqr_distance(b),
            (Self::UInt8(a), Self::UInt8(b)) => a.l2_sqr_distance(b),
            _ => panic!("Cannot calculate distance between different types of vectors."),
        }
    }
    fn dot_product_sqr(&self, other: &Self) -> f32 {
        match (self, other) {
            (Self::Float32(a), Self::Float32(b)) => a.dot_product_sqr(b),
            (Self::UInt8(a), Self::UInt8(b)) => a.dot_product_sqr(b),
            _ => panic!("Cannot calculate distance between different types of vectors."),
        }
    }
    fn cosine_distance(&self, other: &Self) -> f32 {
        match (self, other) {
            (Self::Float32(a), Self::Float32(b)) => a.cosine_distance(b),
            (Self::UInt8(a), Self::UInt8(b)) => a.cosine_distance(b),
            _ => panic!("Cannot calculate distance between different types of vectors."),
        }
    }
}

impl DistanceAlgorithm {
    /// Calculate distance between two vectors using the specified algorithm.
    pub fn distance<T: Distance + ?Sized>(&self, a: &T, b: &T) -> f32 {
        use DistanceAlgorithm::*;
        match self {
            L2Sqr => a.l2_sqr_distance(b),
            L2 => a.l2_distance(b),
            Cosine => a.cosine_distance(b),
        }
    }

    /// Alias for `distance`.
    /// Calculate distance between two vectors using the specified algorithm.
    pub fn d<T: Distance + ?Sized>(&self, a: &T, b: &T) -> f32 {
        self.distance(a, b)
    }

    /// Calculate distance between two slices using the specified algorithm.
    ///
    /// This may help the type inference system.
    pub fn distance_slice<T>(&self, a: &[T], b: &[T]) -> f32
    where
        [T]: Distance,
    {
        self.distance(a, b)
    }

    /// Alias for `distance_slice`.
    /// Calculate distance between two slices using the specified algorithm.
    ///
    /// This may help the type inference system.
    pub fn ds<T>(&self, a: &[T], b: &[T]) -> f32
    where
        [T]: Distance,
    {
        self.distance(a, b)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_l2_sqr_distance() {
        use DistanceAlgorithm::*;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((L2Sqr.ds(&a, &b) - 27.0_f32).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_distance() {
        use DistanceAlgorithm::*;
        let a = [1, 2, 3];
        let b = [2, 4, 6];
        assert!((Cosine.ds(&a, &b) - 0.0_f32).abs() < EPSILON);
    }
}
