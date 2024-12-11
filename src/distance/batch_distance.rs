use ndarray::prelude::*;

use super::DistanceAlgorithm;

/// Calculate the dist_cache for the given vectors.
///
/// - For L2Sqr, cache dot_product(a, a).
/// - For Cosine, cache vec_norm(a).
pub fn calc_dist_cache(dist: DistanceAlgorithm, vectors: ArrayView2<f32>) -> Array1<f32> {
    use DistanceAlgorithm::*;
    let dot_product: Array1<f32> = (&vectors * &vectors).sum_axis(Axis(1));

    match dist {
        L2Sqr => dot_product,
        Cosine => dot_product.sqrt(),
    }
}
pub struct VecBatch {
    pub(crate) dist: DistanceAlgorithm,
    pub(crate) vectors: Array2<f32>,
    pub(crate) dist_cache: Array1<f32>,
}

impl VecBatch {
    pub fn new(dist: DistanceAlgorithm, vectors: Array2<f32>) -> Self {
        let dist_cache = calc_dist_cache(dist, vectors.view());
        Self {
            dist,
            vectors,
            dist_cache,
        }
    }
    pub fn len(&self) -> usize {
        self.vectors.shape()[0]
    }
    pub fn dim(&self) -> usize {
        self.vectors.shape()[1]
    }
}

pub fn distance_parallel(a: &VecBatch, b: &VecBatch) -> Array2<f32> {
    use DistanceAlgorithm::*;
    assert!(a.dim() == b.dim(), "Dimension mismatch");
    assert!(a.dist == b.dist, "Distance algorithm mismatch");
    let dot_product = a.vectors.dot(&b.vectors.t());
    let broadcast_a = a.dist_cache.to_shape((a.len(), 1)).unwrap();
    let broadcast_a: ArrayView2<f32> = broadcast_a.broadcast((a.len(), b.len())).unwrap();
    let broadcast_b = b.dist_cache.to_shape((1, b.len())).unwrap();
    let broadcast_b: ArrayView2<f32> = broadcast_b.broadcast((a.len(), b.len())).unwrap();
    match a.dist {
        L2Sqr => {
            // (a-b)^2 = a^2 + b^2 - 2ab
            &broadcast_a + &broadcast_b - 2.0 * &dot_product
        }
        Cosine => {
            let norm_product = &broadcast_a * &broadcast_b;
            1.0 - &dot_product / norm_product
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::DistanceAdapter, vec_set::VecSet};

    use super::*;
    use ndarray::array;

    #[test]
    fn test_dist_parallel() {
        fn test_dist_parallel_with_dist(dist: DistanceAlgorithm) {
            let a = VecBatch::new(dist, array![[1.0, 2.0], [3.0, 4.0]]);
            let b = VecBatch::new(dist, array![[5.0, 6.0], [7.0, 8.0]]);
            let result: Array2<f32> = distance_parallel(&a, &b);
            let a_vec_set = VecSet::new(a.dim(), a.vectors.as_slice().unwrap().to_vec());
            let b_vec_set = VecSet::new(b.dim(), b.vectors.as_slice().unwrap().to_vec());
            for i in 0..a.len() {
                for j in 0..b.len() {
                    let dist_ij = dist.d(&a_vec_set[i], &b_vec_set[j]);
                    assert!((result[(i, j)] - dist_ij).abs() < 1e-6);
                }
            }
        }
        test_dist_parallel_with_dist(DistanceAlgorithm::L2Sqr);
        test_dist_parallel_with_dist(DistanceAlgorithm::Cosine);
    }
}
