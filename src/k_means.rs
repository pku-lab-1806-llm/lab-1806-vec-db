use crate::{binary_scalar::BinaryScalar, config::DistanceAlgorithm, vec_set::VecSet};
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// The number of clusters.
    pub k: usize,
    /// The maximum number of iterations.
    pub max_iter: usize,
    /// The tolerance to declare convergence.
    pub tol: f32,
    /// The distance algorithm to use.
    pub dist: DistanceAlgorithm,
}

pub struct KMeans<T> {
    pub centroids: VecSet<T>,
}
#[allow(unused)] // Remove this line soon.
impl<T: BinaryScalar> KMeans<T> {
    fn k_means_init(vec_set: &VecSet<T>, config: &KMeansConfig, rng: &mut impl Rng) -> VecSet<T> {
        assert!(
            (0..=vec_set.len()).contains(&config.k),
            "k in k-means should be in the range [0, len(vec_set)]"
        );
        let centroids = VecSet::zeros(vec_set.dim(), config.k);
        unimplemented!("k_means_xx_init");
        centroids
    }
    pub fn from_vec_set(vec_set: &VecSet<T>, config: &KMeansConfig, rng: &mut impl Rng) -> Self {
        let centroids = Self::k_means_init(vec_set, config, rng);
        unimplemented!("from_vec_set");
        Self { centroids }
    }
}
