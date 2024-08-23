use std::ops::{AddAssign, DivAssign};

use crate::{
    binary_scalar::BinaryScalar, config::DistanceAlgorithm, distance::Distance, vec_set::VecSet,
};
use rand::{distributions::WeightedIndex, prelude::*};

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
impl<T: BinaryScalar + From<f32> + Into<f32>> KMeans<T>
where
    [T]: Distance,
{
    fn k_means_init(vec_set: &VecSet<T>, config: &KMeansConfig, rng: &mut impl Rng) -> VecSet<T> {
        let k = config.k;
        let dist = config.dist;
        let mut centroids = VecSet::zeros(vec_set.dim(), k);
        centroids.put(0, &vec_set[rng.gen_range(0..vec_set.len())]);
        let mut weight = vec![0_f32; vec_set.len()];
        for (v, w) in vec_set.iter().zip(weight.iter_mut()) {
            *w = dist.d(&centroids[0], v);
        }
        for idx in 1..k {
            for (v, w) in vec_set.iter().zip(weight.iter_mut()) {
                *w = w.min(dist.d(&centroids[idx - 1], v));
            }
            let c = WeightedIndex::new(&weight)
                .expect("WeightedIndex::new failed")
                .sample(rng);

            centroids.put(idx, &vec_set[c]);
        }
        centroids
    }
    pub fn from_vec_set(vec_set: &VecSet<T>, config: &KMeansConfig, rng: &mut impl Rng) -> Self {
        assert!(
            (0..=vec_set.len()).contains(&config.k),
            "k in k-means should be in the range [0, len(vec_set)]"
        );
        let mut centroids = Self::k_means_init(vec_set, config, rng);

        for _ in 0..config.max_iter {
            let mut new_centroids = VecSet::<f32>::zeros(vec_set.dim(), config.k);
            let mut count = vec![0; config.k];
            for v in vec_set.iter() {
                let mut min_dist = f32::INFINITY;
                let mut min_idx = 0;
                for (idx, c) in centroids.iter().enumerate() {
                    let dist = config.dist.d(v, c);
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = idx;
                    }
                }
                count[min_idx] += 1;
                let nc = new_centroids.get_mut(min_idx);
                for (v_new, v) in nc.iter_mut().zip(v.iter()) {
                    *v_new += Into::<f32>::into(*v);
                }
            }
            for i in 0..config.k {
                let c = new_centroids.get_mut(i);
                for v in c.iter_mut() {
                    *v /= count[i] as f32;
                }
            }
            let mut max_diff = 0_f32;
            for (c, nc) in centroids.to_f32().iter().zip(new_centroids.iter()) {
                max_diff = max_diff.max(config.dist.d(c, nc));
            }
            if max_diff < config.tol {
                break;
            }
            for i in 0..config.k {
                let c = centroids.get_mut(i);
                let nc = &new_centroids[i];
                for (c, nc) in c.iter_mut().zip(nc.iter()) {
                    *c = From::<f32>::from(*nc);
                }
            }
        }
        Self { centroids }
    }
}
