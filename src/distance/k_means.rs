use core::f32;
use rayon::prelude::*;
use std::ops::Range;

use crate::{
    distance::{DistanceAdapter, DistanceAlgorithm},
    index_algorithm::{CandidatePair, ResultSet},
    scalar::Scalar,
    vec_set::VecSet,
};
use rand::{distributions::WeightedIndex, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansConfig {
    /// The number of clusters.
    pub k: usize,
    /// The maximum number of iterations.
    ///
    /// Recommend: 20.
    pub max_iter: usize,
    /// The tolerance to declare convergence.
    ///
    /// Recommend: 1e-6.
    pub tol: f32,
    /// The distance algorithm to use.
    pub dist: DistanceAlgorithm,
    /// The range of the dimension to use.
    /// If None, use all dimensions.
    pub selected: Option<Range<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans<T> {
    pub config: KMeansConfig,
    pub centroids: VecSet<T>,
}
/// Find the nearest centroid to the given vector.
/// The number of centroids should be greater than 0.
pub fn find_nearest_base<T: Scalar>(
    v: &[T],
    centroids: &VecSet<T>,
    dist: &DistanceAlgorithm,
) -> usize {
    assert!(
        centroids.len() > 0,
        "The number of centroids should be greater than 0."
    );
    use crate::index_algorithm::CandidatePair;
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| CandidatePair::new(i, dist.d(v, c)))
        .min()
        .unwrap()
        .index
}
impl<T: Scalar> KMeans<T> {
    /// Initialize the centroids using the k-means++ algorithm.
    /// Returns the initialized centroids.
    fn k_means_init(
        dim: usize,
        sel_set: &Vec<&[T]>,
        config: &KMeansConfig,
        rng: &mut impl Rng,
    ) -> VecSet<T> {
        let dist = config.dist;
        let k = config.k;
        let mut centroids = VecSet::with_capacity(dim, k);
        let n = sel_set.len();
        let first_idx = rng.gen_range(0..n);
        centroids.push(sel_set[first_idx]);
        let mut weight = vec![f32::INFINITY; n];
        for idx in 1..k {
            weight.par_iter_mut().zip(sel_set).for_each(|(w, v)| {
                *w = w.min(dist.d(&centroids[idx - 1], v));
            });
            // Randomly choose the next centroid with probability proportional to the distance.
            // If all weights are zero, choose randomly.
            let c = WeightedIndex::new(&weight)
                .map(|d| d.sample(rng))
                .unwrap_or(rng.gen_range(0..n));

            centroids.push(sel_set[c]);
        }
        centroids
    }
    /// Perform the k-means clustering on the given vector set.
    /// The initial centroids are initialized by the k-means++ algorithm.
    ///
    /// The number of clusters should be greater than 0.
    /// The selected range should be in the range [0, vec_set.dim()).
    ///
    /// Call [VecSet::random_sample] before this to select a subset of the vectors for the clustering.
    pub fn from_vec_set(vec_set: &VecSet<T>, config: KMeansConfig, rng: &mut impl Rng) -> Self {
        assert!(
            config.k > 0,
            "The number of clusters should be greater than 0."
        );
        assert!(
            config.selected.is_none() || config.selected.as_ref().unwrap().end <= vec_set.dim(),
            "The selected range should be in the range [0, vec_set.dim())"
        );

        let (l, r, dim) = match &config.selected {
            Some(selected) => (selected.start, selected.end, selected.len()),
            None => (0, vec_set.dim(), vec_set.dim()),
        };
        let sel_set = vec_set.iter().map(|v| &v[l..r]).collect::<Vec<_>>();
        let mut centroids = Self::k_means_init(dim, &sel_set, &config, rng);

        // Use f32 to avoid overflow when summing up.
        let mut new_centroid_sums = VecSet::new(dim, vec![0.0; config.k * dim]);
        for _ in 0..config.max_iter {
            let mut new_centroid_iter: Vec<&mut [f32]> = new_centroid_sums.iter_mut().collect();

            let centroid_for_vec: Vec<usize> = sel_set
                .par_iter()
                .map(|v| find_nearest_base(v, &centroids, &config.dist))
                .collect();
            let mut vec_for_centroid = vec![Vec::new(); config.k];
            for (i, c) in centroid_for_vec.iter().enumerate() {
                vec_for_centroid[*c].push(i);
            }

            new_centroid_iter
                .par_iter_mut()
                .zip(vec_for_centroid)
                .enumerate()
                .for_each(|(idx, (c, vec_idx))| {
                    if vec_idx.is_empty() {
                        // If the centroid is empty, keep it unchanged.
                        c.iter_mut()
                            .zip(centroids[idx].iter())
                            .for_each(|(s, v)| *s = v.cast_to_f32());
                        return;
                    }
                    c.iter_mut().for_each(|v| *v = 0.0);
                    for &i in vec_idx.iter() {
                        let v = sel_set[i];
                        for (s, v) in c.iter_mut().zip(v.iter()) {
                            *s += v.cast_to_f32();
                        }
                    }
                    let n = vec_idx.len() as f32;
                    c.iter_mut().for_each(|v| *v /= n);
                });

            let new_centroids = new_centroid_sums.to_type::<T>();
            let max_diff = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(a, b)| DistanceAlgorithm::L2Sqr.d(a, b))
                .fold(f32::NEG_INFINITY, f32::max);

            centroids = new_centroids;
            if max_diff < config.tol {
                break;
            }
        }
        Self { config, centroids }
    }

    /// Find the nearest centroid to the given vector.
    /// This will use the selected range if it is specified in the config.
    pub fn find_nearest(&self, v: &[T]) -> usize {
        let dim = self.centroids.dim();
        let v = &v[self.config.selected.clone().unwrap_or(0..dim)];
        find_nearest_base(v, &self.centroids, &self.config.dist)
    }

    /// Find the `n_probes` nearest centroids to the given vector.
    /// This will use the selected range if it is specified in the config.
    pub fn find_n_nearest(&self, v: &[T], n_probes: usize) -> Vec<usize> {
        assert!(
            n_probes > 0,
            "The number of probes should be greater than 0."
        );
        let dim = self.centroids.dim();
        let dist = self.config.dist;
        let v = &v[self.config.selected.clone().unwrap_or(0..dim)];
        let mut result_set = ResultSet::new(n_probes);
        for (i, c) in self.centroids.iter().enumerate() {
            result_set.add(CandidatePair::new(i, dist.d(v, c)));
        }
        result_set
            .into_sorted_vec()
            .iter()
            .map(|p| p.index)
            .collect()
    }
}

#[cfg(test)]
mod test {

    use anyhow::Result;

    use crate::config::VecDataConfig;

    use super::*;

    #[test]
    fn test_k_means() {
        let vec_set = VecSet::new(2, vec![0.0, 0.0, 1.0, 0.0, -1.0, -2.0, -2.0, -1.0]);
        let config = KMeansConfig {
            k: 2,
            max_iter: 20,
            tol: 1e-6,
            dist: DistanceAlgorithm::L2Sqr,
            selected: None,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let k_means = KMeans::from_vec_set(&vec_set, config.clone(), &mut rng);
        assert_eq!(k_means.centroids.len(), config.k);
        for c in k_means.centroids.iter() {
            assert_eq!(c.len(), vec_set.dim());
            println!("{:?}", c);
        }
    }

    #[test]
    fn test_k_means_u8() {
        let vec_set = VecSet::new(2, vec![0, 0, 1, 0, 255, 254, 255, 255]);
        let config = KMeansConfig {
            k: 2,
            max_iter: 20,
            tol: 1e-6,
            dist: DistanceAlgorithm::L2Sqr,
            selected: None,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let k_means = KMeans::from_vec_set(&vec_set, config.clone(), &mut rng);
        assert_eq!(k_means.centroids.len(), config.k);
        for c in k_means.centroids.iter() {
            assert_eq!(c.len(), vec_set.dim());
            println!("{:?}", c);
        }
    }

    #[test]
    fn test_k_means_on_real_set() -> Result<()> {
        fn clip_msg(s: &str) -> String {
            if s.len() > 100 {
                format!("{}...", &s[..100])
            } else {
                s.to_string()
            }
        }
        let file_path = "config/gist_1000.toml";
        let config = VecDataConfig::load_from_toml_file(file_path)?;
        let vec_set = VecSet::<f32>::load_with(&config)?;
        let k_means_config = KMeansConfig {
            k: 3,
            max_iter: 20,
            tol: 1e-6,
            dist: DistanceAlgorithm::L2Sqr,
            selected: Some(0..5),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vec_set = vec_set.random_sample(400, &mut rng);
        let k_means = KMeans::from_vec_set(&vec_set, k_means_config.clone(), &mut rng);

        assert_eq!(k_means.centroids.len(), k_means_config.k);
        for c in k_means.centroids.iter() {
            println!("{}", clip_msg(&format!("{:?}", c)));
        }

        let i = 1;
        let find_i = k_means.find_nearest(&k_means.centroids[i]);
        assert_eq!(
            find_i, i,
            "The nearest centroid should be the vector itself."
        );

        Ok(())
    }
}
