use std::ops::Range;

use crate::{scalar::Scalar, distance::DistanceAlgorithm, vec_set::VecSet};
use rand::{distributions::WeightedIndex, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansConfig {
    /// The number of clusters.
    pub k: usize,
    /// The maximum number of iterations.
    pub max_iter: usize,
    /// The tolerance to declare convergence.
    pub tol: f32,
    /// The distance algorithm to use.
    pub dist: DistanceAlgorithm,
    /// The range of the dimension to use.
    /// If None, use all dimensions.
    pub selected: Option<Range<usize>>,
}

#[derive(Debug, Clone)]
pub struct KMeans<T> {
    pub config: KMeansConfig,
    pub centroids: VecSet<T>,
}
impl<T: Scalar> KMeans<T> {
    /// Initialize the centroids using the k-means++ algorithm.
    /// Returns the initialized centroids.
    fn k_means_init(vec_set: &VecSet<T>, config: &KMeansConfig, rng: &mut impl Rng) -> VecSet<T> {
        let k = config.k;
        let dist = config.dist;
        let selected = match &config.selected {
            Some(selected) => selected.clone(),
            None => 0..vec_set.dim(),
        };
        let mut centroids = VecSet::zeros(selected.len(), k);
        centroids.put(
            0,
            &vec_set[rng.gen_range(0..vec_set.len())][selected.clone()],
        );
        let mut weight = vec![0_f32; vec_set.len()];
        for (v, w) in vec_set.iter().zip(weight.iter_mut()) {
            *w = dist.d(&centroids[0], &v[selected.clone()]);
        }
        for idx in 1..k {
            for (v, w) in vec_set.iter().zip(weight.iter_mut()) {
                *w = w.min(dist.d(&centroids[idx - 1], &v[selected.clone()]));
            }
            // Randomly choose the next centroid with probability proportional to the distance.
            // If all weights are zero, choose randomly.
            let c = WeightedIndex::new(&weight)
                .map(|d| d.sample(rng))
                .unwrap_or(rng.gen_range(0..vec_set.len()));

            centroids.put(idx, &vec_set[c][selected.clone()]);
        }
        centroids
    }
    /// Perform the k-means clustering on the given vector set.
    /// The initial centroids are initialized by the k-means++ algorithm.
    ///
    /// *May panic* when:
    /// - NaN or Inf exists since `f32` is partially ordered.
    /// - The selected range is out of the dimension range.
    /// - The number of clusters is 0.
    pub fn from_vec_set(vec_set: &VecSet<T>, config: &KMeansConfig, rng: &mut impl Rng) -> Self {
        assert!(
            config.k > 0,
            "The number of clusters should be greater than 0."
        );
        assert!(
            config.selected.is_none() || config.selected.as_ref().unwrap().end <= vec_set.dim(),
            "The selected range should be in the range [0, vec_set.dim())"
        );
        let mut centroids = Self::k_means_init(vec_set, config, rng);
        let selected = match &config.selected {
            Some(selected) => selected.clone(),
            None => 0..vec_set.dim(),
        };

        for _ in 0..config.max_iter {
            // Use f32 to avoid overflow when summing up.
            let mut new_centroids = VecSet::<f32>::zeros(selected.len(), config.k);
            let mut count = vec![0; config.k];

            for v in vec_set.iter() {
                let v = &v[selected.clone()];
                // Find the nearest centroid.
                // *May panic* since f32 is not Ord.
                let (_, min_idx) = centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (config.dist.d(v, c), i))
                    .min_by(|(d0, _), (d1, _)| d0.partial_cmp(d1).unwrap())
                    .unwrap();

                count[min_idx] += 1;
                let c = new_centroids.get_mut(min_idx);
                for (c, v) in c.iter_mut().zip(v.iter()) {
                    *c += v.cast_to_f32();
                }
            }
            for i in 0..config.k {
                let c = new_centroids.get_mut(i);
                if count[i] == 0 {
                    // If there is no vector assigned to the centroid, keep the centroid unchanged.
                    for (nc, c) in c.iter_mut().zip(centroids[i].iter()) {
                        *nc = c.cast_to_f32();
                    }
                    continue;
                }
                let n = count[i] as f32;
                c.iter_mut().for_each(|v| *v /= n);
            }
            let new_centroids = new_centroids.to_type::<T>();
            let mut max_diff = 0_f32;
            for (c, nc) in centroids.iter().zip(new_centroids.iter()) {
                max_diff = max_diff.max(config.dist.d(c, nc));
            }

            centroids = new_centroids;
            if max_diff < config.tol {
                break;
            }
        }
        Self {
            config: config.clone(),
            centroids,
        }
    }

    /// Find the nearest centroid to the given vector.
    /// This will use the selected range if it is specified in the config.
    ///
    /// *May panic* since f32 is not Ord or k is accidentally set to 0.
    pub fn find_nearest(&self, v: &[T]) -> usize {
        let dim = self.centroids.dim();
        let v = &v[self.config.selected.clone().unwrap_or(0..dim)];
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (self.config.dist.d(v, c), i))
            .min_by(|(d0, _), (d1, _)| d0.partial_cmp(d1).unwrap())
            .unwrap()
            .1
    }
}

#[cfg(test)]
mod test {

    use anyhow::Result;

    use crate::config::DBConfig;

    use super::*;

    #[test]
    fn test_k_means_init() {
        let dim = 2;
        let vec_set = VecSet::new(dim, vec![0.0, 0.0, 1.0, 0.0, -1.0, -2.0, -2.0, -1.0]);
        let k = 2;
        let config = KMeansConfig {
            k,
            max_iter: 0, // Not used in k_means_init.
            tol: 0.0,    // Not used in k_means_init.
            dist: DistanceAlgorithm::L2Sqr,
            selected: None,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let centroids = KMeans::k_means_init(&vec_set, &config, &mut rng);
        assert_eq!(centroids.len(), k);
        for c in centroids.iter() {
            assert_eq!(c.len(), dim);
            println!("{:?}", c);
        }
    }

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
        let k_means = KMeans::from_vec_set(&vec_set, &config, &mut rng);
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
        let k_means = KMeans::from_vec_set(&vec_set, &config, &mut rng);
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
        let file_path = "config/example/db_config.toml";
        let config = DBConfig::load_from_toml_file(file_path)?;
        let vec_set = VecSet::<f32>::load_with(&config.vec_data)?;
        let k_means_config = KMeansConfig {
            k: 3,
            max_iter: 20,
            tol: 1e-6,
            dist: DistanceAlgorithm::L2Sqr,
            selected: Some(0..2),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let k_means = KMeans::from_vec_set(&vec_set, &k_means_config, &mut rng);

        let k_means = KMeans::<f32>::try_from(k_means)?;
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
