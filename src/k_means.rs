use std::ops::Range;

use crate::{
    binary_scalar::BinaryScalar,
    config::DistanceAlgorithm,
    distance::Distance,
    vec_set::{DynamicVecRef, DynamicVecSet, VecSet},
};
use anyhow::{bail, Result};
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
impl<T: BinaryScalar> KMeans<T>
where
    [T]: Distance,
{
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
    /// `k` should be in the range `[0, len(vec_set)]`.
    ///
    // *May panic* since f32 is partially ordered.
    pub fn from_vec_set(vec_set: &VecSet<T>, config: &KMeansConfig, rng: &mut impl Rng) -> Self {
        assert!(
            (0..=vec_set.len()).contains(&config.k),
            "k in k-means should be in the range [0, len(vec_set)]"
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
                for v in c.iter_mut() {
                    *v /= count[i] as f32;
                }
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
    /// *May panic* since f32 is not Ord.
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

#[derive(Debug)]
pub enum DynamicKMeans {
    Float32(KMeans<f32>),
    UInt8(KMeans<u8>),
}

impl DynamicKMeans {
    /// Perform the k-means clustering on the given vector set.
    /// The initial centroids are initialized by the k-means++ algorithm.
    /// `k` should be in the range `[0, len(vec_set)]`.
    ///
    // *May panic* since f32 is partially ordered.
    pub fn from_vec_set(
        vec_set: &DynamicVecSet,
        config: &KMeansConfig,
        rng: &mut impl Rng,
    ) -> Self {
        use DynamicVecSet::*;
        match vec_set {
            Float32(vec_set) => {
                let k_means = KMeans::from_vec_set(vec_set, config, rng);
                Self::Float32(k_means)
            }
            UInt8(vec_set) => {
                let k_means = KMeans::from_vec_set(vec_set, config, rng);
                Self::UInt8(k_means)
            }
        }
    }
    /// Get the config.
    pub fn config(&self) -> &KMeansConfig {
        match self {
            Self::Float32(k_means) => &k_means.config,
            Self::UInt8(k_means) => &k_means.config,
        }
    }

    pub fn find_nearest(&self, v: DynamicVecRef) -> usize {
        use DynamicVecRef::*;
        match (self, v) {
            (Self::Float32(k_means), Float32(v)) => k_means.find_nearest(v),
            (Self::UInt8(k_means), UInt8(v)) => k_means.find_nearest(v),
            _ => panic!("The vector type does not match the k-means type."),
        }
    }
}

impl From<KMeans<f32>> for DynamicKMeans {
    fn from(k_means: KMeans<f32>) -> Self {
        Self::Float32(k_means)
    }
}

impl From<KMeans<u8>> for DynamicKMeans {
    fn from(k_means: KMeans<u8>) -> Self {
        Self::UInt8(k_means)
    }
}

impl TryFrom<DynamicKMeans> for KMeans<f32> {
    type Error = anyhow::Error;
    fn try_from(k_means: DynamicKMeans) -> Result<KMeans<f32>> {
        match k_means {
            DynamicKMeans::Float32(k_means) => Ok(k_means),
            _ => bail!("Failed to convert to KMeans<f32>."),
        }
    }
}

impl TryFrom<DynamicKMeans> for KMeans<u8> {
    type Error = anyhow::Error;
    fn try_from(k_means: DynamicKMeans) -> Result<KMeans<u8>> {
        match k_means {
            DynamicKMeans::UInt8(k_means) => Ok(k_means),
            _ => bail!("Failed to convert to KMeans<u8>."),
        }
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
        let vec_set = VecSet::new(
            dim,
            vec![0.0, 0.0, 1.0, 0.0, -1.0, -2.0, -2.0, -1.0].into_boxed_slice(),
        );
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
        let vec_set = VecSet::new(
            2,
            vec![0.0, 0.0, 1.0, 0.0, -1.0, -2.0, -2.0, -1.0].into_boxed_slice(),
        );
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
        let vec_set = VecSet::new(2, vec![0, 0, 1, 0, 255, 254, 255, 255].into_boxed_slice());
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
        let vec_set = DynamicVecSet::load_with(config.vec_data)?;
        let k_means_config = KMeansConfig {
            k: 3,
            max_iter: 20,
            tol: 1e-6,
            dist: DistanceAlgorithm::L2Sqr,
            selected: Some(0..2),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let k_means = DynamicKMeans::from_vec_set(&vec_set, &k_means_config, &mut rng);
        let k_means = KMeans::<f32>::try_from(k_means)?;
        assert_eq!(k_means.centroids.len(), k_means_config.k);
        for c in k_means.centroids.iter() {
            println!("{}", clip_msg(&format!("{:?}", c)));
        }
        Ok(())
    }
}
