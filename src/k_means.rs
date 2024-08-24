use crate::{
    binary_scalar::BinaryScalar,
    config::DistanceAlgorithm,
    distance::Distance,
    vec_set::{DynamicVecSet, VecSet},
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
impl<T: BinaryScalar> KMeans<T>
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
    /// Perform the k-means clustering on the given vector set.
    /// The initial centroids are initialized by the k-means++ algorithm.
    /// `k` should be in the range `[0, len(vec_set)]`.
    ///
    // *May panic* since f32 is not Ord.
    pub fn from_vec_set(vec_set: &VecSet<T>, config: &KMeansConfig, rng: &mut impl Rng) -> Self {
        assert!(
            (0..=vec_set.len()).contains(&config.k),
            "k in k-means should be in the range [0, len(vec_set)]"
        );
        let mut centroids = Self::k_means_init(vec_set, config, rng);

        for _ in 0..config.max_iter {
            // Use f32 to avoid overflow when summing up.
            let mut new_centroids = VecSet::<f32>::zeros(vec_set.dim(), config.k);
            let mut count = vec![0; config.k];

            for (idx, v) in vec_set.iter().enumerate() {
                // Find the nearest centroid.
                // *May panic* since f32 is not Ord.
                let (min_dist, min_idx) = centroids
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
        Self { centroids }
    }
}

pub struct DynamicKMeans {
    pub centroids: DynamicVecSet,
}

impl DynamicKMeans {
    pub fn from_vec_set(
        vec_set: &DynamicVecSet,
        config: &KMeansConfig,
        rng: &mut impl Rng,
    ) -> Self {
        use DynamicVecSet::*;
        match vec_set {
            Float32(vec_set) => {
                let k_means = KMeans::from_vec_set(vec_set, config, rng);
                Self {
                    centroids: Float32(k_means.centroids),
                }
            }
            UInt8(vec_set) => {
                let k_means = KMeans::from_vec_set(vec_set, config, rng);
                Self {
                    centroids: UInt8(k_means.centroids),
                }
            }
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
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let k_means = DynamicKMeans::from_vec_set(&vec_set, &k_means_config, &mut rng);
        assert_eq!(k_means.centroids.len(), k_means_config.k);
        for c in k_means.centroids.iter() {
            println!("{}", clip_msg(&format!("{:?}", c)));
        }
        Ok(())
    }
}
