use rand::Rng;

use crate::{
    distance::DistanceAlgorithm,
    k_means::{KMeans, KMeansConfig},
    scalar::Scalar,
    vec_set::VecSet,
};
#[derive(Clone)]
pub struct IVFConfig {
    /// The number of clusters.
    pub k: usize,
    /// The distance algorithm.
    pub dist: DistanceAlgorithm,
    /// The number of iterations for the k-means algorithm.
    k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    k_means_tol: f32,
}

pub struct IVFIndex<T> {
    pub config: IVFConfig,
    /// The vector sets of the clusters. Length: k.
    pub clusters: Vec<VecSet<T>>,
    /// K-means struct for the centroids.
    pub k_means: KMeans<T>,
}

impl<T: Scalar> IVFIndex<T> {
    /// Create an IVF index from a `VecSet`.
    pub fn from_vec_set(vec_set: &VecSet<T>, config: &IVFConfig, rng: &mut impl Rng) -> Self {
        let k = config.k;
        let dist = config.dist;
        let k_means_config = KMeansConfig {
            k,
            max_iter: config.k_means_max_iter,
            tol: config.k_means_tol,
            dist,
            selected: None,
        };
        let k_means = KMeans::from_vec_set(vec_set, &k_means_config, rng);
        let mut clusters = vec![vec![]; k];
        for (i, v) in vec_set.iter().enumerate() {
            let idx = k_means.find_nearest(v);
            clusters[idx].push(i);
        }
        let clusters = clusters
            .into_iter()
            .map(|ids| {
                let mut cluster = VecSet::zeros(vec_set.dim(), ids.len());
                for (i, vec_id) in ids.iter().enumerate() {
                    cluster.put(i, &vec_set[*vec_id]);
                }
                cluster
            })
            .collect();
        IVFIndex {
            config: config.clone(),
            clusters,
            k_means,
        }
    }
}
