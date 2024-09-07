use std::rc::Rc;

use rand::Rng;

use crate::{
    distance::DistanceAlgorithm,
    k_means::{KMeans, KMeansConfig},
    scalar::Scalar,
    vec_set::VecSet,
};

use super::{IndexAlgorithm, ResponsePair};
#[derive(Debug, Clone)]
pub struct IVFIndexConfig {
    /// The number of clusters.
    pub k: usize,
    /// The distance algorithm.
    pub dist: DistanceAlgorithm,
    /// The number of iterations for the k-means algorithm.
    pub k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    pub k_means_tol: f32,
}

pub struct IVFIndex<T> {
    pub config: Rc<IVFIndexConfig>,
    /// The vector sets of the clusters. Length: k.
    pub clusters: Vec<VecSet<T>>,
    /// K-means struct for the centroids.
    pub k_means: KMeans<T>,
}

impl<T: Scalar> IndexAlgorithm<T> for IVFIndex<T> {
    type Config = IVFIndexConfig;

    fn from_vec_set(vec_set: Rc<VecSet<T>>, config: Rc<Self::Config>, rng: &mut impl Rng) -> Self {
        let k = config.k;
        let dist = config.dist;
        let k_means_config = Rc::new(KMeansConfig {
            k,
            max_iter: config.k_means_max_iter,
            tol: config.k_means_tol,
            dist,
            selected: None,
        });
        let k_means = KMeans::from_vec_set(&vec_set, k_means_config, rng);
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
            config,
            clusters,
            k_means,
        }
    }

    fn knn(&self, _: &[T], _: usize) -> Vec<ResponsePair> {
        unimplemented!("IVFIndex::knn is not implemented yet.");
    }
}

#[cfg(test)]
mod test {
    use crate::config::DBConfig;
    use anyhow::{Ok, Result};
    use rand::prelude::*;

    use super::*;

    #[test]
    pub fn ivf_index_test() -> Result<()> {
        // The `dim` and `limit` has been limited for debug mode performance.

        let file_path = "config/example/db_config.toml";
        let mut config = DBConfig::load_from_toml_file(file_path)?;

        config.vec_data.limit = Some(64);

        let raw_vec_set = VecSet::<f32>::load_with(&config.vec_data)?;

        let clipped_dim = raw_vec_set.dim().min(12);

        let mut vec_set = VecSet::zeros(clipped_dim, raw_vec_set.len());
        for i in 0..raw_vec_set.len() {
            let src = &raw_vec_set[i];
            let dst = vec_set.get_mut(i);
            dst.copy_from_slice(&src[..clipped_dim]);
        }

        let ivf_config = Rc::new(IVFIndexConfig {
            k: 3,
            dist: DistanceAlgorithm::L2Sqr,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        });
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vec_set = Rc::new(vec_set);

        let index = IVFIndex::from_vec_set(vec_set, ivf_config, &mut rng);
        for (id, cluster) in index.clusters.iter().enumerate() {
            println!("cluster id: {}, cluster size: {}", id, cluster.len());
        }
        Ok(())
    }
}
