use std::{collections::HashMap, ops::Index};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{
        k_means::{KMeans, KMeansConfig},
        DistanceAlgorithm,
    },
    scalar::Scalar,
    vec_set::VecSet,
};

use super::prelude::*;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFConfig {
    /// The number of clusters.
    pub k: usize,
    /// The number of iterations for the k-means algorithm.
    pub k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    pub k_means_tol: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFIndex<T> {
    /// The distance algorithm.
    pub dist: DistanceAlgorithm,
    /// The configuration of the index.
    pub config: IVFConfig,
    /// The vector sets of the clusters. Length: k.
    pub clusters: Vec<VecSet<T>>,
    /// K-means struct for the centroids.
    pub k_means: KMeans<T>,
    /// The number of vectors in the index.
    pub num_vec: usize,
    /// Map index -> (cluster_id, cluster_index).
    pub index_map: HashMap<usize, (usize, usize)>,
}
impl<T: Scalar> Index<usize> for IVFIndex<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let (cluster_id, cluster_index) = self.index_map[&index];
        &self.clusters[cluster_id][cluster_index]
    }
}
impl<T: Scalar> IndexIter<T> for IVFIndex<T> {
    fn dim(&self) -> usize {
        self.k_means.centroids.dim()
    }
    fn len(&self) -> usize {
        self.num_vec
    }
}

impl<T: Scalar> IndexFromVecSet<T> for IVFIndex<T> {
    type Config = IVFConfig;

    fn from_vec_set(
        vec_set: VecSet<T>,
        dist: DistanceAlgorithm,
        config: Self::Config,
        rng: &mut impl Rng,
    ) -> Self {
        let num_vec = vec_set.len();
        let k = config.k;
        let k_means_config = KMeansConfig {
            k,
            max_iter: config.k_means_max_iter,
            tol: config.k_means_tol,
            dist,
            selected: None,
        };
        let k_means = KMeans::from_vec_set(&vec_set, k_means_config, rng);
        let mut clusters = vec![vec![]; k];
        let mut index_map = HashMap::new();
        for (i, v) in vec_set.iter().enumerate() {
            let idx = k_means.find_nearest(v);
            index_map.insert(i, (idx, clusters[idx].len()));
            clusters[idx].push(i);
        }
        let clusters = clusters
            .into_iter()
            .map(|ids| {
                let mut cluster = VecSet::with_capacity(vec_set.dim(), ids.len());
                for id in ids {
                    cluster.push(&vec_set[id]);
                }
                cluster
            })
            .collect();
        IVFIndex {
            dist,
            config,
            clusters,
            k_means,
            num_vec,
            index_map,
        }
    }
}
impl<T: Scalar> IndexSerde for IVFIndex<T> {}
#[cfg(test)]
mod test {
    use std::fs;

    use crate::config::DBConfig;
    use anyhow::{Ok, Result};
    use rand::prelude::*;

    use super::*;

    #[test]
    pub fn ivf_index_test() -> Result<()> {
        // The `dim` and `limit` has been limited for debug mode performance.

        let file_path = "config/db_config.toml";
        let mut config = DBConfig::load_from_toml_file(file_path)?;

        config.vec_data.limit = Some(64);

        let raw_vec_set = VecSet::<f32>::load_with(&config.vec_data)?;

        let clipped_dim = raw_vec_set.dim().min(12);

        let mut vec_set = VecSet::with_capacity(clipped_dim, raw_vec_set.len());
        for vec in raw_vec_set.iter() {
            vec_set.push(&vec[..clipped_dim]);
        }
        let dist = DistanceAlgorithm::L2Sqr;
        let ivf_config = IVFConfig {
            k: 3,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let index = IVFIndex::from_vec_set(vec_set, dist, ivf_config, &mut rng);

        // Save and load the index. >>>>
        let path = "data/ivf_index.test.bin";
        index.save(path)?;

        let index = IVFIndex::<f32>::load(path)?;
        // <<<< Save and load the index.

        for (id, cluster) in index.clusters.iter().enumerate() {
            println!("cluster id: {}, cluster size: {}", id, cluster.len());
        }
        fs::remove_file(path)?;
        Ok(())
    }
}
