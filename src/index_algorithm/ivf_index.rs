use std::{collections::HashMap, ops::Index};

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[cfg(feature = "gpu")]
use crate::distance::gpu_dist::GpuVecSet;
use crate::{
    distance::{
        k_means::{KMeans, KMeansConfig},
        DistanceAlgorithm,
    },
    index_algorithm::ResultSet,
    prelude::DistanceAdapter,
    scalar::Scalar,
    vec_set::VecSet,
};

use super::{prelude::*, CandidatePair};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFConfig {
    /// The number of clusters.
    pub k: usize,
    /// The number of vectors to be sampled for the k-means algorithm.
    ///
    /// None for using all vectors in the dataset.
    pub k_means_size: Option<usize>,
    /// The number of iterations for the k-means algorithm.
    pub k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    pub k_means_tol: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFIndex<T> {
    /// The distance algorithm.
    pub dist: DistanceAlgorithm,
    /// The number of probes for the search.
    pub default_n_probes: usize,
    /// The original vector set.
    pub vec_set: VecSet<T>,
    /// The configuration of the index.
    pub config: IVFConfig,
    /// The vector index in the clusters.
    pub clusters: Vec<Vec<usize>>,
    /// K-means struct for the centroids.
    pub k_means: KMeans<T>,
}
impl<T: Scalar> Index<usize> for IVFIndex<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec_set[index]
    }
}
impl<T: Scalar> IndexIter<T> for IVFIndex<T> {
    fn dim(&self) -> usize {
        self.vec_set.dim()
    }
    fn len(&self) -> usize {
        self.vec_set.len()
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
        let k = config.k;
        let k_means_config = KMeansConfig {
            k,
            max_iter: config.k_means_max_iter,
            tol: config.k_means_tol,
            dist,
            selected: None,
        };
        let k_means = match config.k_means_size {
            Some(size) => {
                let sub_vec_set = vec_set.random_sample(size, rng);
                KMeans::from_vec_set(&sub_vec_set, k_means_config, rng)
            }
            None => KMeans::from_vec_set(&vec_set, k_means_config, rng),
        };
        let mut clusters = vec![vec![]; k];
        let mut index_map = HashMap::new();
        for (i, v) in vec_set.iter().enumerate() {
            let idx = k_means.find_nearest(v);
            index_map.insert(i, (idx, clusters[idx].len()));
            clusters[idx].push(i);
        }
        let default_n_probes = 4;
        IVFIndex {
            dist,
            default_n_probes,
            vec_set,
            config,
            clusters,
            k_means,
        }
    }
}
impl<T: Scalar> IndexSerde for IVFIndex<T> {}
impl<T: Scalar> IndexSerdeExternalVecSet<T> for IVFIndex<T> {
    fn save_without_vec_set(mut self, path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        // Move the vec_set out of the index.
        let vec_set = self.vec_set;
        self.vec_set = VecSet::new(vec_set.dim(), vec![]);

        // Call the original save method.
        self.save(path)?;

        // Restore the vec_set.
        self.vec_set = vec_set;
        Ok(self)
    }
    fn load_with_external_vec_set(
        path: impl AsRef<std::path::Path>,
        vec_set: VecSet<T>,
    ) -> anyhow::Result<Self> {
        let mut index = Self::load(path)?;
        index.vec_set = vec_set;
        Ok(index)
    }
}

impl<T: Scalar> IndexKNN<T> for IVFIndex<T> {
    fn knn(&self, query: &[T], k: usize) -> Vec<CandidatePair> {
        self.knn_with_ef(query, k, self.default_n_probes)
    }
}
impl<T: Scalar> IndexKNNWithEf<T> for IVFIndex<T> {
    /// `ef` for IVF is exactly the number of probes.
    fn set_default_ef(&mut self, n_probes: usize) {
        self.default_n_probes = n_probes;
    }
    /// `ef` for IVF is exactly the number of probes.
    fn knn_with_ef(&self, query: &[T], k: usize, n_probes: usize) -> Vec<CandidatePair> {
        let clusters = self.k_means.find_n_nearest(query, n_probes);
        let dist = self.dist;
        let mut result_set = ResultSet::new(k);
        for cluster in clusters.iter().map(|&idx| &self.clusters[idx]) {
            for &i in cluster {
                let v = &self[i];
                result_set.add(CandidatePair::new(i, dist.d(v, query)));
            }
        }
        result_set.into_sorted_vec()
    }
}

#[cfg(feature = "gpu")]
pub struct GpuIVFCache {
    pub centroids: GpuVecSet,
    pub clusters: Vec<GpuVecSet>,
}

#[cfg(feature = "gpu")]
impl<T: Scalar> IndexGpuKNNWithEf<T> for IVFIndex<T> {
    fn gpu_knn_with_ef(
        &self,
        gpu_cache: &Self::GpuCache,
        query: &[T],
        k: usize,
        ef: usize,
    ) -> Result<Vec<CandidatePair>> {
        let dist = self.dist;
        let query = gpu_cache.centroids.parse_query(query)?;
        let distance = gpu_cache
            .centroids
            .batch_distance(&query, dist)?
            .to_vec1()?;
        let mut centroids_result = ResultSet::new(ef);
        for (i, d) in distance.into_iter().enumerate() {
            centroids_result.add(CandidatePair::new(i, d));
        }
        let activated = centroids_result
            .into_sorted_vec()
            .into_iter()
            .map(|p| p.index);
        let mut result = ResultSet::new(k);
        for cluster_idx in activated {
            let cluster = &gpu_cache.clusters[cluster_idx];
            let distance = cluster.batch_distance(&query, dist)?.to_vec1()?;
            for (i, d) in distance.into_iter().enumerate() {
                let index = self.clusters[cluster_idx][i];
                result.add(CandidatePair::new(index, d));
            }
        }
        Ok(result.into_sorted_vec())
    }
}

#[cfg(feature = "gpu")]
impl<T: Scalar> IndexGpuKNN<T> for IVFIndex<T> {
    type GpuCache = GpuIVFCache;
    fn build_gpu_cache(&self) -> Result<Self::GpuCache> {
        let dim = self.dim();
        let centroids = GpuVecSet::try_from(&self.k_means.centroids)?;
        let clusters = self
            .clusters
            .iter()
            .map(|cluster| {
                let len = cluster.len();
                let mut vec_set = VecSet::with_capacity(dim, len);
                for &i in cluster {
                    vec_set.push(&self[i]);
                }
                GpuVecSet::try_from(&vec_set)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(GpuIVFCache {
            centroids,
            clusters,
        })
    }
    fn gpu_knn(
        &self,
        gpu_cache: &Self::GpuCache,
        query: &[T],
        k: usize,
    ) -> Result<Vec<CandidatePair>> {
        self.gpu_knn_with_ef(gpu_cache, query, k, self.default_n_probes)
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use crate::{config::VecDataConfig, index_algorithm::LinearIndex};
    use anyhow::{Ok, Result};
    use rand::prelude::*;

    use super::*;

    #[test]
    pub fn ivf_index_test() -> Result<()> {
        fn clip_msg(s: &str) -> String {
            if s.len() > 100 {
                format!("{}...", &s[..100])
            } else {
                s.to_string()
            }
        }
        // The `dim` and `limit` has been limited for debug mode performance.

        let file_path = "config/gist_1000.toml";
        let config = VecDataConfig::load_from_toml_file(file_path)?;

        let raw_vec_set = VecSet::<f32>::load_with(&config)?;

        let clipped_dim = raw_vec_set.dim().min(12);

        let mut vec_set = VecSet::with_capacity(clipped_dim, raw_vec_set.len());
        for vec in raw_vec_set.iter() {
            vec_set.push(&vec[..clipped_dim]);
        }
        let dist = DistanceAlgorithm::L2Sqr;
        let ivf_config = IVFConfig {
            k: 7,
            k_means_size: Some(vec_set.len() / 10),
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let linear_index = LinearIndex::from_vec_set(vec_set.clone(), dist, (), &mut rng);

        let index = IVFIndex::from_vec_set(vec_set, dist, ivf_config, &mut rng);

        // Save and load the index. >>>>
        let path = "data/ivf_index.tmp.bin";
        let index = index.save_without_vec_set(&path)?;
        let vec_set = index.vec_set;

        let index = IVFIndex::<f32>::load_with_external_vec_set(&path, vec_set)?;
        // <<<< Save and load the index.

        for (id, cluster) in index.clusters.iter().enumerate() {
            println!("cluster id: {}, cluster size: {}", id, cluster.len());
        }

        let k = 6;
        let query_index = 200;

        println!("Query Index: {}", query_index);
        println!(
            "Query Vector: {}",
            clip_msg(&format!("{:?}", &index[query_index]))
        );

        let result = index.knn(&index[query_index], k);
        let linear_result = linear_index.knn(&linear_index[query_index], k);

        for (res, l_res) in result.iter().zip(linear_result.iter()) {
            println!("Index: {}, Distance: {}", res.index, res.distance);
            println!("Vector: {}", clip_msg(&format!("{:?}", &index[res.index])));
            assert_eq!(res.index, l_res.index, "Index mismatch");
        }
        assert_eq!(result.len(), k.min(index.len()));

        assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));
        fs::remove_file(path)?;
        Ok(())
    }
}
