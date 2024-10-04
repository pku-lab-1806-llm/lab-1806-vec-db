use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{
        pq_table::{PQConfig, PQTable},
        DistanceAdapter, DistanceAlgorithm,
    },
    index_algorithm::CandidatePair,
    scalar::Scalar,
    vec_set::VecSet,
};
use std::ops::Index;

use super::{prelude::*, IndexPQ, LinearIndex, ResultSet};

/// Linear index for the k-nearest neighbors search.
/// The distance algorithm is configurable.
///
/// Holds a reference to the `VecSet`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQLinearIndex<T> {
    /// The distance algorithm.
    pub dist: DistanceAlgorithm,
    /// Search Radius
    pub default_ef: usize,
    /// The vector set.
    pub vec_set: VecSet<T>,
    /// The encoded vector set.
    pub encoded_vec_set: VecSet<u8>,
    /// The PQ table.
    pub pq_table: PQTable<T>,
}
impl<T: Scalar> Index<usize> for PQLinearIndex<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec_set[index]
    }
}
impl<T: Scalar> IndexIter<T> for PQLinearIndex<T> {
    fn dim(&self) -> usize {
        self.vec_set.dim()
    }
    fn len(&self) -> usize {
        self.vec_set.len()
    }
}

impl<T: Scalar> IndexKNN<T> for PQLinearIndex<T> {
    fn knn(&self, query: &[T], k: usize) -> Vec<CandidatePair> {
        self.knn_with_ef(query, k, self.default_ef)
    }
}
impl<T: Scalar> IndexKNNWithEf<T> for PQLinearIndex<T> {
    fn set_default_ef(&mut self, ef: usize) {
        self.default_ef = ef;
    }
    fn knn_with_ef(&self, query: &[T], k: usize, ef: usize) -> Vec<CandidatePair> {
        let mut pq_result = ResultSet::new(ef);
        let dist = self.dist;
        let lookup = self.pq_table.create_lookup(query);
        for (i, v) in self.encoded_vec_set.iter().enumerate() {
            let d = dist.d(v, &lookup);
            pq_result.add(CandidatePair::new(i, d));
        }
        let mut result = ResultSet::new(k);
        for p in pq_result.into_sorted_vec() {
            let i = p.index;
            let v = &self[i];
            let d = dist.d(v, query);
            result.add(CandidatePair::new(i, d));
        }
        result.into_sorted_vec()
    }
}
impl<T: Scalar> IndexPQ for PQLinearIndex<T> {
    type NonPQIndex = LinearIndex<T>;
    fn pq_from(
        index: Self::NonPQIndex,
        pq_config: PQConfig,
        k_means_size: Option<usize>,
        rng: &mut impl Rng,
    ) -> Self {
        let vec_set = index.vec_set;
        let dist = index.dist;
        let default_ef = 100;
        assert_eq!(dist, pq_config.dist, "Distance algorithm mismatch.");

        let pq_table = match k_means_size {
            Some(k_means_size) => {
                let sub_vec_set = vec_set.random_sample(k_means_size, rng);
                PQTable::from_vec_set(&sub_vec_set, pq_config, rng)
            }
            None => PQTable::from_vec_set(&vec_set, pq_config, rng),
        };

        let encoded_vec_set = pq_table.encode_batch(&vec_set);
        Self {
            dist,
            default_ef,
            vec_set,
            encoded_vec_set,
            pq_table,
        }
    }
}
impl<T: Scalar> IndexSerde for PQLinearIndex<T> {}
impl<T: Scalar> IndexSerdeExternalVecSet<T> for PQLinearIndex<T> {
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

#[cfg(test)]
mod test {

    use anyhow::Result;
    use rand::SeedableRng;

    use crate::{
        config::VecDataConfig,
        index_algorithm::{candidate_pair::GroundTruthRow, linear_index, IndexFromVecSet},
    };

    use super::*;

    #[test]
    pub fn pq_linear_index_test() -> Result<()> {
        fn clip_msg(s: &str) -> String {
            if s.len() > 100 {
                format!("{}...", &s[..100])
            } else {
                s.to_string()
            }
        }
        let file_path = "config/gist_1000.toml";
        let config = VecDataConfig::load_from_toml_file(file_path)?;
        println!("Loaded config: {:#?}", config);
        let raw_vec_set = VecSet::<f32>::load_with(&config)?;
        let dist = DistanceAlgorithm::L2Sqr;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Limit the dimension for testing.
        let clipped_dim = raw_vec_set.dim().min(12);

        // Clipped vector set.
        let mut vec_set = VecSet::with_capacity(clipped_dim, raw_vec_set.len());
        for vec in raw_vec_set.iter() {
            vec_set.push(&vec[..clipped_dim]);
        }

        let pq_config = PQConfig {
            dist,
            n_bits: 4,
            m: clipped_dim / 4,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        };

        // Test the HNSWIndex by comparing with LinearIndex.
        let linear_index = linear_index::LinearIndex::from_vec_set(vec_set, dist, (), &mut rng);
        let index = PQLinearIndex::pq_from(
            linear_index.clone(),
            pq_config,
            Some(linear_index.len() / 10),
            &mut rng,
        );

        // Save and load the index. >>>>
        let path = "data/pq_linear_index.tmp.bin";
        let index = index.save_without_vec_set(&path)?;
        let vec_set = index.vec_set;

        let index = PQLinearIndex::<f32>::load_with_external_vec_set(&path, vec_set)?;
        // <<<< Save and load the index.

        let k = 10;
        let query_index = 200;

        println!("Query Index: {}", query_index);
        println!(
            "Query Vector: {}",
            clip_msg(&format!("{:?}", &index[query_index]))
        );

        let result = index.knn(&index[query_index], k);
        let linear_result = linear_index.knn(&linear_index[query_index], k);

        let gnd = GroundTruthRow::new(linear_result.iter().map(|p| p.index).collect());

        for res in result.iter() {
            println!("Index: {}, Distance: {}", res.index, res.distance);
            println!("Vector: {}", clip_msg(&format!("{:?}", &index[res.index])));
        }
        assert_eq!(result.len(), k.min(index.len()));

        assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));

        println!("Ground Truth: {:?}, Recall: {}", gnd, gnd.recall(&result));

        assert!(gnd.recall(&result) > 0.7, "Recall too low.");

        Ok(())
    }
}
