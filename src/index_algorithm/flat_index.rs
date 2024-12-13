use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{DistanceAdapter, DistanceAlgorithm},
    index_algorithm::CandidatePair,
    scalar::Scalar,
    vec_set::VecSet,
};
use std::{ops::Index, path::Path};

use super::{prelude::*, ResultSet};

/// Flat index for the k-nearest neighbors search.
/// The distance algorithm is configurable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndex<T> {
    /// The distance algorithm.
    pub(crate) dist: DistanceAlgorithm,
    /// The vector set.
    pub(crate) vec_set: VecSet<T>,
}
impl<T: Scalar> FlatIndex<T> {
    pub fn new(dim: usize, dist: DistanceAlgorithm) -> Self {
        Self {
            dist,
            vec_set: VecSet::new(dim, vec![]),
        }
    }
}
impl<T: Scalar> Index<usize> for FlatIndex<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec_set[index]
    }
}
impl<T: Scalar> IndexIter<T> for FlatIndex<T> {
    fn dim(&self) -> usize {
        self.vec_set.dim()
    }
    fn len(&self) -> usize {
        self.vec_set.len()
    }
}

impl<T: Scalar> IndexKNN<T> for FlatIndex<T> {
    fn knn(&self, query: &[T], k: usize) -> Vec<CandidatePair> {
        let mut result = ResultSet::new(k);
        for (i, v) in self.vec_set.iter().enumerate() {
            let dist = self.dist.d(query, v);
            result.add(CandidatePair::new(i, dist));
        }
        result.into_sorted_vec()
    }
}

impl<T: Scalar> IndexFromVecSet<T> for FlatIndex<T> {
    type Config = ();

    fn from_vec_set(
        vec_set: VecSet<T>,
        dist: DistanceAlgorithm,
        _: Self::Config,
        _: &mut impl Rng,
    ) -> Self {
        Self { dist, vec_set }
    }
}
impl<T: Scalar> IndexSerde for FlatIndex<T> {}
impl<T: Scalar> IndexSerdeExternalVecSet<T> for FlatIndex<T> {
    fn save_without_vec_set(self, path: impl AsRef<Path>) -> Result<Self> {
        let mut file = std::fs::File::create(path)?;
        bincode::serialize_into(&mut file, &self.dist)?;
        Ok(self)
    }
    fn load_with_external_vec_set(path: impl AsRef<Path>, vec_set: VecSet<T>) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let dist: DistanceAlgorithm = bincode::deserialize_from(&mut file)?;
        Ok(Self { dist, vec_set })
    }
}
impl<T: Scalar> IndexPQ<T> for FlatIndex<T> {
    fn knn_pq(
        &self,
        query: &[T],
        k: usize,
        ef: usize,
        pq_table: &crate::distance::pq_table::PQTable<T>,
    ) -> Vec<CandidatePair> {
        assert_eq!(
            self.dist, pq_table.config.dist,
            "Distance algorithm mismatch."
        );
        let mut pq_result = ResultSet::new(ef.max(k));
        let lookup = pq_table.create_lookup(query);
        for (i, v) in pq_table.encoded_vec_set.iter().enumerate() {
            let d = self.dist.d(v, &lookup);
            pq_result.add(CandidatePair::new(i, d));
        }
        pq_result.pq_resort(k, |idx| self.dist.d(query, &self[idx]))
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use anyhow::Result;
    use rand::SeedableRng;

    use crate::config::VecDataConfig;

    use super::*;

    #[test]
    pub fn flat_index_test() -> Result<()> {
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

        let clipped_dim = raw_vec_set.dim().min(12);
        let mut vec_set = VecSet::with_capacity(clipped_dim, raw_vec_set.len());
        for vec in raw_vec_set.iter() {
            vec_set.push(&vec[..clipped_dim]);
        }

        let index = FlatIndex::from_vec_set(vec_set, dist, (), &mut rng);

        // Save and load the index. >>>>
        let path = "data/flat_index.tmp.bin";
        let vec_set = index.save_without_vec_set(path)?.vec_set;

        let index = FlatIndex::<f32>::load_with_external_vec_set(path, vec_set)?;
        // <<<< Save and load the index.

        let k = 4;
        let query_index = 200;

        println!("Query Index: {}", query_index);
        println!(
            "Query Vector: {}",
            clip_msg(&format!("{:?}", &index[query_index]))
        );

        let result = index.knn(&index[query_index], k);

        for res in result.iter() {
            println!("Index: {}, Distance: {}", res.index, res.distance);
            println!("Vector: {}", clip_msg(&format!("{:?}", &index[res.index])));
        }
        assert_eq!(result.len(), k.min(index.len()));
        assert_eq!(result[0].index, query_index);
        assert!(result[0].distance.abs() < 1e-6);

        assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));
        fs::remove_file(path)?;
        Ok(())
    }
}
