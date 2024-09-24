use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{DistanceAdapter, DistanceAlgorithm},
    index_algorithm::CandidatePair,
    scalar::Scalar,
    vec_set::VecSet,
};
use std::ops::Index;

use super::{IndexFromVecSet, IndexIter, IndexKNN, IndexSerde, ResultSet};

/// Linear index for the k-nearest neighbors search.
/// The distance algorithm is configurable.
///
/// Holds a reference to the `VecSet`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearIndex<T> {
    /// The distance algorithm.
    pub dist: DistanceAlgorithm,
    /// The vector set.
    pub vec_set: VecSet<T>,
}
impl<T: Scalar> Index<usize> for LinearIndex<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec_set[index]
    }
}
impl<T: Scalar> IndexIter<T> for LinearIndex<T> {
    fn dim(&self) -> usize {
        self.vec_set.dim()
    }
    fn len(&self) -> usize {
        self.vec_set.len()
    }
}

impl<T: Scalar> IndexKNN<T> for LinearIndex<T> {
    fn knn(&self, query: &[T], k: usize) -> Vec<CandidatePair> {
        let mut result = ResultSet::new(k);
        for (i, v) in self.vec_set.iter().enumerate() {
            let dist = self.dist.d(query, v);
            result.add(CandidatePair::new(i, dist));
        }
        result.into_sorted_vec()
    }
}

impl<T: Scalar> IndexFromVecSet<T> for LinearIndex<T> {
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
impl<T: Scalar> IndexSerde for LinearIndex<T> {}

#[cfg(test)]
mod test {
    use anyhow::Result;
    use rand::SeedableRng;

    use crate::config::DBConfig;

    use super::*;

    #[test]
    pub fn linear_index_test() -> Result<()> {
        fn clip_msg(s: &str) -> String {
            if s.len() > 100 {
                format!("{}...", &s[..100])
            } else {
                s.to_string()
            }
        }
        let file_path = "config/example/db_config.toml";
        let config = DBConfig::load_from_toml_file(file_path)?;
        println!("Loaded config: {:#?}", config);
        let raw_vec_set = VecSet::<f32>::load_with(&config.vec_data)?;
        let dist = DistanceAlgorithm::L2Sqr;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let clipped_dim = raw_vec_set.dim().min(12);
        let mut vec_set = VecSet::with_capacity(clipped_dim, raw_vec_set.len());
        for vec in raw_vec_set.iter() {
            vec_set.push(&vec[..clipped_dim]);
        }

        let index = LinearIndex::from_vec_set(vec_set, dist, (), &mut rng);

        // Save and load the index. >>>>
        let path = "config/example/linear_index.test.bin";
        index.save(path)?;

        let index = LinearIndex::<f32>::load(path)?;
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
        Ok(())
    }
}
