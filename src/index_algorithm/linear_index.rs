use crate::distance::{DistanceAdapter, DistanceAlgorithm};
use crate::index_algorithm::ResponsePair;
use crate::scalar::Scalar;
use crate::vec_set::VecSet;
use std::collections::BTreeSet;

/// Linear index for the k-nearest neighbors search.
/// The distance algorithm is configurable.
///
/// Holds a reference to the `VecSet`.
pub struct LinearIndex<'a, T> {
    vec_set: &'a VecSet<T>,
    distance: DistanceAlgorithm,
}

impl<'a, T: Scalar> LinearIndex<'a, T> {
    pub fn from_vec_set(vec_set: &'a VecSet<T>, distance: DistanceAlgorithm) -> Self {
        Self { vec_set, distance }
    }
    /// Get the precise k-nearest neighbors.
    /// Returns a vector of pairs of the index and the distance.
    /// The vector is sorted by the distance in ascending order.
    pub fn knn(&self, query: &[T], k: usize) -> Vec<ResponsePair> {
        let mut result = BTreeSet::new();
        for (i, v) in self.vec_set.iter().enumerate() {
            let dist = self.distance.d(query, v);
            if result.len() < k {
                result.insert(ResponsePair::new(i, dist));
            } else if let Some(max) = result.last() {
                if dist < max.distance {
                    result.insert(ResponsePair::new(i, dist));
                    result.pop_last();
                }
            }
        }
        result.into_iter().collect()
    }
}

#[cfg(test)]
mod test {
    use anyhow::Result;

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
        let vec_set = VecSet::<f32>::load_with(&config.vec_data)?;

        let index = LinearIndex::from_vec_set(&vec_set, config.distance);

        let k = 4;
        let query_index = 200;

        println!("Query Index: {}", query_index);
        println!(
            "Query Vector: {}",
            clip_msg(&format!("{:?}", &vec_set[query_index]))
        );

        let result = index.knn(&vec_set[query_index], k);

        for res in result.iter() {
            println!("Index: {}, Distance: {}", res.index, res.distance);
            println!(
                "Vector: {}",
                clip_msg(&format!("{:?}", &vec_set[res.index]))
            );
        }
        assert_eq!(result.len(), k.min(vec_set.len()));
        assert_eq!(result[0].index, query_index);
        assert!(result[0].distance.abs() < 1e-6);

        assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));
        Ok(())
    }
}
