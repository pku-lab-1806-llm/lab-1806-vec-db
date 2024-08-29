use std::collections::BTreeSet;

use crate::binary_scalar::BinaryScalar;
use crate::config::DistanceAlgorithm;
use crate::distance::Distance;
use crate::vec_set::{DynamicVecRef, DynamicVecSet, VecSet};

/// A pair of the index and the distance.
/// For the response of the k-nearest neighbors search.
///
/// This should implement `Ord` to be used in `BTreeSet`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponsePair {
    pub index: usize,
    pub distance: f32,
}
impl ResponsePair {
    pub fn new(index: usize, distance: f32) -> Self {
        Self { index, distance }
    }
}
impl Eq for ResponsePair {}
impl PartialOrd for ResponsePair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
impl Ord for ResponsePair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .expect("Failed to compare f32 distance in response pair.")
    }
}

pub mod linear_index;
use linear_index::DynamicLinearIndex as DynamicLinearIndex;

pub mod ivf_index;

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
        let vec_set = DynamicVecSet::load_with(config.vec_data)?;
        let index = DynamicLinearIndex::from_dynamic_vec_set(&vec_set, config.distance);

        let k = 4;
        let query_index = 200;

        println!("Query Index: {}", query_index);
        println!(
            "Query Vector: {}",
            clip_msg(&format!("{:?}", vec_set.i(query_index)))
        );

        let result = index.knn(&vec_set.i(query_index), k);

        for res in result.iter() {
            println!("Index: {}, Distance: {}", res.index, res.distance);
            println!(
                "Vector: {}",
                clip_msg(&format!("{:?}", vec_set.i(res.index)))
            );
        }
        assert_eq!(result.len(), k.min(vec_set.len()));
        assert_eq!(result[0].index, query_index);
        assert!(result[0].distance.abs() < 1e-6);

        assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));
        Ok(())
    }
}
