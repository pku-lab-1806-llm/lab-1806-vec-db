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

/// Linear index for the k-nearest neighbors search.
/// The distance algorithm is configurable.
///
/// Holds a reference to the `VecSet`.
pub struct LinearIndex<'a, T> {
    vec_set: &'a VecSet<T>,
    distance: DistanceAlgorithm,
}

impl<'a, T> LinearIndex<'a, T>
where
    T: BinaryScalar,
    [T]: Distance,
{
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

/// Linear index for the k-nearest neighbors search.
///
/// Holds a reference to the `DynamicVecSet`.
///
/// The data type is determined at runtime.
pub enum DynamicLinearIndex<'a> {
    Float32(LinearIndex<'a, f32>),
    UInt8(LinearIndex<'a, u8>),
}

impl<'a> DynamicLinearIndex<'a> {
    pub fn from_dynamic_vec_set(vec_set: &'a DynamicVecSet, distance: DistanceAlgorithm) -> Self {
        match vec_set {
            DynamicVecSet::Float32(vec_set) => {
                Self::Float32(LinearIndex::from_vec_set(vec_set, distance))
            }
            DynamicVecSet::UInt8(vec_set) => {
                Self::UInt8(LinearIndex::from_vec_set(vec_set, distance))
            }
        }
    }
    pub fn knn(&self, query: &DynamicVecRef, k: usize) -> Vec<ResponsePair> {
        match (self, query) {
            (Self::Float32(index), DynamicVecRef::Float32(query)) => index.knn(query, k),
            (Self::UInt8(index), DynamicVecRef::UInt8(query)) => index.knn(query, k),
            _ => panic!("Mismatched types when calling knn in DynamicLinearIndex."),
        }
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
