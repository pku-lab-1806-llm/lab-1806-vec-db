use std::{borrow::Borrow, fs::File, io::BufWriter, ops::Index, path::Path};

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{pq_table::PQTable, DistanceAlgorithm},
    scalar::Scalar,
    vec_set::VecSet,
};
pub mod prelude {
    // All Index Traits
    pub use super::{
        IndexBuilder, IndexFromVecSet, IndexIter, IndexKNN, IndexKNNWithEf, IndexPQ, IndexSerde,
        IndexSerdeExternalVecSet,
    };
}

pub mod candidate_pair;
pub mod flat_index;
pub mod hnsw_index;
pub mod ivf_index;
// All Index Algorithms & Candidate Tools
pub use self::{
    candidate_pair::{CandidatePair, ResultSet},
    flat_index::FlatIndex,
    hnsw_index::{HNSWConfig, HNSWIndex},
    ivf_index::{IVFConfig, IVFIndex},
};

/// The trait for index that can be accessed by the index.
///
/// Automatically implement `iter` method.
pub trait IndexIter<T: Scalar>: Index<usize, Output = [T]> {
    /// Get the number of vectors in the index.
    fn len(&self) -> usize;
    /// Check if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get the dimension of the vectors.
    fn dim(&self) -> usize;

    /// Get an iterator of the vectors.
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a [T]>
    where
        T: 'a,
    {
        (0..self.len()).map(move |i| &self[i])
    }
}

/// The trait for index that can be built by adding vectors one by one.
pub trait IndexBuilder<T: Scalar>: IndexIter<T> {
    /// The Builder configuration of the index.
    type Config;
    /// Create a new index.
    fn new(dim: usize, dist: DistanceAlgorithm, config: Self::Config) -> Self;
    /// Add a vector to the index.
    ///
    /// Returns the index of the vector.
    fn add(&mut self, vec: &[T], rng: &mut impl Rng) -> usize;

    /// Add multiple vectors to the index.
    ///
    /// Returns a vector of the indices of the vectors.
    ///
    /// May lead to better performance than adding vectors one by one.
    fn batch_add(&mut self, vec_list: &[&[T]], rng: &mut impl Rng) -> Vec<usize>;

    /// Same as `batch_add`, but show a progress bar.
    fn batch_add_process(&mut self, vec_list: &[&[T]], rng: &mut impl Rng) -> Vec<usize>;

    /// Build the index from a `VecSet`.
    fn build_on_vec_set(
        vec_set: impl Borrow<VecSet<T>>,
        dist: DistanceAlgorithm,
        config: Self::Config,
        process_bar: bool,
        rng: &mut impl Rng,
    ) -> Self;
}

/// The trait for index that can search the k-nearest neighbors.
pub trait IndexKNN<T: Scalar>: IndexIter<T> {
    /// Get the precise k-nearest neighbors.
    /// Returns a vector of pairs of the index and the distance.
    /// The vector is sorted by the distance in ascending order.
    fn knn(&self, query: &[T], k: usize) -> Vec<CandidatePair>;
}

/// The trait for index that can search the k-nearest neighbors with a search radius `ef`.
pub trait IndexKNNWithEf<T: Scalar>: IndexKNN<T> {
    /// Set the default search radius `ef`.
    ///
    /// `knn` will use `ef.max(k)` as the search radius by default.
    /// `knn_with_ef` will not be affected.
    fn set_default_ef(&mut self, ef: usize);

    /// Same as `knn`, but with a search radius `ef`.
    /// When ef < k, ef will be set to k.
    fn knn_with_ef(&self, query: &[T], k: usize, ef: usize) -> Vec<CandidatePair>;
}

/// The trait for index that can be built from a `VecSet`.
pub trait IndexFromVecSet<T: Scalar>: IndexIter<T> {
    /// The `from_vec_set` configuration of the index.
    type Config;

    /// Create an index from a vector set.
    fn from_vec_set(
        vec_set: VecSet<T>,
        dist: DistanceAlgorithm,
        config: Self::Config,
        rng: &mut impl Rng,
    ) -> Self;
}

pub trait IndexSerde: Serialize + for<'de> Deserialize<'de> {
    fn init_after_load(&mut self) {}
    /// Save the index to the file.
    ///
    /// Some Index may have different Implementations.
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }
    /// Load the index from the file.
    ///
    /// Some Index may have different Implementations.
    fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut index: Self = bincode::deserialize_from(reader)?;
        index.init_after_load();
        Ok(index)
    }
}

pub trait IndexSerdeExternalVecSet<T: Scalar>: IndexSerde {
    /// Save the index to the file without original VecSet.
    fn save_without_vec_set(self, path: impl AsRef<Path>) -> Result<Self>;
    /// Load the index saved by `save_without_vec_set`.
    fn load_with_external_vec_set(path: impl AsRef<Path>, vec_set: VecSet<T>) -> Result<Self>;
}

pub trait IndexPQ<T: Scalar>: IndexKNN<T> {
    /// Get the precise k-nearest neighbors with PQ and a search radius `ef`.
    fn knn_pq(&self, query: &[T], k: usize, ef: usize, pq_table: &PQTable<T>)
        -> Vec<CandidatePair>;
}
