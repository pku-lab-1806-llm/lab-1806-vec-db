use pyo3::prelude::*;

#[pymodule]
pub mod lab_1806_vec_db {
    use crate::database::{MetadataIndex, VecDBManager};
    use crate::prelude::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    /// Get the distance algorithm from a string.
    ///
    /// Not exposed to Python.
    fn distance_algorithm_from_str(dist: &str) -> PyResult<DistanceAlgorithm> {
        match dist {
            "l2sqr" => Ok(DistanceAlgorithm::L2Sqr),
            "l2" => Ok(DistanceAlgorithm::L2),
            "cosine" => Ok(DistanceAlgorithm::Cosine),
            _ => Err(PyValueError::new_err("Invalid distance function")),
        }
    }
    fn distance_algorithm_to_str(dist: DistanceAlgorithm) -> &'static str {
        match dist {
            DistanceAlgorithm::L2Sqr => "l2sqr",
            DistanceAlgorithm::L2 => "l2",
            DistanceAlgorithm::Cosine => "cosine",
        }
    }

    /// Calculate the distance between two vectors.
    ///
    /// `dist` can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)
    ///
    /// Raises:
    ///     ValueError: If the distance function is invalid.
    #[pyfunction]
    #[pyo3(signature = (a, b, dist="cosine"))]
    pub fn calc_dist(a: Vec<f32>, b: Vec<f32>, dist: &str) -> PyResult<f32> {
        let dist = distance_algorithm_from_str(dist)?;
        Ok(dist.d(a.as_slice(), b.as_slice()))
    }

    /// Bare Vector Database Table.
    ///
    /// Prefer using VecDB to manage multiple tables.
    #[pyclass]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BareVecTable {
        pub(crate) inner: MetadataIndex,
    }

    #[pymethods]
    impl BareVecTable {
        #[new]
        #[pyo3(signature = (dim, dist="cosine"))]
        /// Create a new HNSW index.
        ///
        /// Args:
        ///    dim (int): Dimension of the vectors.
        ///    dist (str): Distance function. Can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)
        ///
        /// Raises:
        ///     ValueError: If the distance function is invalid.
        pub fn new(dim: usize, dist: &str) -> PyResult<Self> {
            let dist = distance_algorithm_from_str(dist)?;
            let inner = MetadataIndex::new(dim, dist);
            Ok(Self { inner })
        }

        /// Get the dimension of the vectors.
        pub fn dim(&self) -> usize {
            self.inner.dim()
        }

        /// Get the distance algorithm name.
        pub fn dist(&self) -> String {
            distance_algorithm_to_str(self.inner.dist()).to_string()
        }

        /// Get the number of vectors in the index.
        pub fn __len__(&self) -> usize {
            self.inner.len()
        }

        /// Load an existing HNSW index from disk.
        ///
        /// Raises:
        ///     RuntimeError: If the file is not found or the index is corrupted.
        #[staticmethod]
        pub fn load(path: &str) -> PyResult<Self> {
            let inner =
                MetadataIndex::load(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }

        /// Save the HNSW index to disk.
        /// The random seed is not saved.
        ///
        /// Raises:
        ///     RuntimeError: If the file cannot be written.
        pub fn save(&self, path: &str) -> PyResult<()> {
            self.inner
                .save(path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        /// Add a vector to the index.
        ///
        /// Returns the ID of the added vector.
        ///
        /// Use `batch_add` for better performance.
        pub fn add(&mut self, vec: Vec<f32>, metadata: BTreeMap<String, String>) -> usize {
            self.inner.add(vec, metadata)
        }

        /// Add multiple vectors to the index.
        /// Returns the id list of the added vectors.
        ///
        /// If the vec_list is too large, it will be split into smaller chunks.
        /// If the vec_list is too small or the index is too small, it will be the same as calling `add` multiple times.
        pub fn batch_add(
            &mut self,
            vec_list: Vec<Vec<f32>>,
            metadata_list: Vec<BTreeMap<String, String>>,
        ) -> Vec<usize> {
            self.inner.batch_add(vec_list, metadata_list)
        }

        /// Search for the nearest neighbors of a vector.
        ///
        /// Returns a list of (id, distance) pairs.
        #[pyo3(signature = (query, k, ef=None, upper_bound=None))]
        pub fn search(
            &self,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            upper_bound: Option<f32>,
        ) -> Vec<(BTreeMap<String, String>, f32)> {
            self.inner.search(&query, k, ef, upper_bound)
        }
    }

    #[pyclass]
    pub struct VecDB {
        pub(crate) inner: VecDBManager,
    }
    #[pymethods]
    impl VecDB {
        /// Create a new VecDB, it will create a new directory if it does not exist.
        ///
        /// Automatically save the database to disk when dropped. Cache the tables when accessing their contents.
        #[new]
        pub fn new(dir: String) -> PyResult<Self> {
            let inner =
                VecDBManager::new(&dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }

        /// Create a new table if it does not exist.
        ///
        /// Raises:
        ///     RuntimeError: If the file is corrupted.
        pub fn create_table_if_not_exists(
            &self,
            name: String,
            dim: usize,
            dist: &str,
        ) -> PyResult<bool> {
            let dist = distance_algorithm_from_str(dist)?;
            self.inner
                .create_table_if_not_exists(&name, dim, dist)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
        /// Get table info.
        ///
        /// Returns:
        ///    (dim, len, dist)
        ///
        /// Raises:
        ///     RuntimeError: If the table is not found.
        pub fn get_table_info(&self, key: String) -> PyResult<(usize, usize, String)> {
            self.inner
                .get_table_info(&key)
                .map(|info| {
                    (
                        info.dim,
                        info.len,
                        distance_algorithm_to_str(info.dist).to_string(),
                    )
                })
                .ok_or_else(|| PyRuntimeError::new_err("Table not found"))
        }

        /// Get all table names.
        pub fn get_all_keys(&self) -> Vec<String> {
            self.inner.get_all_keys()
        }

        /// Add a vector to the table.
        pub fn add(&self, key: String, vec: Vec<f32>, metadata: BTreeMap<String, String>) -> usize {
            self.inner.add(&key, vec, metadata)
        }

        /// Add multiple vectors to the table.
        pub fn batch_add(
            &self,
            key: String,
            vec_list: Vec<Vec<f32>>,
            metadata_list: Vec<BTreeMap<String, String>>,
        ) -> Vec<usize> {
            self.inner.batch_add(&key, vec_list, metadata_list)
        }

        /// Search for the nearest neighbors of a vector.
        /// Returns a list of (metadata, distance) pairs.
        #[pyo3(signature = (key, query, k, ef=None, upper_bound=None))]
        pub fn search(
            &self,
            key: String,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            upper_bound: Option<f32>,
        ) -> PyResult<Vec<(BTreeMap<String, String>, f32)>> {
            self.inner
                .search(&key, &query, k, ef, upper_bound)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        /// Search for the nearest neighbors of a vector in multiple tables.
        #[pyo3(signature = (key_list, query, k, ef=None, upper_bound=None))]
        pub fn join_search(
            &self,
            key_list: BTreeSet<String>,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            upper_bound: Option<f32>,
        ) -> PyResult<Vec<(String, BTreeMap<String, String>, f32)>> {
            self.inner
                .join_search(&key_list, &query, k, ef, upper_bound)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
    }
}
