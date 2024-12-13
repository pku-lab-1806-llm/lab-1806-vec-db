use pyo3::prelude::*;

#[pymodule]
pub mod lab_1806_vec_db {
    use crate::database::{MetadataVecTable, VecDBManager};
    use crate::prelude::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    /// Get the distance algorithm from a string.
    ///
    /// Not exposed to Python.
    fn distance_algorithm_from_str(dist: &str) -> PyResult<DistanceAlgorithm> {
        use DistanceAlgorithm::*;
        match dist {
            "l2sqr" => Ok(L2Sqr),
            "cosine" => Ok(Cosine),
            _ => Err(PyValueError::new_err("Invalid distance function")),
        }
    }
    fn distance_algorithm_to_str(dist: DistanceAlgorithm) -> &'static str {
        use DistanceAlgorithm::*;
        match dist {
            L2Sqr => "l2sqr",
            Cosine => "cosine",
            #[allow(unreachable_patterns)]
            _ => panic!("Invalid distance function"),
        }
    }

    /// Calculate the distance between two vectors.
    ///
    /// `dist` can be "l2sqr" or "cosine" (default: "cosine", for RAG).
    ///
    ///
    /// - l2sqr: squared Euclidean distance
    /// - cosine: cosine distance (1 - cosine_similarity) [0.0, 2.0]
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
        pub(crate) inner: MetadataVecTable,
    }

    #[pymethods]
    impl BareVecTable {
        #[new]
        #[pyo3(signature = (dim, dist="cosine", ef_c=None))]
        /// Create a new Table. (Using HNSW internally)
        ///
        /// Raises:
        ///     ValueError: If the distance function is invalid.
        pub fn new(dim: usize, dist: &str, ef_c: Option<usize>) -> PyResult<Self> {
            let dist = distance_algorithm_from_str(dist)?;
            let inner = MetadataVecTable::new(dim, dist, ef_c);
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

        /// Load an existing index from disk.
        ///
        /// Raises:
        ///     RuntimeError: If the file is not found or the index is corrupted.
        #[staticmethod]
        pub fn load(path: &str) -> PyResult<Self> {
            let inner =
                MetadataVecTable::load(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }

        /// Save the index to disk.
        ///
        /// Raises:
        ///     RuntimeError: If the file cannot be written.
        pub fn save(&self, path: &str) -> PyResult<()> {
            self.inner
                .save(path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        /// Add a vector to the index.
        /// Use `batch_add` for better performance.
        pub fn add(&mut self, vec: Vec<f32>, metadata: BTreeMap<String, String>) {
            self.inner.add(vec, metadata);
        }

        /// Add multiple vectors to the index.
        /// Call it with a batch size around 64 to avoid long lock time.
        pub fn batch_add(
            &mut self,
            vec_list: Vec<Vec<f32>>,
            metadata_list: Vec<BTreeMap<String, String>>,
        ) {
            self.inner.batch_add(vec_list, metadata_list);
        }

        /// Get the specified row by id.
        pub fn get_row_by_id(&self, id: usize) -> PyResult<(Vec<f32>, BTreeMap<String, String>)> {
            self.inner
                .get_row_by_id(id)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        /// Search for the nearest neighbors of a vector.
        ///
        /// Returns a list of (metadata, distance) pairs.
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

    /// Vector Database.
    ///
    /// Prefer using this to manage multiple tables.
    ///
    ///
    /// Ensures:
    /// - Auto-save. The database will be saved to disk when necessary.
    /// - Parallelism. `allow_threads` is used to allow multi-threading.
    /// - Thread-safe. Read and write operations are atomic.
    /// - Unique. Only one manager for each database.
    #[pyclass]
    pub struct VecDB {
        pub(crate) inner: VecDBManager,
    }
    #[pymethods]
    impl VecDB {
        /// Create a new VecDB, it will create a new directory if it does not exist.
        #[new]
        pub fn new(dir: String) -> PyResult<Self> {
            let inner =
                VecDBManager::new(&dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }

        /// Create a new table if it does not exist.
        ///
        /// Raises:
        ///     ValueError: If the distance function is invalid.
        #[pyo3(signature = (key, dim, dist="cosine", ef_c=None))]
        pub fn create_table_if_not_exists(
            &self,
            py: Python,
            key: &str,
            dim: usize,
            dist: &str,
            ef_c: Option<usize>,
        ) -> PyResult<bool> {
            py.allow_threads(|| {
                let dist = distance_algorithm_from_str(&dist)?;
                self.inner
                    .create_table_if_not_exists(key, dim, dist, ef_c)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }
        /// Get table info.
        ///
        /// Returns:
        ///    (dim, len, dist)
        pub fn get_table_info(&self, py: Python, key: String) -> PyResult<(usize, usize, String)> {
            py.allow_threads(|| {
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
            })
        }

        /// Delete a table and wait for all operations to finish.
        /// Returns false if the table does not exist.
        pub fn delete_table(&self, py: Python, key: String) -> PyResult<bool> {
            py.allow_threads(|| {
                self.inner
                    .delete_table(&key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Get all table names.
        pub fn get_all_keys(&self, py: Python) -> Vec<String> {
            py.allow_threads(|| self.inner.get_all_keys())
        }

        /// Returns a list of table keys that are cached.
        pub fn get_cached_tables(&self, py: Python) -> Vec<String> {
            py.allow_threads(|| self.inner.get_cached_tables())
        }
        /// Remove a table from the cache.
        /// Does nothing if the table is not cached.
        pub fn remove_cached_table(&self, py: Python, key: &str) -> PyResult<()> {
            py.allow_threads(|| {
                self.inner
                    .remove_cached_table(key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Add a vector to the table.
        /// Use `batch_add` for better performance.
        pub fn add(
            &self,
            py: Python,
            key: &str,
            vec: Vec<f32>,
            metadata: BTreeMap<String, String>,
        ) -> PyResult<()> {
            py.allow_threads(|| {
                self.inner
                    .add(key, vec, metadata)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Add multiple vectors to the table.
        pub fn batch_add(
            &self,
            py: Python,
            key: &str,
            vec_list: Vec<Vec<f32>>,
            metadata_list: Vec<BTreeMap<String, String>>,
        ) -> PyResult<()> {
            py.allow_threads(|| {
                self.inner
                    .batch_add(key, vec_list, metadata_list)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Get the specified row by id.
        pub fn get_row_by_id(
            &self,
            py: Python,
            key: &str,
            id: usize,
        ) -> PyResult<(Vec<f32>, BTreeMap<String, String>)> {
            py.allow_threads(|| {
                self.inner
                    .get_row_by_id(key, id)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Search for the nearest neighbors of a vector.
        /// Returns a list of (metadata, distance) pairs.
        #[pyo3(signature = (key, query, k, ef=None, upper_bound=None))]
        pub fn search(
            &self,
            py: Python,
            key: &str,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            upper_bound: Option<f32>,
        ) -> PyResult<Vec<(BTreeMap<String, String>, f32)>> {
            py.allow_threads(|| {
                self.inner
                    .search(key, &query, k, ef, upper_bound)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Search for the nearest neighbors of a vector in multiple tables.
        /// Returns a list of (table_name, metadata, distance) pairs.
        #[pyo3(signature = (key_list, query, k, ef=None, upper_bound=None))]
        pub fn join_search(
            &self,
            py: Python,
            key_list: BTreeSet<String>,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            upper_bound: Option<f32>,
        ) -> PyResult<Vec<(String, BTreeMap<String, String>, f32)>> {
            py.allow_threads(|| {
                self.inner
                    .join_search(&key_list, &query, k, ef, upper_bound)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }
    }
}
