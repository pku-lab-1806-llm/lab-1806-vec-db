use pyo3::prelude::*;

#[pymodule]
pub mod lab_1806_vec_db {
    use crate::database::VecDBManager;
    use crate::prelude::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use std::collections::BTreeMap;

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

    /// Vector Database. Ensures:
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
        #[pyo3(signature = (key, dim, dist="cosine"))]
        pub fn create_table_if_not_exists(
            &self,
            py: Python,
            key: &str,
            dim: usize,
            dist: &str,
        ) -> PyResult<bool> {
            py.allow_threads(|| {
                let dist = distance_algorithm_from_str(dist)?;
                self.inner
                    .create_table_if_not_exists(key, dim, dist)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Get all table names.
        pub fn get_all_keys(&self) -> Vec<String> {
            self.inner.get_all_keys()
        }
        /// Check if a table exists.
        pub fn contains_key(&self, key: &str) -> bool {
            self.inner.contains_key(key)
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

        /// Returns a list of table keys that are cached.
        pub fn get_cached_tables(&self) -> Vec<String> {
            self.inner.get_cached_tables()
        }
        /// Check if a table is cached.
        pub fn contains_cached(&self, key: &str) -> bool {
            self.inner.contains_cached(key)
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

        /// Get the number of vectors in the table.
        pub fn get_len(&self, py: Python, key: &str) -> PyResult<usize> {
            py.allow_threads(|| {
                self.inner
                    .get_len(key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }
        /// Get the dimension of the vectors in the table.
        pub fn get_dim(&self, py: Python, key: &str) -> PyResult<usize> {
            py.allow_threads(|| {
                self.inner
                    .get_dim(key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }
        /// Get the distance algorithm used by the table.
        pub fn get_dist(&self, py: Python, key: &str) -> PyResult<String> {
            py.allow_threads(|| {
                self.inner
                    .get_dist(key)
                    .map(|dist| distance_algorithm_to_str(dist).to_string())
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

        /// Delete vectors with metadata that match the pattern.
        /// Returns the number of vectors deleted.
        pub fn delete(
            &self,
            py: Python,
            key: &str,
            pattern: BTreeMap<String, String>,
        ) -> PyResult<usize> {
            py.allow_threads(|| {
                self.inner
                    .delete(key, &pattern)
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

        /// Extract data from the table.
        pub fn extract_data(
            &self,
            py: Python,
            key: &str,
        ) -> PyResult<Vec<(Vec<f32>, BTreeMap<String, String>)>> {
            py.allow_threads(|| {
                self.inner
                    .extract_data(key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Build HNSW index for the table.
        #[pyo3(signature = (key, ef_construction=None))]
        pub fn build_hnsw_index(
            &self,
            py: Python,
            key: &str,
            ef_construction: Option<usize>,
        ) -> PyResult<()> {
            py.allow_threads(|| {
                self.inner
                    .build_hnsw_index(key, ef_construction)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Clear HNSW index for the table.
        pub fn clear_hnsw_index(&self, py: Python, key: &str) -> PyResult<()> {
            py.allow_threads(|| {
                self.inner
                    .clear_hnsw_index(key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Check if the table has HNSW index.
        pub fn has_hnsw_index(&self, py: Python, key: &str) -> PyResult<bool> {
            py.allow_threads(|| {
                self.inner
                    .has_hnsw_index(key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Build PQ table for the table.
        #[pyo3(signature = (key, train_proportion=None, n_bits=None, m=None))]
        pub fn build_pq_table(
            &self,
            py: Python,
            key: &str,
            train_proportion: Option<f32>,
            n_bits: Option<usize>,
            m: Option<usize>,
        ) -> PyResult<()> {
            py.allow_threads(|| {
                self.inner
                    .build_pq_table(key, train_proportion, n_bits, m)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Clear PQ table for the table.
        pub fn clear_pq_table(&self, py: Python, key: &str) -> PyResult<()> {
            py.allow_threads(|| {
                self.inner
                    .clear_pq_table(key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }

        /// Check if the table has PQ table.
        pub fn has_pq_table(&self, py: Python, key: &str) -> PyResult<bool> {
            py.allow_threads(|| {
                self.inner
                    .has_pq_table(key)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }
    }
}
