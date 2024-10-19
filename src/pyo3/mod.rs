use crate::{
    index_algorithm::{HNSWConfig, HNSWIndex},
    prelude::*,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::{fs::File, io::BufWriter};

#[pymodule]
pub mod lab_1806_vec_db {

    use ordered_float::OrderedFloat;

    use super::*;

    fn distance_algorithm_from_str(dist: &str) -> PyResult<DistanceAlgorithm> {
        match dist {
            "l2sqr" => Ok(DistanceAlgorithm::L2Sqr),
            "l2" => Ok(DistanceAlgorithm::L2),
            "cosine" => Ok(DistanceAlgorithm::Cosine),
            _ => Err(PyValueError::new_err("Invalid distance function")),
        }
    }

    /// Calculate the distance between two vectors.
    ///
    /// `dist` can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)
    #[pyfunction]
    #[pyo3(signature = (a, b, dist="cosine"))]
    pub fn calc_dist(a: Vec<f32>, b: Vec<f32>, dist: &str) -> PyResult<f32> {
        let dist = distance_algorithm_from_str(dist)?;
        Ok(dist.d(a.as_slice(), b.as_slice()))
    }

    /// A vector database for RAG using HNSW index.
    #[pyclass]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RagVecDB {
        pub metadata: Vec<BTreeMap<String, String>>,
        pub inner: HNSWIndex<f32>,
        #[serde(skip, default = "rand::SeedableRng::from_entropy")]
        pub rng: rand::rngs::StdRng,
    }

    #[pymethods]
    impl RagVecDB {
        #[new]
        #[pyo3(signature = (dim, dist="cosine", ef_construction=200, M=16, max_elements=0, seed=None))]
        /// Create a new HNSW index.
        ///
        /// Args:
        ///    dim (int): Dimension of the vectors.
        ///    dist (str): Distance function. Can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)
        ///    ef_construction (int): Number of elements to consider during construction. (default: 200)
        ///    M (int): Number of neighbors to consider during search. (default: 16)
        ///    max_elements (int): The initial capacity of the index. (default: 0, auto-grow)
        ///    seed (int): Random seed for the index. (default: None, random)
        ///
        /// Random seed will never be saved. Never call `add` on a loaded index if you want to have deterministic index construction.
        pub fn new(
            dim: usize,
            dist: &str,
            ef_construction: usize,
            #[allow(non_snake_case)] M: usize,
            max_elements: usize,
            seed: Option<u64>,
        ) -> PyResult<Self> {
            let config = HNSWConfig {
                ef_construction,
                M,
                max_elements,
            };
            let dist = distance_algorithm_from_str(dist)?;
            let rng = match seed {
                Some(seed) => rand::SeedableRng::seed_from_u64(seed),
                None => rand::SeedableRng::from_entropy(),
            };
            Ok(Self {
                metadata: Vec::with_capacity(max_elements),
                inner: HNSWIndex::new(dim, dist, config),
                rng,
            })
        }

        /// Load an existing HNSW index from disk.
        #[staticmethod]
        pub fn load(path: &str) -> PyResult<Self> {
            let file = File::open(path)?;
            let reader = std::io::BufReader::new(file);
            let index = bincode::deserialize_from(reader)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(index)
        }

        /// Save the HNSW index to disk.
        /// The random seed is not saved.
        pub fn save(&self, path: &str) -> PyResult<()> {
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            bincode::serialize_into(writer, self)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        }

        /// Add a vector to the index.
        ///
        /// Returns the ID of the added vector.
        ///
        /// Use `batch_add` for better performance.
        pub fn add(&mut self, vec: Vec<f32>, metadata: BTreeMap<String, String>) -> usize {
            self.metadata.push(metadata);
            self.inner.add(&vec, &mut self.rng)
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
            assert_eq!(vec_list.len(), metadata_list.len());
            self.metadata.extend(metadata_list);
            let vec_list: Vec<_> = vec_list.iter().map(|v| v.as_slice()).collect();
            self.inner.batch_add(&vec_list, &mut self.rng)
        }

        pub fn __len__(&self) -> usize {
            self.inner.len()
        }

        /// Get the vector by id.
        pub fn get_vec(&self, id: usize) -> Vec<f32> {
            self.inner[id].to_vec()
        }

        /// Get the metadata by id.
        pub fn get_metadata(&self, id: usize) -> BTreeMap<String, String> {
            self.metadata[id].clone()
        }

        /// Set the metadata by id.
        pub fn set_metadata(&mut self, id: usize, metadata: BTreeMap<String, String>) {
            self.metadata[id] = metadata;
        }

        /// Search for the nearest neighbors of a vector.
        ///
        /// Returns a list of (id, distance) pairs.
        #[pyo3(signature = (query, k, ef=None, max_distance=None))]
        pub fn search_as_pair(
            &self,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            max_distance: Option<f32>,
        ) -> Vec<(usize, f32)> {
            let results = match ef {
                Some(ef) => self.inner.knn_with_ef(&query, k, ef),
                None => self.inner.knn(&query, k),
            };
            if let Some(max_distance) = max_distance {
                results
                    .into_iter()
                    .filter(|p| p.distance() <= max_distance)
                    .map(|p| (p.index, p.distance()))
                    .collect()
            } else {
                results
                    .into_iter()
                    .map(|p| (p.index, p.distance()))
                    .collect()
            }
        }

        /// Search for the nearest neighbors of a vector.
        ///
        /// Returns a list of metadata.
        #[pyo3(signature = (query, k, ef=None, max_distance=None))]
        pub fn search(
            &self,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            max_distance: Option<f32>,
        ) -> Vec<BTreeMap<String, String>> {
            self.search_as_pair(query, k, ef, max_distance)
                .into_iter()
                .map(|(id, _)| self.metadata[id].clone())
                .collect()
        }
    }

    /// A group of vector databases for automatic searching and merging KNN results.
    #[pyclass]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RagMultiVecDB {
        pub multi_vec_db: Vec<RagVecDB>,
    }

    #[pymethods]
    impl RagMultiVecDB {
        /// Create a new multi-vector database.
        #[new]
        pub fn new(multi_vec_db: Vec<RagVecDB>) -> Self {
            Self { multi_vec_db }
        }

        /// Get a vector by (db_id, vec_id).
        pub fn get_vec(&self, db_id: usize, vec_id: usize) -> Vec<f32> {
            self.multi_vec_db[db_id].get_vec(vec_id)
        }

        /// Get the metadata by (db_id, vec_id).
        pub fn get_metadata(&self, db_id: usize, vec_id: usize) -> BTreeMap<String, String> {
            self.multi_vec_db[db_id].get_metadata(vec_id)
        }

        /// Search for the nearest neighbors of a vector.
        ///
        /// Returns a list of (db_id, vec_id, distance) tuples.
        #[pyo3(signature = (query, k, ef=None, max_distance=None))]
        pub fn search_as_pair(
            &self,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            max_distance: Option<f32>,
        ) -> Vec<(usize, usize, f32)> {
            let (sender, receiver) = std::sync::mpsc::channel();
            let idx_list = (0..self.multi_vec_db.len()).collect::<Vec<_>>();
            std::thread::scope(|s| {
                for idx in idx_list.iter() {
                    let db = &self.multi_vec_db[*idx];
                    let query = query.clone();
                    s.spawn(|| {
                        let results = db.search_as_pair(query, k, ef, max_distance);
                        let results: Vec<(usize, usize, f32)> = results
                            .into_iter()
                            .map(|(id, dist)| (*idx, id, dist))
                            .collect();
                        sender.send(results).unwrap();
                    });
                }
            });
            drop(sender);
            let mut results = Vec::new();
            for r in receiver {
                results.extend(r);
            }
            results.sort_by_key(|(_, _, dist)| OrderedFloat(*dist));
            results.truncate(k);
            results
        }

        /// Search for the nearest neighbors of a vector.
        /// Returns a list of metadata.
        #[pyo3(signature = (query, k, ef=None, max_distance=None))]
        pub fn search(
            &self,
            query: Vec<f32>,
            k: usize,
            ef: Option<usize>,
            max_distance: Option<f32>,
        ) -> Vec<BTreeMap<String, String>> {
            self.search_as_pair(query, k, ef, max_distance)
                .into_iter()
                .map(|(db_id, vec_id, _)| self.get_metadata(db_id, vec_id))
                .collect()
        }
    }
}
