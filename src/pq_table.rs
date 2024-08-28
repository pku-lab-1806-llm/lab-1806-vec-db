use serde::{Deserialize, Serialize};

use crate::{binary_scalar::BinaryScalar, config::DistanceAlgorithm, k_means::KMeans};

/// The configuration for the Product Quantization (PQ) table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// The number of bits for each quantized group.
    ///
    /// Should be 4 or 8. Default is 4.
    n_bits: usize,
    /// The number of groups.
    ///
    /// Should satisfy `dim % m == 0`. Default is `dim / 4`.
    m: usize,
    /// The distance algorithm to use.
    ///
    /// Currently only `L2Sqr` is supported.
    dist: DistanceAlgorithm,
    /// The number of iterations for the k-means algorithm.
    k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    k_means_tol: f32,
}

#[derive(Debug)]
pub struct PQTable<T: BinaryScalar> {
    /// The configuration for the PQ table.
    pub config: PQConfig,
    /// The number of dimensions for each vector.
    ///
    /// `k = 2**n_bits` is the number of centroids for each group.
    pub k: usize,
    /// The k-means centroids for each group.
    pub group_k_means: Vec<KMeans<T>>,
    /// The lookup table for each group (flatten). Size `(m * k * k,)`.
    ///
    /// `v[i, c0, c1] = lookup_table[i * k * k + c0 * k + c1]`.
    pub lookup_table: Vec<f32>,
}
