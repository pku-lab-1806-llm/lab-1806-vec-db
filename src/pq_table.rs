use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    binary_scalar::BinaryScalar,
    config::DistanceAlgorithm,
    distance::Distance,
    k_means::{KMeans, KMeansConfig},
    vec_set::VecSet,
};

/// The configuration for the Product Quantization (PQ) table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// The number of bits for each quantized group.
    ///
    /// Should be 4 or 8. Usually 4.
    n_bits: usize,
    /// The number of groups.
    ///
    /// Should satisfy `dim % m == 0`. Usually `dim / 4`.
    m: usize,
    /// The distance algorithm to use.
    ///
    /// Currently only `L2Sqr` or `L2` is supported.
    dist: DistanceAlgorithm,
    /// The number of iterations for the k-means algorithm.
    k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    k_means_tol: f32,
}

/// The Product Quantization (PQ) table.
///
/// Can be used to encode vectors, and compute the distance between quantized vectors.
///
/// The encoded vector is stored as `Vec<u8>` with `ceil(m * n_bits / 8)` bytes.
#[derive(Debug)]
pub struct PQTable<T: BinaryScalar> {
    /// The configuration for the PQ table.
    pub config: PQConfig,
    /// `k = 2**n_bits` is the number of centroids for each group.
    /// *Cached for convenience.*
    pub k: usize,
    /// The k-means centroids for each group.
    pub group_k_means: Vec<KMeans<T>>,
    /// The lookup table for each group (flatten). Size `(m * k * k,)`.
    ///
    /// `v[i, c0, c1] = lookup_table[i * k * k + c0 * k + c1]`.
    pub lookup_table: Vec<f32>,
}

impl<T: BinaryScalar> PQTable<T>
where
    [T]: Distance,
{
    /// Create a new PQ table from the given vector set.
    pub fn from_vec_set(vec_set: &VecSet<T>, config: &PQConfig, rng: &mut impl Rng) -> PQTable<T> {
        assert!(
            config.n_bits == 4 || config.n_bits == 8,
            "n_bits must be 4 or 8 in PQTable."
        );
        let m = config.m;
        assert!(
            vec_set.dim() % m == 0,
            "dim must be a multiple of m in PQTable."
        );
        let k = 1 << config.n_bits;
        let d = vec_set.dim() / m;
        let mut group_k_means = Vec::with_capacity(m);
        let mut lookup_table = Vec::with_capacity(m * k * k);
        let mut k_means_config = KMeansConfig {
            k,
            max_iter: config.k_means_max_iter,
            tol: config.k_means_tol,
            dist: config.dist,
            selected: None,
        };
        for i in 0..m {
            let selected = d * i..d * (i + 1);
            k_means_config.selected = Some(selected);
            let k_means = KMeans::from_vec_set(vec_set, &k_means_config, rng);
            let cs: &VecSet<T> = &k_means.centroids;
            use DistanceAlgorithm::*;
            match config.dist {
                L2Sqr | L2 => {
                    for c0 in 0..k {
                        for c1 in 0..k {
                            lookup_table.push(L2Sqr.d(&cs[c0], &cs[c1]));
                        }
                    }
                }
                _ => panic!("Only L2Sqr or L2 distance is supported in PQTable."),
            }
            for c0 in 0..k {
                for c1 in 0..k {
                    lookup_table.push(config.dist.d(&cs[c0], &cs[c1]));
                }
            }
            group_k_means.push(k_means);
        }
        Self {
            config: config.clone(),
            k,
            group_k_means,
            lookup_table,
        }
    }
    /// Encode the given vector.
    pub fn encode(&self, v: &[T]) -> Vec<u8> {
        let m = self.config.m;
        let n_bits = self.config.n_bits;

        match n_bits {
            4 => {
                let mut encoded = vec![0_u8; m.div_ceil(2)];
                for i in 0..m / 2 {
                    let v0 = self.group_k_means[2 * i].find_nearest(v);
                    let v1 = self.group_k_means[2 * i + 1].find_nearest(v);
                    encoded[i] = (v0 | (v1 << 4)) as u8;
                }
                if m % 2 == 1 {
                    encoded[m / 2] = self.group_k_means[m - 1].find_nearest(v) as u8;
                }
                encoded
            }
            8 => self
                .group_k_means
                .iter()
                .map(|g| g.find_nearest(v) as u8)
                .collect(),
            _ => panic!("n_bits must be 4 or 8 in PQTable."),
        }
    }

    /// Get the dimension of the encoded vector.
    pub fn encoded_dim(&self) -> usize {
        match self.config.n_bits {
            4 => self.config.m.div_ceil(2),
            8 => self.config.m,
            _ => panic!("n_bits must be 4 or 8 in PQTable."),
        }
    }

    /// Encode the given vector set.
    pub fn encode_batch(&self, vec_set: &VecSet<T>) -> VecSet<u8> {
        let dim = self.encoded_dim();
        let mut target_set = VecSet::zeros(dim, vec_set.len());
        for (i, v) in vec_set.iter().enumerate() {
            target_set.put(i, &self.encode(v));
        }
        target_set
    }

    /// Get indices from a encoded vector.
    pub fn split_indices(&self, v: &[u8]) -> Vec<usize> {
        match self.config.n_bits {
            4 => v
                .iter()
                .flat_map(|&x| [(x & 0xf) as usize, (x >> 4) as usize])
                .take(self.config.m)
                .collect(),
            8 => v.iter().copied().map(Into::into).collect(),
            _ => panic!("n_bits must be 4 or 8 in PQTable."),
        }
    }

    /// Compute the distance between two encoded vectors.
    pub fn distance(&self, v0: &[u8], v1: &[u8]) -> f32 {
        let m = self.config.m;
        let dist = self.config.dist;
        let v0 = self.split_indices(v0);
        let v1 = self.split_indices(v1);
        use DistanceAlgorithm::*;
        let mut d = 0.0;
        for i in 0..m {
            d += self.lookup_table[i * self.k * self.k + v0[i] * self.k + v1[i]];
        }
        match dist {
            L2Sqr => d,
            L2 => d.sqrt(),
            _ => panic!("Only L2Sqr or L2 distance is supported in PQTable."),
        }
    }

    /// Alias of `PQTable::distance()`.
    /// Compute the distance between two encoded vectors.
    pub fn d(&self, v0: &[u8], v1: &[u8]) -> f32 {
        self.distance(v0, v1)
    }
}
