use std::rc::Rc;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{
        k_means::{KMeans, KMeansConfig},
        DistanceAdapter, DistanceAlgorithm, SliceDistance,
    },
    scalar::Scalar,
    vec_set::VecSet,
};
use DistanceAlgorithm::*;

/// The configuration for the Product Quantization (PQ) table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// The number of bits for each quantized group.
    ///
    /// Should be 4 or 8. Usually 4.
    pub n_bits: usize,
    /// The number of groups.
    ///
    /// Should satisfy `dim % m == 0`. Usually `dim / 4`.
    pub m: usize,
    /// The distance algorithm to use.
    pub dist: DistanceAlgorithm,
    /// The number of iterations for the k-means algorithm.
    pub k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    pub k_means_tol: f32,
}

pub fn split_indices(n_bits: usize, m: usize, v: &[u8]) -> Vec<usize> {
    match n_bits {
        4 => v
            .iter()
            .flat_map(|&x| [(x & 0xf) as usize, (x >> 4) as usize])
            .take(m)
            .collect(),
        8 => v.iter().copied().map(Into::into).collect(),
        _ => panic!("n_bits must be 4 or 8 in PQTable."),
    }
}

/// Flattened lookup table for the vector to be queried.
/// Using the PQ in ADC (Asymmetric Distance Computation) algorithm.
#[derive(Debug, Clone)]
pub struct PQLookupTable {
    /// Configuration for the PQ table.
    /// *Cached for convenience.*
    config: Rc<PQConfig>,
    /// The cached dot product of each centroid with the vector.
    /// *Cached for convenience.*
    dot_product_cache: Rc<Vec<f32>>,
    /// Size `(m * k,)`. For group `i` and centroid `c`, the cached value is at `i * k + c`.
    ///
    /// - For L2Sqr and L2 distance, cache the L2Sqr distance between the centroid and the vector.
    /// - For Cosine distance, cache the dot product of the centroid and the vector.
    lookup: Vec<f32>,
    /// Cached norm of the vector itself. Only used for the Cosine distance.
    /// Otherwise, it is `0.0`.
    norm: f32,
}

/// The Product Quantization (PQ) table.
///
/// Can be used to encode vectors, and compute the distance between quantized vectors.
///
/// The encoded vector is stored as `Vec<u8>` with `ceil(m * n_bits / 8)` bytes.
#[derive(Debug, Clone)]
pub struct PQTable<T> {
    /// The configuration for the PQ table.
    pub config: Rc<PQConfig>,
    /// The dimension of the source vectors.
    pub dim: usize,
    /// `k = 2**n_bits` is the number of centroids for each group.
    /// *Cached for convenience.*
    pub k: usize,
    /// The k-means centroids for each group.
    pub group_k_means: Vec<KMeans<T>>,
    /// The dot product of each centroid with itself (flattened).
    /// Size `(m * k,)`, only used for the Cosine distance.
    /// Otherwise, it is empty.
    pub dot_product_cache: Rc<Vec<f32>>,
}

impl<T: Scalar> PQTable<T> {
    /// Create a new PQ table from the given vector set.
    pub fn from_vec_set(
        vec_set: &VecSet<T>,
        config: Rc<PQConfig>,
        rng: &mut impl Rng,
    ) -> PQTable<T> {
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
        let dim = vec_set.dim();
        let d = dim / m;
        let mut group_k_means = Vec::with_capacity(m);
        let mut dot_product_cache = if config.dist == Cosine {
            Vec::with_capacity(m * k)
        } else {
            Vec::new()
        };
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
            let i_config = Rc::new(k_means_config.clone());
            let k_means = KMeans::from_vec_set(vec_set, i_config, rng);
            if config.dist == Cosine {
                for c in k_means.centroids.iter() {
                    dot_product_cache.push(c.dot_product(c));
                }
            }
            group_k_means.push(k_means);
        }
        Self {
            config,
            dim,
            k,
            group_k_means,
            dot_product_cache: Rc::new(dot_product_cache),
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
        split_indices(self.config.n_bits, self.config.m, v)
    }

    /// Decode the given encoded vector.
    ///
    /// *Usually used for debugging.*
    /// See `distance()` for the actual distance computation.
    pub fn decode(&self, v: &[u8]) -> Vec<T> {
        let m = self.config.m;
        let group_dim = self.group_k_means[0].centroids.dim();
        let dim = m * group_dim;
        let indices = self.split_indices(v);
        let mut decoded = Vec::with_capacity(dim);
        for i in 0..m {
            let group = &self.group_k_means[i];
            let index = indices[i];
            let centroid = &group.centroids[index];
            decoded.extend_from_slice(centroid);
        }
        decoded
    }

    /// Create flattened lookup table for the vector to be queried.
    /// Using the PQ in ADC (Asymmetric Distance Computation) algorithm.
    ///
    /// Size `(m * k,)`. For group `i` and centroid `c`, the cached value is at `i * k + c`.
    ///
    /// - For L2Sqr and L2 distance, cache the L2Sqr distance between the centroid and the vector.
    /// - For Cosine distance, cache the dot product of the centroid and the vector.
    pub fn create_lookup(&self, v: &[T]) -> PQLookupTable {
        assert_eq!(v.len(), self.dim);
        let m = self.config.m;
        let k = self.k;
        let d = self.dim / m;
        let mut lookup = Vec::with_capacity(m * k);
        for i in 0..m {
            let k_means = &self.group_k_means[i];
            let centroids = &k_means.centroids;
            let selected = d * i..d * (i + 1);
            let vs = &v[selected];
            match self.config.dist {
                L2Sqr | L2 => centroids
                    .iter()
                    .map(|c| c.l2_sqr_distance(vs))
                    .for_each(|d| lookup.push(d)),
                Cosine => centroids
                    .iter()
                    .map(|c| c.dot_product(vs))
                    .for_each(|d| lookup.push(d)),
            };
        }
        let norm = match self.config.dist {
            L2Sqr | L2 => 0.0,
            Cosine => v.dot_product(v).sqrt(),
        };
        PQLookupTable {
            config: self.config.clone(),
            dot_product_cache: self.dot_product_cache.clone(),
            lookup,
            norm,
        }
    }
}
impl DistanceAdapter<[u8], PQLookupTable> for DistanceAlgorithm {
    /// *** This will ignore the DistanceAlgorithm of `self`
    /// and use the DistanceAlgorithm of `lookup_table`. ***
    fn distance(&self, encoded: &[u8], lookup_table: &PQLookupTable) -> f32 {
        let n_bits = lookup_table.config.n_bits;
        let k = 1 << n_bits;
        let m = lookup_table.config.m;

        let lookup = &lookup_table.lookup;

        let indices = split_indices(n_bits, m, encoded);
        let d: f32 = indices
            .iter()
            .enumerate()
            .map(|(i, &c)| lookup[i * k + c])
            .sum();

        match lookup_table.config.dist {
            L2Sqr => d,
            L2 => d.sqrt(),
            Cosine => {
                let dot_product = d;
                let norm0 = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| lookup_table.dot_product_cache[i * k + c])
                    .sum::<f32>()
                    .sqrt();
                let norm1 = lookup_table.norm;

                1.0 - dot_product / (norm0 * norm1)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use anyhow::Result;
    use rand::SeedableRng;

    use crate::{config::DBConfig, distance::DistanceAdapter, scalar::Scalar};

    use super::*;

    fn pq_table_precise_test_base(dist: DistanceAlgorithm) {
        // Test the PQ table with num_vec < k, so that the centroids are the same as the vectors.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dim = 8;
        let n_bits = 4;
        let m = 2;
        let num_vec = 5;

        let mut src_set = VecSet::<f32>::zeros(dim, num_vec);
        for i in 0..num_vec {
            let v = src_set.get_mut(i);
            for v in v.iter_mut() {
                *v = rng.gen_range(-1.0..1.0);
            }
        }
        let pq_config = Rc::new(PQConfig {
            n_bits,
            m,
            dist,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        });
        let pq_table = PQTable::from_vec_set(&src_set, pq_config, &mut rng);

        let encoded_set = pq_table.encode_batch(&src_set);
        for i in 0..num_vec {
            let src = &src_set[i];
            let decoded = pq_table.decode(&encoded_set[i]);
            println!("{}: {:?}", i, &src_set[i]);
            assert_eq!(src, &decoded);
        }
        for i in 0..num_vec {
            let lookup = pq_table.create_lookup(&src_set[i]);
            for j in 0..num_vec {
                let src0 = &src_set[i];
                let src1 = &src_set[j];
                let src_dist = dist.d(src0, src1);

                let e1 = &encoded_set[j];
                let e_dist = dist.d(e1, &lookup);

                println!("{}<->{}: src={:.6} encoded={:.6}", i, j, src_dist, e_dist);
                assert!((src_dist - e_dist).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn pq_table_precise_test() {
        pq_table_precise_test_base(L2Sqr);
        pq_table_precise_test_base(L2);
        pq_table_precise_test_base(Cosine);
    }

    fn pq_table_test_base<T: Scalar>(vec_set: &VecSet<T>, dist: DistanceAlgorithm) -> Result<()> {
        let dim = vec_set.dim();
        let pq_config = Rc::new(PQConfig {
            n_bits: 4,
            m: dim / 4,
            dist,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        });
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pq_table = PQTable::from_vec_set(&vec_set, pq_config, &mut rng);
        let encoded_set = pq_table.encode_batch(&vec_set);

        println!("Distance Algorithm: {:?}", dist);
        let test_count = 20;
        let mut errors = Vec::new();
        for _ in 0..test_count {
            let i0 = rng.gen_range(0..vec_set.len());
            let i1 = rng.gen_range(0..vec_set.len());
            let v0 = &vec_set[i0];
            let v1 = &vec_set[i1];
            let e0 = &encoded_set[i0];
            let lookup = pq_table.create_lookup(v1);
            let distance = dist.d(e0, &lookup);
            let expected = dist.d(v0, v1);
            let error = (distance - expected).abs() / expected.max(1.0);
            println!(
                "Distance: {} / Expected: {} / Error: {}",
                distance, expected, error
            );
            errors.push(error);
        }
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i90 = (errors.len() as f32 * 0.9).ceil() as usize - 1;
        let p90 = errors[i90];
        println!("90% Error: {}", p90);
        assert!(p90 < 0.2, "90% Error is too large.");
        Ok(())
    }

    #[test]
    fn pq_table_test() -> Result<()> {
        // The `dim` and `limit` has been limited for debug mode performance.
        // Try `cargo r -r --example pq_table_test` for release mode testing,
        // which provides more accurate results.

        let file_path = "config/example/db_config.toml";
        let mut config = DBConfig::load_from_toml_file(file_path)?;

        config.vec_data.limit = Some(64);

        let raw_vec_set = VecSet::<f32>::load_with(&config.vec_data)?;

        let clipped_dim = raw_vec_set.dim().min(12);

        let mut vec_set = VecSet::zeros(clipped_dim, raw_vec_set.len());
        for i in 0..raw_vec_set.len() {
            let src = &raw_vec_set[i];
            let dst = vec_set.get_mut(i);
            dst.copy_from_slice(&src[..clipped_dim]);
        }

        pq_table_test_base(&vec_set, L2Sqr)?;
        pq_table_test_base(&vec_set, L2)?;
        pq_table_test_base(&vec_set, Cosine)?;
        Ok(())
    }
}