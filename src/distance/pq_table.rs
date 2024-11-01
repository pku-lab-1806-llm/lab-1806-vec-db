use std::path::Path;

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    distance::{
        k_means::{KMeans, KMeansConfig},
        DistanceAdapter, DistanceAlgorithm,
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
    /// The number of vectors to be sampled for the k-means algorithm.
    pub k_means_size: Option<usize>,
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
pub fn pq_encode<T: Scalar>(
    m: usize,
    n_bits: usize,
    group_k_means: &Vec<KMeans<T>>,
    v: &[T],
) -> Vec<u8> {
    match n_bits {
        4 => {
            let mut encoded = vec![0_u8; m.div_ceil(2)];
            for i in 0..m / 2 {
                let v0 = group_k_means[2 * i].find_nearest(v);
                let v1 = group_k_means[2 * i + 1].find_nearest(v);
                encoded[i] = (v0 | (v1 << 4)) as u8;
            }
            if m % 2 == 1 {
                encoded[m / 2] = group_k_means[m - 1].find_nearest(v) as u8;
            }
            encoded
        }
        8 => group_k_means
            .iter()
            .map(|g| g.find_nearest(v) as u8)
            .collect(),
        _ => panic!("n_bits must be 4 or 8 in PQTable."),
    }
}

/// Flattened lookup table for the vector to be queried.
/// Using the PQ in ADC (Asymmetric Distance Computation) algorithm.
#[derive(Debug, Clone)]
pub struct PQLookupTable<'a, T> {
    pq_table: &'a PQTable<T>,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQTable<T> {
    /// The configuration for the PQ table.
    pub config: PQConfig,
    /// The dimension of the source vectors.
    pub dim: usize,
    /// `k = 2**n_bits` is the number of centroids for each group.
    /// *Cached for convenience.*
    pub k: usize,
    /// The dimension of each group.
    /// *Cached for convenience.*
    pub encoded_dim: usize,
    /// The k-means centroids for each group.
    pub encoded_vec_set: VecSet<u8>,
    /// The k-means centroids for each group.
    pub group_k_means: Vec<KMeans<T>>,
    /// The dot product of each centroid with itself (flattened).
    /// Size `(m * k,)`, only used for the Cosine distance.
    /// Otherwise, it is empty.
    pub dot_product_cache: Vec<f32>,
}

impl<T: Scalar> PQTable<T> {
    /// Create a new PQ table from the given vector set.
    pub fn from_vec_set(vec_set: &VecSet<T>, config: PQConfig, rng: &mut impl Rng) -> PQTable<T> {
        assert!(
            config.n_bits == 4 || config.n_bits == 8,
            "n_bits must be 4 or 8 in PQTable."
        );
        let m = config.m;
        assert!(
            vec_set.dim() % m == 0,
            "dim must be a multiple of m in PQTable."
        );
        let sub_vec_set = config
            .k_means_size
            .map(|size| vec_set.random_sample(size, rng));
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
            let k_means_vec_set = sub_vec_set.as_ref().unwrap_or(vec_set);
            let k_means = KMeans::from_vec_set(k_means_vec_set, k_means_config.clone(), rng);
            if config.dist == Cosine {
                for c in k_means.centroids.iter() {
                    dot_product_cache.push(T::dot_product(c, c))
                }
            }
            group_k_means.push(k_means);
        }
        let encoded_dim = match config.n_bits {
            4 => m.div_ceil(2),
            8 => m,
            _ => panic!("n_bits must be 4 or 8 in PQTable."),
        };
        let mut encoded_vec_set = VecSet::with_capacity(encoded_dim, vec_set.len());
        for v in vec_set.iter() {
            encoded_vec_set.push(&pq_encode(m, config.n_bits, &group_k_means, v));
        }
        Self {
            config,
            dim,
            k,
            encoded_dim,
            encoded_vec_set,
            group_k_means,
            dot_product_cache,
        }
    }

    /// Create flattened lookup table for the vector to be queried.
    /// Using the PQ in ADC (Asymmetric Distance Computation) algorithm.
    ///
    /// Size `(m * k,)`. For group `i` and centroid `c`, the cached value is at `i * k + c`.
    ///
    /// - For L2Sqr | L2, cache the L2Sqr distance between the centroid and the vector.
    /// - For DotProduct | Cosine, cache the dot product of the centroid and the vector.
    pub fn create_lookup(&self, v: &[T]) -> PQLookupTable<'_, T> {
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
                    .map(|c| T::l2_sqr_distance(vs, c))
                    .for_each(|d| lookup.push(d)),
                IP | Cosine => centroids
                    .iter()
                    .map(|c| T::dot_product(vs, c))
                    .for_each(|d| lookup.push(d)),
            };
        }
        let norm = match self.config.dist {
            Cosine => T::dot_product(v, v).sqrt(),
            _ => 0.0,
        };
        PQLookupTable {
            pq_table: self,
            lookup,
            norm,
        }
    }

    pub fn save(&self, path: &impl AsRef<Path>) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }
    pub fn load(path: &impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let pq_table: PQTable<T> = bincode::deserialize_from(reader)?;
        Ok(pq_table)
    }
}
impl<T: Scalar> DistanceAdapter<[u8], PQLookupTable<'_, T>> for DistanceAlgorithm {
    fn distance(&self, encoded: &[u8], lookup_table: &PQLookupTable<'_, T>) -> f32 {
        let pq_table = lookup_table.pq_table;
        let config = &pq_table.config;
        assert_eq!(
            *self, config.dist,
            "DistanceAlgorithm mismatch in PQLookupTable."
        );

        let n_bits = config.n_bits;
        let k = 1 << n_bits;
        let m = config.m;

        let lookup = &lookup_table.lookup;

        let mut sum = 0.0;
        let mut norm0_sqr = 0.0;

        let mut push_one = |i: usize, idx: usize| {
            if i >= m {
                return;
            }
            sum += lookup[i * k + idx];
            if *self == Cosine {
                norm0_sqr += pq_table.dot_product_cache[i * k + idx];
            }
        };

        match n_bits {
            4 => {
                assert_eq!(
                    encoded.len(),
                    m.div_ceil(2),
                    "encoded.len() mismatch in PQTable."
                );
                let mut i = 0;
                for u in encoded {
                    push_one(i, (u & 0xf) as usize);
                    i += 1;
                    push_one(i, (u >> 4) as usize);
                    i += 1;
                }
            }
            8 => {
                assert_eq!(encoded.len(), m);
                for i in 0..m {
                    push_one(i, encoded[i] as usize);
                }
            }
            _ => panic!("n_bits must be 4 or 8 in PQTable."),
        }

        match self {
            L2Sqr | IP => sum,
            L2 => sum.sqrt(),
            Cosine => {
                let dot_product = sum;
                let norm0 = norm0_sqr.sqrt();
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

    use crate::{config::VecDataConfig, distance::DistanceAdapter, scalar::Scalar};

    use super::*;

    fn pq_table_precise_test_base(dist: DistanceAlgorithm) {
        // Test the PQ table with num_vec < k, so that the centroids are the same as the vectors.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dim = 8;
        let n_bits = 4;
        let m = 2;
        let num_vec = 5;

        let mut src_set = VecSet::<f32>::with_capacity(dim, num_vec);
        for _ in 0..num_vec {
            let v = (0..dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect::<Vec<_>>();
            src_set.push(&v);
        }
        let pq_config = PQConfig {
            n_bits,
            m,
            dist,
            k_means_size: None,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        };
        let pq_table = PQTable::from_vec_set(&src_set, pq_config, &mut rng);

        let encoded_set = &pq_table.encoded_vec_set;

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
        let pq_config = PQConfig {
            n_bits: 4,
            m: dim / 4,
            dist,
            k_means_size: None,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pq_table = PQTable::from_vec_set(&vec_set, pq_config, &mut rng);
        let encoded_set = &pq_table.encoded_vec_set;

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

        let file_path = "config/gist_1000.toml";
        let mut config = VecDataConfig::load_from_toml_file(file_path)?;

        config.limit = Some(64);

        let raw_vec_set = VecSet::<f32>::load_with(&config)?;

        let clipped_dim = raw_vec_set.dim().min(12);

        let mut vec_set = VecSet::with_capacity(clipped_dim, raw_vec_set.len());
        for vec in raw_vec_set.iter() {
            vec_set.push(&vec[..clipped_dim]);
        }

        pq_table_test_base(&vec_set, L2Sqr)?;
        pq_table_test_base(&vec_set, L2)?;
        pq_table_test_base(&vec_set, IP)?;
        pq_table_test_base(&vec_set, Cosine)?;
        Ok(())
    }
}
