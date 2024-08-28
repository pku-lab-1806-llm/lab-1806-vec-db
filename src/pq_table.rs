use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    binary_scalar::BinaryScalar,
    config::DistanceAlgorithm,
    distance::Distance,
    k_means::{KMeans, KMeansConfig},
    vec_set::{DynamicVecRef, DynamicVecSet, VecSet},
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
    /// - For L2Sqr and L2 distance, cache the L2Sqr distance.
    /// - For Cosine distance, cache the dot product.
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
                Cosine => {
                    for c0 in 0..k {
                        for c1 in 0..k {
                            lookup_table.push(cs[c0].dot_product_sqr(&cs[c1]));
                        }
                    }
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

    /// Get the lookup value from the lookup table.
    pub fn lookup(&self, i: usize, c0: usize, c1: usize) -> f32 {
        self.lookup_table[i * self.k * self.k + c0 * self.k + c1]
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
            d += self.lookup(i, v0[i], v1[i]);
        }
        match dist {
            L2Sqr => d,
            L2 => d.sqrt(),
            Cosine => {
                let mut norm0 = 0.0;
                let mut norm1 = 0.0;
                for i in 0..m {
                    norm0 += self.lookup(i, v0[i], v0[i]);
                    norm1 += self.lookup(i, v1[i], v1[i]);
                }
                let norm0 = norm0.sqrt();
                let norm1 = norm1.sqrt();
                1.0 - d / (norm0 * norm1)
            }
        }
    }

    /// Alias of `PQTable::distance()`.
    /// Compute the distance between two encoded vectors.
    pub fn d(&self, v0: &[u8], v1: &[u8]) -> f32 {
        self.distance(v0, v1)
    }
}

#[derive(Debug)]
pub enum DynamicPQTable {
    UInt8(PQTable<u8>),
    Float32(PQTable<f32>),
}

impl DynamicPQTable {
    /// Create a new dynamic PQ table from the given vector set.
    pub fn from_vec_set(vec_set: &DynamicVecSet, config: &PQConfig, rng: &mut impl Rng) -> Self {
        use DynamicVecSet::*;
        match vec_set {
            UInt8(vec_set) => Self::UInt8(PQTable::from_vec_set(vec_set, config, rng)),
            Float32(vec_set) => Self::Float32(PQTable::from_vec_set(vec_set, config, rng)),
        }
    }

    /// Encode the given vector.
    pub fn encode(&self, v: &DynamicVecRef) -> Vec<u8> {
        match (self, v) {
            (Self::UInt8(pq_table), DynamicVecRef::UInt8(v)) => pq_table.encode(v),
            (Self::Float32(pq_table), DynamicVecRef::Float32(v)) => pq_table.encode(v),
            _ => panic!("Cannot encode different types of vectors."),
        }
    }

    /// Get the dimension of the encoded vector.
    pub fn encoded_dim(&self) -> usize {
        match self {
            Self::UInt8(pq_table) => pq_table.encoded_dim(),
            Self::Float32(pq_table) => pq_table.encoded_dim(),
        }
    }

    /// Encode the given vector set.
    pub fn encode_batch(&self, vec_set: &DynamicVecSet) -> VecSet<u8> {
        match (self, vec_set) {
            (Self::UInt8(pq_table), DynamicVecSet::UInt8(vec_set)) => {
                pq_table.encode_batch(vec_set)
            }
            (Self::Float32(pq_table), DynamicVecSet::Float32(vec_set)) => {
                pq_table.encode_batch(vec_set)
            }
            _ => panic!("Cannot encode different types of vectors."),
        }
    }

    /// Get indices from a encoded vector.
    pub fn split_indices(&self, v: &[u8]) -> Vec<usize> {
        match self {
            Self::UInt8(pq_table) => pq_table.split_indices(v),
            Self::Float32(pq_table) => pq_table.split_indices(v),
        }
    }

    /// Get the lookup value from the lookup table.
    pub fn lookup(&self, i: usize, c0: usize, c1: usize) -> f32 {
        match self {
            Self::UInt8(pq_table) => pq_table.lookup(i, c0, c1),
            Self::Float32(pq_table) => pq_table.lookup(i, c0, c1),
        }
    }

    /// Compute the distance between two encoded vectors.
    pub fn distance(&self, v0: &[u8], v1: &[u8]) -> f32 {
        match self {
            Self::UInt8(pq_table) => pq_table.distance(v0, v1),
            Self::Float32(pq_table) => pq_table.distance(v0, v1),
        }
    }

    /// Alias of `DynamicPQTable::distance()`.
    /// Compute the distance between two encoded vectors.
    pub fn d(&self, v0: &[u8], v1: &[u8]) -> f32 {
        self.distance(v0, v1)
    }
}

#[cfg(test)]
mod test {
    use anyhow::Result;
    use rand::SeedableRng;

    use crate::config::DBConfig;

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
        let pq_config = PQConfig {
            n_bits,
            m,
            dist,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        };
        let pq_table = PQTable::from_vec_set(&src_set, &pq_config, &mut rng);

        let encoded_set = pq_table.encode_batch(&src_set);
        for i in 0..num_vec {
            let src = &src_set[i];
            let decoded = pq_table.decode(&encoded_set[i]);
            println!("{}: {:?}", i, &src_set[i]);
            assert_eq!(src, &decoded);
        }
        for i in 0..num_vec {
            for j in 0..num_vec {
                let src0 = &src_set[i];
                let src1 = &src_set[j];
                let src_dist = dist.d(src0, src1);

                let e0 = &encoded_set[i];
                let e1 = &encoded_set[j];
                let e_dist = pq_table.d(e0, e1);

                if i <= j {
                    println!("{}<->{}: src={:.6} encoded={:.6}", i, j, src_dist, e_dist);
                }
                assert!((src_dist - e_dist).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn pq_table_precise_test() {
        pq_table_precise_test_base(DistanceAlgorithm::L2Sqr);
        pq_table_precise_test_base(DistanceAlgorithm::L2);
        pq_table_precise_test_base(DistanceAlgorithm::Cosine);
    }

    fn pq_table_test_on_real_set_base(
        dist: DistanceAlgorithm,
        p90_error_threshold: f32,
    ) -> Result<()> {
        let file_path = "config/example/db_config.toml";
        let mut config = DBConfig::load_from_toml_file(file_path)?;
        config.vec_data.limit = Some(128);
        let vec_set = DynamicVecSet::load_with(config.vec_data)?;
        let dim = vec_set.dim();
        let pq_config = PQConfig {
            n_bits: 4,
            m: dim / 4,
            dist,
            k_means_max_iter: 20,
            k_means_tol: 1e-6,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pq_table = DynamicPQTable::from_vec_set(&vec_set, &pq_config, &mut rng);

        println!("Distance Algorithm: {:?}", dist);
        let test_count = 40;
        let mut errors = Vec::new();
        for _ in 0..test_count {
            let i0 = rng.gen_range(0..vec_set.len());
            let i1 = rng.gen_range(0..vec_set.len());
            let v0 = vec_set.i(i0);
            let v1 = vec_set.i(i1);
            let distance = pq_table.d(&pq_table.encode(&v0), &pq_table.encode(&v1));
            let expected = dist.d(&v0, &v1);
            let error = (distance - expected).abs() / expected.max(1.0);
            println!(
                "Distance: {} / Expected: {} / Error: {}",
                distance, expected, error
            );
            errors.push(error);
        }
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i90 = (errors.len() as f32 * 0.9).floor() as usize;
        let p90 = errors[i90];
        println!("90% Error: {}", p90);
        assert!(p90 < p90_error_threshold, "90% Error is too large.");
        Ok(())
    }

    #[test]
    fn pq_table_test_on_real_set() -> Result<()> {
        pq_table_test_on_real_set_base(DistanceAlgorithm::L2Sqr, 0.25)?;
        pq_table_test_on_real_set_base(DistanceAlgorithm::L2, 0.15)?;
        pq_table_test_on_real_set_base(DistanceAlgorithm::Cosine, 0.2)?;
        Ok(())
    }
}
