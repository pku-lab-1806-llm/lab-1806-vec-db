use criterion::{criterion_group, criterion_main, Criterion};

use anyhow::Result;
use std::{path::Path, time::Duration};

use clap::Parser;
use lab_1806_vec_db::{
    config::{IndexAlgorithmConfig, VecDataConfig},
    distance::{
        pq_table::{PQConfig, PQTable},
        DistanceAlgorithm,
    },
    index_algorithm::{CandidatePair, HNSWIndex, IVFIndex, LinearIndex},
    prelude::*,
    scalar::Scalar,
    vec_set::VecSet,
};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchEfRange {
    start: usize,
    end: usize,
    step: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
enum BenchEf {
    #[serde(rename = "range")]
    Range(BenchEfRange),
    #[serde(rename = "list")]
    List(Vec<usize>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchPQConfig {
    /// Path to the PQ cache file
    pq_cache: String,
    /// The number of bits for each quantized group.
    ///
    /// Should be 4 or 8. Usually 4.
    pub n_bits: usize,
    /// The number of groups.
    ///
    /// Should satisfy `dim % m == 0`. Usually `dim / 4`.
    pub m: usize,
    /// The number of vectors to be sampled for the k-means algorithm.
    pub k_means_size: Option<usize>,
    /// The number of iterations for the k-means algorithm.
    pub k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    pub k_means_tol: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchConfig {
    dist: DistanceAlgorithm,
    label: String,
    gnd_path: String,
    index_cache: String,
    pq_cache: Option<String>,
    ef: BenchEf,
    algorithm: IndexAlgorithmConfig,
    #[serde(rename = "PQ")]
    pq: Option<BenchPQConfig>,
    base: VecDataConfig,
    test: VecDataConfig,
    bench_output: String,
}

impl BenchConfig {
    pub fn load_from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: BenchConfig = toml::from_str(&content)?;
        Ok(config)
    }
}

/// Generate ground truth for the test set by LinearIndex
#[derive(Parser)]
struct Args {
    /// Path to the benchmark config file
    bench_config_path: String,
}

enum DynamicIndex<T> {
    HNSW(HNSWIndex<T>),
    IVF(IVFIndex<T>),
    Linear(LinearIndex<T>),
}
impl<T: Scalar> DynamicIndex<T> {
    pub fn knn_with_ef(
        &self,
        query: &[T],
        k: usize,
        ef: usize,
        pq: &Option<PQTable<T>>,
    ) -> Vec<CandidatePair> {
        use DynamicIndex::*;
        match (self, pq) {
            (HNSW(index), Some(pq)) => index.knn_pq(query, k, ef, pq),
            (Linear(index), Some(pq)) => index.knn_pq(query, k, ef, pq),
            (HNSW(index), _) => index.knn_with_ef(query, k, ef),
            (IVF(index), _) => index.knn_with_ef(query, k, ef),
            _ => unimplemented!("({:?}, {:?}) is not implemented.", self.index_name(), pq),
        }
    }
    pub fn index_name(&self) -> String {
        match self {
            DynamicIndex::HNSW(_) => "HNSW".to_string(),
            DynamicIndex::IVF(_) => "IVF".to_string(),
            DynamicIndex::Linear(_) => "Linear".to_string(),
        }
    }
}
fn load_or_build_pq<T: Scalar>(
    config: &BenchConfig,
    base_set: &VecSet<T>,
    rng: &mut impl Rng,
) -> Result<Option<PQTable<T>>> {
    let (config, pq_cache) = match &config.pq {
        Some(pq_config) => (
            PQConfig {
                dist: config.dist,
                n_bits: pq_config.n_bits,
                m: pq_config.m,
                k_means_size: pq_config.k_means_size,
                k_means_max_iter: pq_config.k_means_max_iter,
                k_means_tol: pq_config.k_means_tol,
            },
            pq_config.pq_cache.clone(),
        ),
        None => return Ok(None),
    };
    let path = Path::new(&pq_cache);
    if path.exists() {
        println!("Trying to load PQTable from {}...", pq_cache);
        let pq_table = PQTable::load(&path)?;
        println!("PQTable loaded.");
        return Ok(Some(pq_table));
    }
    println!("PQTable file not found. Building PQTable...");
    let start = std::time::Instant::now();
    let pq = PQTable::from_vec_set(base_set, config, rng);
    let elapsed = start.elapsed().as_secs_f32();
    println!("PQTable built in {:.2} seconds.", elapsed);
    println!("Saving PQTable to {}...", pq_cache);
    pq.save(&pq_cache)?;
    println!("PQTable saved.");
    Ok(Some(pq))
}
fn load_or_build_index<T: Scalar>(
    config: BenchConfig,
    base_set: VecSet<T>,
    rng: &mut impl Rng,
) -> Result<DynamicIndex<T>> {
    let path = Path::new(&config.index_cache);

    if path.exists() {
        println!("Trying to load index from {}...", path.display());
        let start = std::time::Instant::now();
        let index = match config.algorithm {
            IndexAlgorithmConfig::HNSW(_) => {
                let index = HNSWIndex::load_with_external_vec_set(&config.index_cache, base_set)?;
                DynamicIndex::HNSW(index)
            }
            IndexAlgorithmConfig::IVF(_) => {
                let index = IVFIndex::load_with_external_vec_set(&config.index_cache, base_set)?;
                DynamicIndex::IVF(index)
            }
            IndexAlgorithmConfig::Linear => {
                let index = LinearIndex::load_with_external_vec_set(&config.index_cache, base_set)?;
                DynamicIndex::Linear(index)
            }
        };
        let elapsed = start.elapsed().as_secs_f32();
        println!("Index loaded in {:.2} seconds.", elapsed);
        Ok(index)
    } else {
        println!("Index file not found. Building index...");
        let dist = DistanceAlgorithm::L2Sqr;

        let start = std::time::Instant::now();
        let index = match config.algorithm {
            IndexAlgorithmConfig::HNSW(config) => {
                let index = HNSWIndex::build_on_vec_set(base_set, dist, config, true, rng);
                println!("Saving index to {}...", path.display());
                let index = index.save_without_vec_set(&path)?;
                DynamicIndex::HNSW(index)
            }
            IndexAlgorithmConfig::IVF(config) => {
                let index = IVFIndex::from_vec_set(base_set, dist, config, rng);
                println!("Saving index to {}...", path.display());
                let index = index.save_without_vec_set(&path)?;
                DynamicIndex::IVF(index)
            }
            IndexAlgorithmConfig::Linear => {
                let index = LinearIndex::from_vec_set(base_set, dist, (), rng);
                println!("Saving index to {}...", path.display());
                let index = index.save_without_vec_set(&path)?;
                DynamicIndex::Linear(index)
            }
        };
        let elapsed = start.elapsed().as_secs_f32();
        println!("Index built in {:.2} seconds.", elapsed);
        println!("Index saved.");
        Ok(index)
    }
}

pub fn simd_bench(config_path: impl AsRef<Path>, ef: usize, c: &mut Criterion) -> Result<()> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let bench_config = BenchConfig::load_from_toml_file(&config_path)?;
    let label = bench_config.label.clone();
    let base_set = VecSet::<f32>::load_with(&bench_config.base)?;
    let test_set = VecSet::<f32>::load_with(&bench_config.test)?;
    let k = 10;
    let pq = load_or_build_pq(&bench_config, &base_set, &mut rng)?;
    let index = load_or_build_index(bench_config, base_set, &mut rng)?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(43);
    c.bench_function(&format!("{} (ef={})", label, ef), |b| {
        b.iter(|| {
            let idx = rng.gen_range(0..test_set.len());
            index.knn_with_ef(&test_set[idx], k, ef, &pq)
        })
    });
    Ok(())
}
pub fn simd_bench_wrapper(c: &mut Criterion) {
    simd_bench("config/bench_hnsw.toml", 240, c).unwrap();
    simd_bench("config/bench_simd_hnsw.toml", 240, c).unwrap();
    // simd_bench("config/bench_pq_hnsw.toml", 320, c).unwrap();
}
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(1000).measurement_time(Duration::from_secs(60));
    targets = simd_bench_wrapper
);
criterion_main!(benches);

// $ cargo bench
// HNSW (ef=240)           time:   [3.8459 ms 3.8642 ms 3.8826 ms]
// HNSW+SIMD (ef=240)      time:   [3.8496 ms 3.8683 ms 3.8869 ms]

// We find that the SIMD version just the same as the original version.
// since LLVM can automatically vectorize the code.
// No need to write SIMD code manually.
