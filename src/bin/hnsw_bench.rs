use std::path::Path;

use anyhow::Result;
use clap::Parser;
use lab_1806_vec_db::{
    config::VecDataConfig,
    distance::DistanceAlgorithm,
    index_algorithm::{candidate_pair::GroundTruth, HNSWConfig, HNSWIndex},
    prelude::*,
    scalar::Scalar,
    vec_set::VecSet,
};
use rand::SeedableRng;

/// Generate ground truth for the test set by LinearIndex
#[derive(Parser)]
struct Args {
    /// Path to the train set config file
    train: String,
    /// Path to the test set config file
    #[clap(short, long, default_value = "config/gist_test.toml")]
    test: String,
    /// Path to the ground truth file
    #[clap(short, long)]
    gnd: String,
    #[clap(long, default_value = "data/gist_hnsw.local.bin")]
    index_cache: String,
    /// The search radius for the HNSW construction
    #[clap(long, default_value = "200")]
    ef_construction: usize,
    /// The number of neighbors in HNSW
    #[clap(short = 'M', default_value = "16")]
    m: usize,
    /// The search radius for the HNSW search
    #[clap(long, default_value = "50")]
    ef: usize,
}

struct AvgRecorder {
    sum: f32,
    count: usize,
}
impl AvgRecorder {
    fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
    fn add(&mut self, value: f32) {
        self.sum += value;
        self.count += 1;
    }
    fn avg(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f32
    }
}
fn try_load<T: Scalar>(args: &Args, train_set: &VecSet<T>) -> Option<HNSWIndex<T>> {
    let path = Path::new(&args.index_cache);
    println!("Trying to load index from {}...", path.display());
    match HNSWIndex::<T>::load(&args.index_cache) {
        Ok(index) => {
            if index.len() == train_set.len() {
                println!("Index loaded successfully.");
                return Some(index);
            }
            println!("Index file is outdated, need to rebuild.");
            None
        }
        Err(err) => {
            println!("Failed to load index: {}", err);
            None
        }
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    let train_config = VecDataConfig::load_from_toml_file(&args.train)?;
    let test_config = VecDataConfig::load_from_toml_file(&args.test)?;
    let train_set = VecSet::<f32>::load_with(&train_config)?;
    println!("Loaded train set (size: {}).", train_set.len());

    let test_set = VecSet::<f32>::load_with(&test_config)?;
    println!("Loaded test set (size: {}).", test_set.len());

    let dist = DistanceAlgorithm::L2Sqr;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let config = HNSWConfig {
        max_elements: train_set.len(),
        ef_construction: args.ef_construction,
        M: args.m,
    };

    let start = std::time::Instant::now();
    let index = match try_load(&args, &train_set) {
        Some(index) => index,
        None => {
            println!("Building index...");
            let mut index = HNSWIndex::<f32>::new(train_set.dim(), dist, config);
            for vec in train_set.iter() {
                index.add(vec, &mut rng);
            }
            println!("Indexing time: {:.3}s", start.elapsed().as_secs_f32());
            index.save(&args.index_cache)?;
            index
        }
    };
    let gnd = GroundTruth::load(&args.gnd)?;
    let k = gnd[0].knn_indices.len(); // default 10
    println!("Loaded ground truth (size: {}).", gnd.len());

    let mut recall_avg = AvgRecorder::new();

    let start = std::time::Instant::now();
    for (query, gnd) in test_set.iter().zip(gnd.iter()) {
        let result_set = index.knn_with_ef(query, k, args.ef);
        let recall = gnd.recall(&result_set);
        recall_avg.add(recall);
    }
    let elapsed = start.elapsed().as_secs_f32();
    println!("Total time: {:.3}s", elapsed);
    println!(
        "Average Search Time: {:.6}ms",
        elapsed * 1000.0 / test_set.len() as f32
    );
    println!("Average recall: {:.4}", recall_avg.avg());

    Ok(())
}
// cargo r -r --bin hnsw_bench -- config/gist_10000.local.toml -g data/gnd_10000.local.bin
