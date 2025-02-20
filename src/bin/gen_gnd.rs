use anyhow::Result;
use clap::Parser;
use lab_1806_vec_db::{
    config::{DataType, VecDataConfig},
    distance::DistanceAlgorithm,
    index_algorithm::{candidate_pair::GroundTruth, FlatIndex},
    prelude::*,
    vec_set::VecSet,
};
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Generate ground truth for the test set by FlatIndex
#[derive(Parser)]
struct Args {
    /// Dimension of the vectors
    #[clap(short, long, default_value = "960")]
    dim: usize,
    /// Path to the base set
    #[clap(long, default_value = "data/gist.local.bin")]
    base: String,
    /// Path to the test set
    #[clap(long, default_value = "data/gist_test.bin")]
    test: String,
    /// Path to the output ground truth file
    #[clap(short, long, default_value = "data/gnd.local.bin")]
    out: String,
    #[clap(long, default_value = "L2Sqr")]
    dist_fn: String,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let base_config = VecDataConfig {
        dim: args.dim,
        data_type: DataType::Float32,
        data_path: args.base,
        limit: None,
    };
    let test_config = VecDataConfig {
        dim: args.dim,
        data_type: DataType::Float32,
        data_path: args.test,
        limit: None,
    };
    let base_set = VecSet::<f32>::load_with(&base_config)?;
    println!("Loaded base set (size: {}).", base_set.len());

    let dist = match args.dist_fn.as_str() {
        "L2Sqr" => DistanceAlgorithm::L2Sqr,
        "Cosine" => DistanceAlgorithm::Cosine,
        _ => panic!("Unknown distance function: {}", args.dist_fn),
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let k = 10;

    let index = FlatIndex::from_vec_set(base_set, dist, (), &mut rng);
    let test_set = VecSet::<f32>::load_with(&test_config)?;
    println!("Loaded test set (size: {}).", test_set.len());

    let mut ground_truth = GroundTruth::new();
    println!("Generating ground truth...");

    let test_refs = test_set.iter().collect::<Vec<_>>();

    let raw_gnd = test_refs
        .par_iter()
        .map(|query| index.knn(query, k))
        .collect::<Vec<_>>();

    for gnd in raw_gnd {
        ground_truth.push(gnd);
    }
    println!("Saving ground truth to {}...", args.out);
    ground_truth.save(&args.out)?;
    Ok(())
}
// cargo r -r --bin gen_gnd
// cargo r -r --bin gen_gnd -- --base data/gist_10000.local.bin --out data/gnd_10000.local.bin
