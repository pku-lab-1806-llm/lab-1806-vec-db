use std::thread;

use anyhow::Result;
use clap::Parser;
use lab_1806_vec_db::{
    config::VecDataConfig,
    distance::DistanceAlgorithm,
    index_algorithm::{candidate_pair::GroundTruth, LinearIndex},
    prelude::*,
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
    /// Path to the output ground truth file
    #[clap(short, long)]
    output_file: String,
    /// The number of threads
    #[clap(long, default_value = "20")]
    threads: usize,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let train_config = VecDataConfig::load_from_toml_file(&args.train)?;
    let test_config = VecDataConfig::load_from_toml_file(&args.test)?;
    let train_set = VecSet::<f32>::load_with(&train_config)?;
    println!("Loaded train set (size: {}).", train_set.len());

    let dist = DistanceAlgorithm::L2Sqr;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let k = 10;

    let index = LinearIndex::from_vec_set(train_set, dist, (), &mut rng);
    let test_set = VecSet::<f32>::load_with(&test_config)?;
    println!("Loaded test set (size: {}).", test_set.len());

    let mut ground_truth = GroundTruth::new();
    println!("Generating ground truth...");
    println!("Using {} threads.", args.threads);

    assert!(
        args.threads > 0,
        "The number of threads must be greater than 0."
    );
    let (sender, receiver) = std::sync::mpsc::channel();
    let threads_ids: Vec<_> = (0..args.threads).collect();
    thread::scope(|s| {
        for t_idx in threads_ids.iter() {
            s.spawn(|| {
                let mut results = Vec::new();
                for i in (*t_idx..test_set.len()).step_by(args.threads) {
                    results.push((i, index.knn(&test_set[i], k)));
                }
                sender.send(results).unwrap();
            });
        }
    });
    // Drop the sender to close the channel.
    drop(sender);
    let mut result: Vec<_> = receiver.iter().flatten().collect();
    result.sort_by_key(|(i, _)| *i);
    for (_, knn) in result {
        ground_truth.push(knn);
    }
    println!("Saving ground truth to {}...", args.output_file);
    ground_truth.save(&args.output_file)?;
    Ok(())
}
// cargo r -r --bin gen_ground_truth -- config/gist_10000.local.toml -o data/gnd_10000.local.bin
// cargo r -r --bin gen_ground_truth -- config/gist.local.toml -o data/gnd.local.bin
