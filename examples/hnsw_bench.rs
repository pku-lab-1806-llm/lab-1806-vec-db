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
use plotly::{Plot, Scatter};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

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
    /// Path to the index cache file
    #[clap(long, default_value = "data/gist_hnsw.local.bin")]
    index_cache: String,
    /// Path to the benchmark result html file (optional)
    #[clap(short, long)]
    output: Option<String>,
    /// The search radius for the HNSW construction
    #[clap(long, default_value = "200")]
    ef_construction: usize,
    /// The number of neighbors in HNSW
    #[clap(short = 'M', default_value = "16")]
    m: usize,

    /// The start value of ef for benchmarking
    #[clap(long, default_value = "30")]
    ef_start: usize,
    /// The end value of ef for benchmarking
    #[clap(long, default_value = "200")]
    ef_end: usize,
    /// The step value of ef for benchmarking
    #[clap(long, default_value = "10")]
    ef_step: usize,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchResult {
    /// The search radius for the HNSW construction
    ef: Vec<usize>,
    /// Average search time in ms
    search_time: Vec<f32>,
    /// Recall (k = 10)
    recall: Vec<f32>,
}
impl BenchResult {
    pub fn new() -> Self {
        Self {
            ef: Vec::new(),
            search_time: Vec::new(),
            recall: Vec::new(),
        }
    }
    pub fn plot(self, title: impl Into<String>, path: Option<impl AsRef<Path>>) -> Result<()> {
        let mut plot = Plot::new();
        println!("=== Source Data ===");
        println!("ef: {:?}", self.ef);
        println!("search_time: {:?}", self.search_time);
        println!("recall: {:?}", self.recall);
        let text = self
            .ef
            .iter()
            .map(|&ef| format!("ef={}", ef))
            .collect::<Vec<_>>();
        let trace = Scatter::new(self.search_time, self.recall)
            .text_array(text)
            .text_font(plotly::common::Font::new().size(10).family("Arial"))
            .text_position(plotly::common::Position::BottomRight)
            .mode(plotly::common::Mode::LinesMarkersText)
            .name("HNSW");
        plot.add_trace(trace);

        let layout = plotly::Layout::new()
            .title(plotly::common::Title::with_text(title))
            .x_axis(plotly::layout::Axis::new().title("Search Time(ms)"))
            .y_axis(plotly::layout::Axis::new().title("Recall"))
            .show_legend(true);
        plot.set_layout(layout);

        if let Some(path) = path {
            println!("Saved plot to {}", path.as_ref().display());
            plot.write_html(path);
        }
        println!("Try to show the plot...");
        plot.show_image(plotly::ImageFormat::SVG, 1024, 768);
        Ok(())
    }
    pub fn push(&mut self, ef: usize, search_time: f32, recall: f32) {
        self.ef.push(ef);
        self.search_time.push(search_time);
        self.recall.push(recall);
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

    let mut bench_result = BenchResult::new();

    for ef in (args.ef_start..=args.ef_end).step_by(args.ef_step) {
        let mut avg_recall = AvgRecorder::new();

        let start = std::time::Instant::now();
        for (query, gnd) in test_set.iter().zip(gnd.iter()) {
            let result_set = index.knn_with_ef(query, k, ef);
            let recall = gnd.recall(&result_set);
            avg_recall.add(recall);
        }
        let elapsed = start.elapsed().as_secs_f32();
        // ms
        let search_time = elapsed * 1000.0 / test_set.len() as f32;
        let recall = avg_recall.avg();

        println!(
            "ef: {}, Average Search Time: {:.6}ms, Average recall: {:.4}",
            ef, search_time, recall
        );
        bench_result.push(ef, search_time, recall);
    }
    println!("Finished benchmarking.");
    let title = format!("HNSW Bench ({} elements)", train_set.len());
    bench_result.plot(title, args.output.as_ref())?;
    Ok(())
}
// cargo r -r --example hnsw_bench -- config/gist_10000.local.toml -g data/gnd_10000.local.bin
