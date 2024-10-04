use std::path::Path;

use anyhow::Result;
use clap::Parser;
use lab_1806_vec_db::{
    config::VecDataConfig,
    distance::DistanceAlgorithm,
    index_algorithm::{candidate_pair::GroundTruth, IVFConfig, IVFIndex},
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
    /// Path to the base set config file
    base: String,
    /// Path to the test set config file
    #[clap(short, long, default_value = "config/gist_test.toml")]
    test: String,
    /// Path to the ground truth file
    #[clap(short, long)]
    gnd: String,
    /// Path to the index cache file
    #[clap(long, default_value = "data/gist_ivf.local.bin")]
    index_cache: String,
    /// Path to the benchmark result html file (optional)
    #[clap(short, long)]
    output: Option<String>,

    /// The number of clusters.
    #[clap(long, default_value = "128")]
    pub k: usize,
    /// The number of vectors to be sampled for the k-means algorithm.
    #[clap(long, default_value = "1000")]
    pub k_means_size: Option<usize>,
    /// The number of iterations for the k-means algorithm.
    #[clap(long, default_value = "20")]
    pub k_means_max_iter: usize,
    /// The tolerance for the k-means algorithm.
    #[clap(long, default_value = "1e-6")]
    pub k_means_tol: f32,

    /// The start value of n_probes for benchmarking
    #[clap(long, default_value = "8")]
    n_start: usize,
    /// The end value of n_probes for benchmarking
    #[clap(long, default_value = "24")]
    n_end: usize,
    /// The step value of n_probes for benchmarking
    #[clap(long, default_value = "4")]
    n_step: usize,
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
fn load_or_build<T: Scalar>(args: &Args, base_set: VecSet<T>) -> Result<IVFIndex<T>> {
    let path = Path::new(&args.index_cache);

    if path.exists() {
        println!("Trying to load index from {}...", path.display());
        let start = std::time::Instant::now();
        let index = IVFIndex::load_with_external_vec_set(&args.index_cache, base_set)?;
        let elapsed = start.elapsed().as_secs_f32();
        println!("Index loaded in {:.2} seconds.", elapsed);
        Ok(index)
    } else {
        println!("Index file not found. Building index...");
        let dist = DistanceAlgorithm::L2Sqr;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = IVFConfig {
            k: args.k,
            k_means_size: args.k_means_size,
            k_means_max_iter: args.k_means_max_iter,
            k_means_tol: args.k_means_tol,
        };
        let start = std::time::Instant::now();
        let index = IVFIndex::from_vec_set(base_set, dist, config, &mut rng);
        let elapsed = start.elapsed().as_secs_f32();
        println!("Index built in {:.2} seconds.", elapsed);
        println!("Saving index to {}...", path.display());
        let index = index.save_without_vec_set(&path)?;
        println!("Index saved.");
        Ok(index)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchResult {
    /// The search radius for the IVF construction
    n_probes: Vec<usize>,
    /// Average search time in ms
    search_time: Vec<f32>,
    /// Recall (k = 10)
    recall: Vec<f32>,
}
impl BenchResult {
    pub fn new() -> Self {
        Self {
            n_probes: Vec::new(),
            search_time: Vec::new(),
            recall: Vec::new(),
        }
    }
    pub fn plot(self, title: impl Into<String>, path: Option<impl AsRef<Path>>) -> Result<()> {
        let mut plot = Plot::new();
        println!("=== Source Data ===");
        println!("n_probes: {:?}", self.n_probes);
        println!("search_time: {:?}", self.search_time);
        println!("recall: {:?}", self.recall);
        let text = self
            .n_probes
            .iter()
            .map(|&n| format!("n={}", n))
            .collect::<Vec<_>>();
        let trace = Scatter::new(self.search_time, self.recall)
            .text_array(text)
            .text_font(plotly::common::Font::new().size(10).family("Arial"))
            .text_position(plotly::common::Position::BottomRight)
            .mode(plotly::common::Mode::LinesMarkersText)
            .name("IVF");
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
        #[cfg(target_os = "windows")]
        {
            println!("Try to show the plot...");
            plot.show_image(plotly::ImageFormat::SVG, 1024, 768);
        }
        Ok(())
    }
    pub fn push(&mut self, n_probes: usize, search_time: f32, recall: f32) {
        self.n_probes.push(n_probes);
        self.search_time.push(search_time);
        self.recall.push(recall);
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    let load_start = std::time::Instant::now();
    let base_config = VecDataConfig::load_from_toml_file(&args.base)?;
    let test_config = VecDataConfig::load_from_toml_file(&args.test)?;
    let base_set = VecSet::<f32>::load_with(&base_config)?;
    println!("Loaded base set (size: {}).", base_set.len());

    let test_set = VecSet::<f32>::load_with(&test_config)?;
    println!("Loaded test set (size: {}).", test_set.len());
    let elapsed = load_start.elapsed().as_secs_f32();
    println!("VecSet loaded in {:.2} seconds.", elapsed);

    let index = load_or_build(&args, base_set)?;
    let gnd = GroundTruth::load(&args.gnd)?;
    let k = gnd[0].knn_indices.len(); // default 10
    println!("Loaded ground truth (size: {}).", gnd.len());

    let mut bench_result = BenchResult::new();

    for n in (args.n_start..=args.n_end).step_by(args.n_step) {
        let mut avg_recall = AvgRecorder::new();

        let start = std::time::Instant::now();
        for (query, gnd) in test_set.iter().zip(gnd.iter()) {
            let result_set = index.knn_with_ef(query, k, n);
            let recall = gnd.recall(&result_set);
            avg_recall.add(recall);
        }
        let elapsed = start.elapsed().as_secs_f32();
        // ms
        let search_time = elapsed * 1000.0 / test_set.len() as f32;
        let recall = avg_recall.avg();

        println!(
            "n_probes: {}, Average Search Time: {:.6}ms, Average recall: {:.4}",
            n, search_time, recall
        );
        bench_result.push(n, search_time, recall);
    }
    println!("Finished benchmarking.");
    let title = format!("IVF Bench ({} elements)", index.len());
    bench_result.plot(title, args.output.as_ref())?;
    Ok(())
}

// It will be used in LLM RAG Application, so we care about the 1e4 scale data.
// cargo r -r --example ivf_bench -- config/gist_10000.local.toml -g data/gnd_10000.local.bin --index-cache data/gist_10000_ivf.local.bin
