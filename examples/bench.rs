use std::path::Path;

use anyhow::Result;
use clap::Parser;
use lab_1806_vec_db::{
    config::{IndexAlgorithmConfig, VecDataConfig},
    distance::DistanceAlgorithm,
    index_algorithm::{candidate_pair::GroundTruth, CandidatePair, HNSWIndex, IVFIndex},
    prelude::*,
    scalar::Scalar,
    vec_set::VecSet,
};
use plotly::{Plot, Scatter};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchEf {
    start: usize,
    end: usize,
    step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchConfig {
    dist: DistanceAlgorithm,
    gnd_path: String,
    index_cache: String,
    ef: BenchEf,
    algorithm: IndexAlgorithmConfig,
    base: VecDataConfig,
    test: VecDataConfig,
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
    /// Path to the benchmark result html file (optional)
    #[clap(short, long)]
    output: Option<String>,
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
enum DynamicIndex<T> {
    HNSW(HNSWIndex<T>),
    IVF(IVFIndex<T>),
}
impl<T: Scalar> DynamicIndex<T> {
    pub fn knn_with_ef(&self, query: &[T], k: usize, ef: usize) -> Vec<CandidatePair> {
        match self {
            DynamicIndex::HNSW(index) => index.knn_with_ef(query, k, ef),
            DynamicIndex::IVF(index) => index.knn_with_ef(query, k, ef),
        }
    }
    pub fn algorithm_name(&self) -> String {
        match self {
            DynamicIndex::HNSW(_) => "HNSW".to_string(),
            DynamicIndex::IVF(_) => "IVF".to_string(),
        }
    }
}
fn load_or_build<T: Scalar>(config: BenchConfig, base_set: VecSet<T>) -> Result<DynamicIndex<T>> {
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
            _ => unimplemented!("{:?} is not implemented.", config.algorithm),
        };
        let elapsed = start.elapsed().as_secs_f32();
        println!("Index loaded in {:.2} seconds.", elapsed);
        Ok(index)
    } else {
        println!("Index file not found. Building index...");
        let dist = DistanceAlgorithm::L2Sqr;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let start = std::time::Instant::now();
        let index = match config.algorithm {
            IndexAlgorithmConfig::HNSW(config) => {
                let index = HNSWIndex::build_on_vec_set(base_set, dist, config, true, &mut rng);
                println!("Saving index to {}...", path.display());
                let index = index.save_without_vec_set(&path)?;
                DynamicIndex::HNSW(index)
            }
            IndexAlgorithmConfig::IVF(config) => {
                let index = IVFIndex::from_vec_set(base_set, dist, config, &mut rng);
                println!("Saving index to {}...", path.display());
                let index = index.save_without_vec_set(&path)?;
                DynamicIndex::IVF(index)
            }
            _ => unimplemented!("{:?} is not implemented.", config.algorithm),
        };
        let elapsed = start.elapsed().as_secs_f32();
        println!("Index built in {:.2} seconds.", elapsed);
        println!("Index saved.");
        Ok(index)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchResult {
    /// The search radius
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
    pub fn plot(
        self,
        title: impl Into<String>,
        algorithm_name: String,
        path: Option<impl AsRef<Path>>,
    ) -> Result<()> {
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
            .name(algorithm_name);
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
    pub fn push(&mut self, ef: usize, search_time: f32, recall: f32) {
        self.ef.push(ef);
        self.search_time.push(search_time);
        self.recall.push(recall);
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    let load_start = std::time::Instant::now();
    let bench_config = BenchConfig::load_from_toml_file(&args.bench_config_path)?;
    let base_set = VecSet::<f32>::load_with(&bench_config.base)?;
    println!("Loaded base set (size: {}).", base_set.len());
    let test_set = VecSet::<f32>::load_with(&bench_config.test)?;
    println!("Loaded test set (size: {}).", test_set.len());
    let elapsed = load_start.elapsed().as_secs_f32();
    println!("VecSet loaded in {:.2} seconds.", elapsed);

    let base_size = base_set.len();

    let gnd = GroundTruth::load(&bench_config.gnd_path)?;
    let k = gnd[0].knn_indices.len(); // default 10
    println!("Loaded ground truth (size: {}).", gnd.len());

    let ef = bench_config.ef.clone();

    let index = load_or_build(bench_config, base_set)?;

    let mut bench_result = BenchResult::new();

    for ef in (ef.start..=ef.end).step_by(ef.step) {
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
    let title = format!("HNSW Bench ({} elements)", base_size);
    bench_result.plot(title, index.algorithm_name(), args.output.as_ref())?;
    Ok(())
}
// cargo r -r --example bench -- config/bench_hnsw.toml
