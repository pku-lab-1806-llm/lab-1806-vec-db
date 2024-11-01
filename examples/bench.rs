use std::{
    io::Write,
    ops::AddAssign,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
};

use anyhow::Result;
use clap::Parser;
use lab_1806_vec_db::{
    config::{IndexAlgorithmConfig, VecDataConfig},
    distance::{
        pq_table::{PQConfig, PQTable},
        DistanceAlgorithm,
    },
    index_algorithm::{
        candidate_pair::GroundTruth, CandidatePair, HNSWIndex, IVFIndex, LinearIndex,
    },
    prelude::*,
    scalar::Scalar,
    vec_set::VecSet,
};
use plotly::{Plot, Scatter, Trace};
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
impl BenchEf {
    pub fn to_vec(&self) -> Vec<usize> {
        match self {
            BenchEf::Range(range) => (range.start..=range.end).step_by(range.step).collect(),
            BenchEf::List(list) => list.clone(),
        }
    }
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
    /// Plot the result only
    #[clap(short, long)]
    plot_only: bool,
    /// Save the plot to a html file
    #[clap(long)]
    html: Option<String>,
    /// Repeat times for more stable results
    #[clap(short, long, default_value = "1")]
    repeat_times: usize,
    /// Number of threads, recommended to be the number of physical cores.
    ///
    /// Generally, multi-threading is disabled for benchmarking,
    /// but it is common in real applications, so you can try it here.
    #[clap(short, long)]
    num_threads: Option<usize>,
}

struct AvgRecorder {
    sum: f32,
    count: usize,
}
impl AvgRecorder {
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
    pub fn add(&mut self, value: f32) {
        self.sum += value;
        self.count += 1;
    }
    pub fn avg(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f32
    }
}
impl AddAssign for AvgRecorder {
    fn add_assign(&mut self, rhs: Self) {
        self.sum += rhs.sum;
        self.count += rhs.count;
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchResult {
    label: String,
    /// The search radius
    ef: Vec<usize>,
    /// Average search time in ms
    search_time: Vec<f32>,
    /// Recall (k = 10)
    recall: Vec<f32>,
}
impl BenchResult {
    pub fn new(label: String) -> Self {
        Self {
            label,
            ef: Vec::new(),
            search_time: Vec::new(),
            recall: Vec::new(),
        }
    }
    pub fn trace(self) -> Box<dyn Trace> {
        let text = self
            .ef
            .iter()
            .map(|&ef| format!("ef={}", ef))
            .collect::<Vec<_>>();
        Scatter::new(self.recall, self.search_time)
            .text_array(text)
            .text_font(plotly::common::Font::new().size(10).family("Arial"))
            .text_position(plotly::common::Position::TopLeft)
            .mode(plotly::common::Mode::LinesMarkersText)
            .name(self.label)
    }
    pub fn push(&mut self, ef: usize, search_time: f32, recall: f32) {
        self.ef.push(ef);
        self.search_time.push(search_time);
        self.recall.push(recall);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResultList {
    #[serde(default)]
    title: String,
    #[serde(default)]
    results: Vec<BenchResult>,
}
impl ResultList {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        println!("Loading results from {}...", path.as_ref().display());
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let results: ResultList = toml::from_str(&content)?;
        println!("Loaded {} results.", results.results.len());
        Ok(results)
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write(content.as_bytes())?;
        Ok(())
    }
    pub fn plot(self, multi_thread: bool, html_path: Option<impl AsRef<Path>>) -> Result<()> {
        let mut plot = Plot::new();
        for result in self.results {
            plot.add_trace(result.trace());
        }

        let layout = plotly::Layout::new()
            .title(plotly::common::Title::with_text(self.title))
            .x_axis(plotly::layout::Axis::new().title("Recall"))
            .y_axis(plotly::layout::Axis::new().title(if multi_thread {
                "Inverse Throughput(ms)"
            } else {
                "Search Time(ms)"
            }))
            .show_legend(true);
        plot.set_layout(layout);

        if let Some(path) = html_path {
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
    pub fn push(&mut self, result: BenchResult) {
        for r in self.results.iter_mut() {
            if r.label == result.label {
                *r = result;
                return;
            }
        }
        self.results.push(result);
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let load_start = std::time::Instant::now();
    if args.plot_only {
        let result_list = ResultList::load(&args.bench_config_path)?;
        result_list.plot(args.num_threads.is_some(), args.html.as_ref())?;
        return Ok(());
    }
    let bench_config = BenchConfig::load_from_toml_file(&args.bench_config_path)?;
    let label = bench_config.label.clone();
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
    let bench_output = bench_config.bench_output.clone();

    let pq = load_or_build_pq(&bench_config, &base_set, &mut rng)?;

    let index = load_or_build_index(bench_config, base_set, &mut rng)?;

    let mut bench_result = BenchResult::new(label);

    for ef in ef.to_vec() {
        println!("Benchmarking ef: {}...", ef);
        let mut avg_recall = AvgRecorder::new();

        let start = std::time::Instant::now();
        for _ in 0..args.repeat_times {
            if let Some(num_threads) = args.num_threads {
                let (sender, receiver) = mpsc::channel();
                thread::scope(|s| {
                    for idx in 0..num_threads {
                        let (test_set, gnd, pq, index) = (&test_set, &gnd, &pq, &index);
                        let sender = sender.clone();
                        s.spawn(move || {
                            let mut thread_recorder = AvgRecorder::new();
                            for (query, gnd) in test_set
                                .iter()
                                .zip(gnd.iter())
                                .skip(idx)
                                .step_by(num_threads)
                            {
                                let result_set = index.knn_with_ef(query, k, ef, &pq);

                                let recall = gnd.recall(&result_set);
                                thread_recorder.add(recall);
                            }
                            sender.send(thread_recorder).unwrap();
                        });
                    }
                });
                for thread_recorder in receiver.iter().take(num_threads) {
                    avg_recall.add(thread_recorder.avg());
                }
            } else {
                for (query, gnd) in test_set.iter().zip(gnd.iter()) {
                    // benchmark here
                    let result_set = index.knn_with_ef(query, k, ef, &pq);

                    let recall = gnd.recall(&result_set);
                    avg_recall.add(recall);
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f32();
        // ms
        let search_time = elapsed / (args.repeat_times * test_set.len()) as f32 * 1000.0;
        let recall = avg_recall.avg();

        println!(
            "ef: {}, Average Search Time: {:.2}ms, Average recall: {:.4}",
            ef, search_time, recall
        );
        bench_result.push(ef, search_time, recall);
    }
    println!("Finished benchmarking.");
    let (title, out) = if let Some(num_threads) = args.num_threads {
        let title = format!("Bench ({} elements, {} threads)", base_size, num_threads);
        let out = PathBuf::from(bench_output);
        let out = out.with_file_name(format!(
            "t{}_{}",
            num_threads,
            out.file_name().unwrap_or_default().to_string_lossy(),
        ));
        (title, out)
    } else {
        let title = format!("Bench ({} elements)", base_size);
        let out = PathBuf::from(bench_output);
        (title, out)
    };
    let mut result_list = ResultList::load(&out)?;
    result_list.push(bench_result);
    result_list.title = title;
    result_list.save(&out)?;
    println!("Saved results to {}.", out.display());
    result_list.plot(args.num_threads.is_some(), args.html.as_ref())?;
    Ok(())
}
// # Add `-r 20` or other repeat times to get more stable results.
// cargo r -r --example bench -- config/bench_hnsw.toml
// cargo r -r --example bench -- config/bench_pq_hnsw.toml

// # Since LLVM has auto-vectorization, this is just the same as the normal one.
// cargo r -r --example bench -- config/bench_simd_hnsw.toml
