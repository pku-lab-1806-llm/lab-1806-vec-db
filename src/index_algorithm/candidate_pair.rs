use std::collections::BTreeSet;

use anyhow::Result;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

/// A pair of the index and the distance.
/// For the response of the k-nearest neighbors search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidatePair {
    /// The distance to the query vector.
    ///
    /// The distance is wrapped in `OrderedFloat` to implement `Ord`.
    pub distance: OrderedFloat<f32>,
    /// The index of the vector in the index.
    pub index: usize,
}
impl CandidatePair {
    /// Get the distance. (unwrapped)
    pub fn distance(&self) -> f32 {
        self.distance.into_inner()
    }
    /// Create a new candidate pair.
    pub fn new(index: usize, distance: f32) -> Self {
        Self {
            index,
            distance: distance.into(),
        }
    }
}
impl PartialOrd for CandidatePair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CandidatePair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.distance, self.index).cmp(&(other.distance, other.index))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultSet {
    pub k: usize,
    pub results: BTreeSet<CandidatePair>,
}
impl ResultSet {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            results: BTreeSet::new(),
        }
    }
    /// Check if a candidate with distance `d` is worth searching its neighbors.
    pub fn check_candidate(&self, pair: &CandidatePair) -> bool {
        self.results.len() < self.k || pair < self.results.iter().last().unwrap()
    }
    /// Add a candidate pair to the result set.
    ///
    /// Returns true if the pair is added.
    pub fn add(&mut self, pair: CandidatePair) -> bool {
        if self.results.len() < self.k {
            self.results.insert(pair);
            return true;
        }
        if let Some(last) = self.results.iter().last() {
            if pair.distance < last.distance {
                self.results.pop_last();
                self.results.insert(pair);
                return true;
            }
        }
        false
    }
    /// Convert the result set into a sorted vector.
    pub fn into_sorted_vec(self) -> Vec<CandidatePair> {
        self.results.into_iter().collect()
    }
    /// Convert the result set into a sorted vector with a limit.
    pub fn into_sorted_vec_limit(self, limit: usize) -> Vec<CandidatePair> {
        self.results.into_iter().take(limit).collect()
    }

    /// Pick at most m candidates as new neighbors heuristically.
    pub fn heuristic(self, m: usize, dist_fn: impl Fn(usize, usize) -> f32) -> Vec<CandidatePair> {
        let mut neighbors: Vec<CandidatePair> = Vec::with_capacity(m);

        for pair in self.results {
            if neighbors.len() >= m {
                break;
            }
            let d = pair.distance();
            let v = pair.index;
            if neighbors.iter().all(|p| dist_fn(v, p.index) >= d) {
                neighbors.push(pair);
            }
        }
        neighbors
    }

    /// Call this after PQ search to resort the result set.
    pub fn pq_resort(self, k: usize, dist_fn: impl Fn(usize) -> f32) -> Vec<CandidatePair> {
        let mut result = Self::new(k);
        for p in self.results {
            result.add(CandidatePair::new(p.index, dist_fn(p.index)));
        }
        result.into_sorted_vec()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthRow {
    /// The index of the k-nearest neighbor of the query vector.
    ///
    /// Default size: 10.
    pub knn_indices: Vec<usize>,
}
impl GroundTruthRow {
    /// Create a new ground truth row.
    pub fn new(knn_indices: Vec<usize>) -> Self {
        Self { knn_indices }
    }

    /// Recall of the result set.
    ///
    /// Correctly recalled / Total number of ground truth.
    pub fn recall(&self, result: &Vec<CandidatePair>) -> f32 {
        let pred = result
            .iter()
            .map(|pair| pair.index)
            .collect::<BTreeSet<_>>();

        let recalled = self
            .knn_indices
            .iter()
            .filter(|idx| pred.contains(idx))
            .count();

        recalled as f32 / self.knn_indices.len() as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    /// Rows for each vector in the test set.
    ///
    /// Default size: 1000.
    pub rows: Vec<GroundTruthRow>,
}

impl Default for GroundTruth {
    fn default() -> Self {
        Self::new()
    }
}

impl GroundTruth {
    /// Create a new ground truth.
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn iter(&self) -> std::slice::Iter<GroundTruthRow> {
        self.rows.iter()
    }
    /// Push a row to the ground truth.
    pub fn push(&mut self, result: Vec<CandidatePair>) {
        let indices = result.iter().map(|pair| pair.index).collect();
        self.rows.push(GroundTruthRow::new(indices));
    }
    /// Save the ground truth to a binary file.
    pub fn save(&self, path: &str) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load the ground truth from a binary file.
    pub fn load(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let ground_truth = bincode::deserialize_from(reader)?;
        Ok(ground_truth)
    }
}

impl std::ops::Index<usize> for GroundTruth {
    type Output = GroundTruthRow;

    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}
