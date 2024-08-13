use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWConfig {
    pub num_elements: usize,
    pub max_elements: usize,
    pub ef_construction: usize,
    pub M: usize,
}

/// TODO: Implement IVFConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFConfig {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VecDBAlgorithm {
    Linear,
    /// HNSW (Hierarchical Navigable Small World)
    HNSW(HNSWConfig),
    IVF(IVFConfig),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceAlgorithm {
    L2,
    Cosine,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    // 32-bit floating point number `f32`
    #[serde(rename = "float32")]
    Float32,
    // unsigned 8-bit integer `u8`
    #[serde(rename = "uint8")]
    UInt8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecDataConfig {
    pub dim: usize,
    pub data_type: DataType,
    pub data_path: String,
    /// Number of vectors to read from the data file.
    /// If `None`, read all vectors;
    /// If `Some(n)`, read the first `min(n, total_vectors)` vectors.
    pub size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBConfig {
    pub algorithm: VecDBAlgorithm,
    pub distance: DistanceAlgorithm,
    pub vec_data: VecDataConfig,
}

impl DBConfig {
    pub fn load_from_toml_file(file_path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(file_path)?;
        Ok(toml::from_str(&content)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_from_toml_file() {
        let config = DBConfig::load_from_toml_file("config/example/db_config.toml").unwrap();
        println!("{:#?}", config);
    }
}
