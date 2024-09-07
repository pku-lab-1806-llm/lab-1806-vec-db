use crate::{
    distance::DistanceAlgorithm,
    index_algorithm::{hnsw_index::HNSWConfig, ivf_index::IVFConfig},
};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// The configuration of the index algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexAlgorithmConfig {
    /// Linear search. *Precise but slow.*
    /// No configuration is required.
    Linear,
    /// HNSW (Hierarchical Navigable Small World)
    HNSW(HNSWConfig),
    IVF(IVFConfig),
}

/// Data type of the vector elements.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    // 32-bit floating point number `f32`
    #[serde(rename = "float32")]
    Float32,
    // unsigned 8-bit integer `u8`
    #[serde(rename = "uint8")]
    UInt8,
}

/// The configuration of the vector data file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecDataConfig {
    /// Dimension of each vector.
    pub dim: usize,
    /// Data type of the vector elements.
    pub data_type: DataType,
    /// Path to the data file.
    pub data_path: String,
    /// *Optional:* The maximum number of vectors to be loaded.
    pub limit: Option<usize>,
}

/// The configuration of the vector database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBConfig {
    /// The configuration of the index algorithm.
    pub algorithm: IndexAlgorithmConfig,
    /// The distance algorithm to be used in the vector database.
    pub distance: DistanceAlgorithm,
    /// The configuration of the vector data file.
    pub vec_data: VecDataConfig,
}

impl DBConfig {
    pub fn load_from_toml_file(file_path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(file_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to read the TOML file at {}: {}",
                file_path,
                e.to_string()
            )
        })?;
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
