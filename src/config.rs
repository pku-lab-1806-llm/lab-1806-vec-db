use std::path::Path;

use crate::index_algorithm::{
    hnsw_index::HNSWConfig, ivf_index::IVFConfig, pq_linear_index::PQLinearConfig,
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
    PQLinear(PQLinearConfig),
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
impl VecDataConfig {
    pub fn load_from_toml_file(file_path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(file_path.as_ref()).map_err(|e| {
            anyhow::anyhow!(
                "Failed to read the TOML file at {}: {}",
                file_path.as_ref().display(),
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
        let config = VecDataConfig::load_from_toml_file("config/gist_1000.toml").unwrap();
        println!("{:#?}", config);
    }
}
