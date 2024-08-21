use std::{
    ops::{Index, Range},
    path::Path,
};

use crate::binary_scalar::BinaryScalar;

use anyhow::Result;

use crate::config::{DataType, VecDataConfig};

pub type Vector<ScalarType> = [ScalarType];

#[derive(Debug, Clone)]
pub struct VecSet<ScalarType> {
    pub dim: usize,
    pub data: Box<[ScalarType]>,
}

impl<T> Index<usize> for VecSet<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.dim;
        let end = start + self.dim;
        &self.data[start..end]
    }
}
impl<T> Index<Range<usize>> for VecSet<T> {
    type Output = [T];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        let start = range.start * self.dim;
        let end = range.end * self.dim;
        &self.data[start..end]
    }
}

impl<T> VecSet<T> {
    fn new(dim: usize, data: Box<[T]>) -> Self {
        assert!(
            data.len() % dim == 0,
            "Data length must be a multiple of the dimension."
        );
        Self { dim, data }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }
}
impl<T: BinaryScalar> VecSet<T> {
    /// Deserialize a `VecSet` from a binary file.
    pub fn from_binary_file(
        dim: usize,
        size: Option<usize>,
        file_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let data = T::from_binary_file(&file_path, size.map(|size| size * dim)).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load the binary file at {}: {}",
                file_path.as_ref().display(),
                e.to_string()
            )
        })?;
        Ok(Self::new(dim, data))
    }

    /// Serialize the `VecSet` to a binary file.
    pub fn save_binary_file(&self, file_path: impl AsRef<Path>) -> Result<()> {
        T::to_binary_file(&self.data, &file_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to save the binary file at {}: {}",
                file_path.as_ref().display(),
                e.to_string()
            )
        })
    }
}

pub enum DynamicVecRef<'a> {
    Float32(&'a [f32]),
    UInt8(&'a [u8]),
}
#[derive(Debug, Clone)]
pub enum DynamicVecSet {
    Float32(VecSet<f32>),
    UInt8(VecSet<u8>),
}
impl DynamicVecSet {
    pub fn len(&self) -> usize {
        match self {
            Self::Float32(vec_set) => vec_set.len(),
            Self::UInt8(vec_set) => vec_set.len(),
        }
    }
    pub fn load_with(config: VecDataConfig) -> Result<Self> {
        let dim = config.dim;
        let size = config.size;
        let file = config.data_path;
        use DataType::*;
        let vec_set = match config.data_type {
            Float32 => Self::Float32(VecSet::from_binary_file(dim, size, &file)?),
            UInt8 => Self::UInt8(VecSet::from_binary_file(dim, size, &file)?),
        };
        Ok(vec_set)
    }

    pub fn save_binary_file(&self, file_path: impl AsRef<Path>) -> Result<()> {
        let path = file_path.as_ref();
        match self {
            Self::Float32(vec_set) => vec_set.save_binary_file(path),
            Self::UInt8(vec_set) => vec_set.save_binary_file(path),
        }
    }

    /// Get the reference to the vector at the specified index.
    pub fn index(&self, index: usize) -> DynamicVecRef {
        match self {
            Self::Float32(vec_set) => DynamicVecRef::Float32(&vec_set[index]),
            Self::UInt8(vec_set) => DynamicVecRef::UInt8(&vec_set[index]),
        }
    }
    /// Alias for `index`.
    pub fn i(&self, index: usize) -> DynamicVecRef {
        self.index(index)
    }
}
impl From<VecSet<f32>> for DynamicVecSet {
    fn from(vec_set: VecSet<f32>) -> Self {
        Self::Float32(vec_set)
    }
}
impl From<VecSet<u8>> for DynamicVecSet {
    fn from(vec_set: VecSet<u8>) -> Self {
        Self::UInt8(vec_set)
    }
}
impl TryFrom<DynamicVecSet> for VecSet<f32> {
    type Error = anyhow::Error;

    fn try_from(value: DynamicVecSet) -> Result<Self> {
        match value {
            DynamicVecSet::Float32(vec_set) => Ok(vec_set),
            _ => Err(anyhow::anyhow!("Failed to convert to VecSet<f32>.")),
        }
    }
}
impl TryFrom<DynamicVecSet> for VecSet<u8> {
    type Error = anyhow::Error;

    fn try_from(value: DynamicVecSet) -> Result<Self> {
        match value {
            DynamicVecSet::UInt8(vec_set) => Ok(vec_set),
            _ => Err(anyhow::anyhow!("Failed to convert to VecSet<u8>.")),
        }
    }
}

#[cfg(test)]
mod test {
    use std::{fs::create_dir_all, path::PathBuf};

    use anyhow::anyhow;

    use crate::{config::DBConfig, distance::Distance};

    use super::*;

    #[test]
    fn test_vec_set() {
        let vec_set = VecSet::new(3, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
        assert_eq!(vec_set.len(), 2);
        assert_eq!(vec_set[0], [1, 2, 3]);
        assert_eq!(vec_set[1], [4, 5, 6]);
    }

    #[test]
    fn load_vec_set_test() -> Result<()> {
        let file_path = "config/example/db_config.toml";
        let config = DBConfig::load_from_toml_file(file_path)?;
        println!("Loaded config: {:#?}", config);
        let vec_set = DynamicVecSet::load_with(config.vec_data)?;
        let v0 = vec_set.i(0);
        let v1 = vec_set.i(1);
        println!("Distance Algorithm: {:?}", config.distance);
        let distance = v0.distance(&v1, config.distance);
        println!("Distance: {}", distance);
        assert!((distance - 2.3230).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn save_vec_set_test() -> Result<()> {
        let file_path = PathBuf::from("data/example/test_vec_set.test.bin");
        let dir = file_path.parent().ok_or_else(|| anyhow!("Invalid path."))?;
        create_dir_all(dir)?;

        // Save a TypedVecSet to a binary file.
        let vec_set = VecSet::new(2, vec![0.0, 1.0, 2.0, 3.0].into_boxed_slice());
        let vec_set = DynamicVecSet::Float32(vec_set);
        let cloned_vec_set = vec_set.clone();
        vec_set.save_binary_file(&file_path)?;

        // Load a VecSet<f32> from the binary file.
        let loaded_vec_set = VecSet::<f32>::from_binary_file(2, None, &file_path)?;

        let cloned_vec_set: VecSet<f32> = cloned_vec_set.try_into()?;

        assert_eq!(loaded_vec_set.data, cloned_vec_set.data);
        Ok(())
    }
}
