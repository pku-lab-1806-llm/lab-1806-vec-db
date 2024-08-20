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
    pub fn from_binary_file(dim: usize, size: Option<usize>, file_path: &str) -> Result<Self> {
        let data = T::from_binary_file(file_path, size.map(|size| size * dim)).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load the binary file at {}: {}",
                file_path,
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

pub enum TypedVecRef<'a> {
    Float32(&'a [f32]),
    UInt8(&'a [u8]),
}
#[derive(Debug, Clone)]
pub enum TypedVecSet {
    Float32(VecSet<f32>),
    UInt8(VecSet<u8>),
}
impl TypedVecSet {
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
    pub fn index(&self, index: usize) -> TypedVecRef {
        match self {
            Self::Float32(vec_set) => TypedVecRef::Float32(&vec_set[index]),
            Self::UInt8(vec_set) => TypedVecRef::UInt8(&vec_set[index]),
        }
    }
    /// Alias for `index`.
    pub fn i(&self, index: usize) -> TypedVecRef {
        self.index(index)
    }
}
impl From<VecSet<f32>> for TypedVecSet {
    fn from(vec_set: VecSet<f32>) -> Self {
        Self::Float32(vec_set)
    }
}
impl From<VecSet<u8>> for TypedVecSet {
    fn from(vec_set: VecSet<u8>) -> Self {
        Self::UInt8(vec_set)
    }
}

#[cfg(test)]
mod test {
    use std::{fs::create_dir_all, path::PathBuf};

    use anyhow::{anyhow, bail};

    use super::*;

    #[test]
    fn test_vec_set() {
        let vec_set = VecSet::new(3, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
        assert_eq!(vec_set.len(), 2);
        assert_eq!(vec_set[0], [1, 2, 3]);
        assert_eq!(vec_set[1], [4, 5, 6]);
    }

    #[test]
    fn save_and_load_vec_set() -> Result<()> {
        use TypedVecSet::*;
        let config = VecDataConfig {
            dim: 2,
            data_type: DataType::Float32,
            data_path: "data/example/test_vec_set.test.bin".to_string(),
            size: None,
        };
        let path = PathBuf::from(&config.data_path);
        let dir = path.parent().ok_or_else(|| anyhow!("Invalid path."))?;
        create_dir_all(dir)?;

        let vec_set = VecSet::new(2, vec![0.0, 1.0, 2.0, 3.0].into_boxed_slice());
        let vec_set = Float32(vec_set);
        let cloned_vec_set = vec_set.clone();
        vec_set.save_binary_file(&path)?;
        let loaded_vec_set = TypedVecSet::load_with(config)?;
        let loaded_vec_set = if let Float32(v) = loaded_vec_set {
            v
        } else {
            bail!("Failed to load the vector set.");
        };
        let cloned_vec_set = if let Float32(v) = cloned_vec_set {
            v
        } else {
            bail!("Failed to access the cloned vector set.");
        };
        assert_eq!(loaded_vec_set.data, cloned_vec_set.data);
        dbg!(loaded_vec_set.data, cloned_vec_set.data);
        dbg!(1.0_f32.to_ne_bytes().map(|v| format!("{:02x}", v)));
        Ok(())
    }
}
