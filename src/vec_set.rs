use std::{
    io::Read,
    mem,
    ops::{Index, Range},
    path::Path,
};

use anyhow::Result;

use crate::config::{DataType, DistanceAlgorithm, VecDataConfig};

pub type Vector<ScalarType> = [ScalarType];

/// Trait for loading data from a binary file.
/// Occupies constant space, apart from the data itself.
pub trait BinaryScalar: Sized {
    fn file_size_limit(file_path: impl AsRef<Path>, limit: Option<usize>) -> Result<usize> {
        let file_size = std::fs::metadata(file_path)?.len() as usize;
        let file_limit = file_size / mem::size_of::<Self>();
        Ok(limit.unwrap_or(usize::MAX).min(file_limit))
    }
    /// Load data from a binary file.
    /// The layout of the binary file is assumed to be a sequence of scalar values.
    /// The number of scalar values to be loaded is limited by `limit`.
    fn from_binary_file(file_path: impl AsRef<Path>, limit: Option<usize>) -> Result<Box<[Self]>>;

    fn to_binary_file(data: &[Self], file_path: impl AsRef<Path>) -> Result<()> {
        let mut file = std::fs::File::create(file_path)?;
        std::io::Write::write_all(&mut file, unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                mem::size_of::<Self>() * data.len(),
            )
        })?;
        Ok(())
    }
}

impl BinaryScalar for u8 {
    fn from_binary_file(file_path: impl AsRef<Path>, limit: Option<usize>) -> Result<Box<[Self]>> {
        let limit = Self::file_size_limit(&file_path, limit)?;
        let mut buffer = vec![0; limit].into_boxed_slice();
        let mut file = std::fs::File::open(file_path)?;
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }
}

impl BinaryScalar for f32 {
    fn from_binary_file(file_path: impl AsRef<Path>, limit: Option<usize>) -> Result<Box<[Self]>> {
        let limit = Self::file_size_limit(&file_path, limit)?;
        let mut buffer = vec![0.0; limit].into_boxed_slice();
        let mut file = std::fs::File::open(file_path)?;
        file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut u8,
                mem::size_of::<Self>() * buffer.len(),
            )
        })?;
        Ok(buffer)
    }
}

pub trait Distance {
    fn l2_distance(&self, other: &Self) -> f32;
    fn euclidean_distance(&self, other: &Self) -> f32 {
        self.l2_distance(other).sqrt()
    }
    fn distance(&self, other: &Self, algorithm: DistanceAlgorithm) -> f32 {
        use DistanceAlgorithm::*;
        match algorithm {
            L2 => self.l2_distance(other),
            Euclidean => self.euclidean_distance(other),
        }
    }
}
impl Distance for Vector<f32> {
    fn l2_distance(&self, other: &Self) -> f32 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
    }
}
impl Distance for Vector<u8> {
    fn l2_distance(&self, other: &Self) -> f32 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).pow(2))
            .sum::<i32>() as f32
    }
}
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
        let data = T::from_binary_file(file_path, size.map(|size| size * dim))?;
        Ok(Self::new(dim, data))
    }

    /// Serialize the `VecSet` to a binary file.
    pub fn into_bin_file(&self, file_path: impl AsRef<Path>) -> Result<()> {
        T::to_binary_file(&self.data, file_path)
    }
}

pub enum TypedVecRef<'a> {
    Float32(&'a [f32]),
    UInt8(&'a [u8]),
}
impl Distance for TypedVecRef<'_> {
    fn l2_distance(&self, other: &Self) -> f32 {
        match (self, other) {
            (Self::Float32(a), Self::Float32(b)) => a.l2_distance(b),
            (Self::UInt8(a), Self::UInt8(b)) => a.l2_distance(b),
            _ => panic!("Cannot calculate distance between different types of vectors."),
        }
    }
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

    pub fn into_bin_file(self, file_path: impl AsRef<Path>) -> Result<()> {
        let path = file_path.as_ref();
        match self {
            Self::Float32(vec_set) => vec_set.into_bin_file(path),
            Self::UInt8(vec_set) => vec_set.into_bin_file(path),
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
            data_path: "data/example/test_vec_set.bin".to_string(),
            size: None,
        };
        let path = PathBuf::from(&config.data_path);
        let dir = path.parent().ok_or_else(|| anyhow!("Invalid path."))?;
        create_dir_all(dir)?;

        let vec_set = VecSet::new(2, vec![0.0, 1.0, 2.0, 3.0].into_boxed_slice());
        let vec_set = Float32(vec_set);
        let cloned_vec_set = vec_set.clone();
        vec_set.into_bin_file(&path)?;
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
