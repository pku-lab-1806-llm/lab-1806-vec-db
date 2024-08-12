use std::ops::{Index, Range};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::{DataType, DistanceAlgorithm, VecDataConfig};

pub type Vector<ScalarType> = [ScalarType];

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

    /// Create a `VecSet` from the data.
    pub fn from_data(dim: usize, size: Option<usize>, mut data: Vec<T>) -> Self {
        if let Some(size) = size {
            data.truncate(size * dim);
        }
        Self::new(dim, data.into_boxed_slice())
    }

    pub fn into_data(self) -> Vec<T> {
        self.data.into_vec()
    }
}
impl<'a, T: Deserialize<'a>> VecSet<T> {
    /// Deserialize a `VecSet` from the first `size` vectors in the binary file.
    /// Fewer than `size` vectors will be used if the data has fewer than `size` vectors.
    /// The binary file is assumed to be serialized by `bincode`.
    pub fn try_from_bincode(dim: usize, size: Option<usize>, data: &'a [u8]) -> Result<Self> {
        let data = bincode::deserialize::<Vec<T>>(data)?;
        Ok(Self::from_data(dim, size, data))
    }
}
impl<T: Serialize> VecSet<T> {
    /// Serialize the `VecSet` to a binary file.
    /// The binary file is serialized by `bincode`.
    pub fn try_into_bincode(self) -> Result<Vec<u8>> {
        bincode::serialize(&self.into_data()).map_err(Into::into)
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
        let data = std::fs::read(config.data_path)?;
        let vec_set = match config.data_type {
            DataType::Float32 => Self::Float32(VecSet::try_from_bincode(dim, size, &data)?),
            DataType::UInt8 => Self::UInt8(VecSet::try_from_bincode(dim, size, &data)?),
        };
        Ok(vec_set)
    }
    pub fn try_into_bincode(self) -> Result<Vec<u8>> {
        match self {
            Self::Float32(vec_set) => vec_set.try_into_bincode(),
            Self::UInt8(vec_set) => vec_set.try_into_bincode(),
        }
    }

    pub fn into_bincode_file(self, file_path: &str) -> Result<()> {
        let data = self.try_into_bincode()?;
        std::fs::write(file_path, data)?;
        Ok(())
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
        let path = "data/example/test_vec_set.bin";
        let vec_set = VecSet::new(3, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
        let vec_set = TypedVecSet::UInt8(vec_set);
        let cloned_vec_set = vec_set.clone();
        vec_set.into_bincode_file(path)?;
        let loaded_vec_set = TypedVecSet::load_with(VecDataConfig {
            dim: 3,
            data_type: DataType::UInt8,
            data_path: path.to_string(),
            size: None,
        })?;
        assert_eq!(
            cloned_vec_set.try_into_bincode()?,
            loaded_vec_set.try_into_bincode()?
        );
        Ok(())
    }
}
