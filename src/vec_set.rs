use std::{ops::Index, path::Path};

use crate::scalar::Scalar;

use anyhow::{bail, Result};
use rand::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};

use crate::config::{DataType, VecDataConfig};

/// The vector set with scalar type T.
/// Can be indexed to get the vector at the specified index.
/// Load and save the vector set from/to a binary file with constant extra memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecSet<T> {
    /// The dimension of the vector.
    dim: usize,
    /// The data of the vectors. Size&Cap: dim * len.
    data: Vec<T>,
}

impl<T> Index<usize> for VecSet<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.dim;
        let end = start + self.dim;
        &self.data[start..end]
    }
}

impl<T: Scalar> VecSet<T> {
    /// Create a `VecSet` with the specified dimension and data.
    pub fn new(dim: usize, data: Vec<T>) -> Self {
        assert!(
            data.len() % dim == 0,
            "Data length must be a multiple of the dimension."
        );
        Self { dim, data }
    }

    /// Get the dimension of the vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Create a `VecSet` with the specified capacity.
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            dim,
            data: Vec::with_capacity(dim * capacity),
        }
    }
    /// Reserve additional capacity for the `VecSet`.
    ///
    /// May reserve more space to avoid frequent reallocations.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.dim);
    }
    /// Reserve EXACT additional capacity for the `VecSet`.
    ///
    /// May cause more reallocations.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.data.reserve_exact(additional * self.dim);
    }
    /// Get the capacity of the `VecSet`.
    pub fn capacity(&self) -> usize {
        self.data.capacity() / self.dim
    }
    /// Shrink the capacity of the `VecSet` to fit the data.
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Get the number of vectors in the `VecSet`.
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    /// Set the vector at the specified index.
    pub fn put(&mut self, index: usize, vector: &[T]) {
        assert_eq!(vector.len(), self.dim);
        self.get_mut(index).clone_from_slice(vector);
    }

    /// Get a mutable reference to the vector at the specified index.
    pub fn get_mut(&mut self, index: usize) -> &mut [T] {
        let start = index * self.dim;
        let end = start + self.dim;
        &mut self.data[start..end]
    }

    /// Get an iterator of the vectors.
    pub fn iter(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks_exact(self.dim)
    }

    /// Get a mutable iterator of the vectors.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_exact_mut(self.dim)
    }

    /// Push a vector to the `VecSet`.
    ///
    /// Make sure you have created the `VecSet` with `with_capacity`.
    ///
    /// Or use `reserve` or `reserve_exact` to reserve additional capacity.
    pub fn push(&mut self, vector: &[T]) -> usize {
        let index = self.len();
        assert_eq!(vector.len(), self.dim);
        self.data.extend_from_slice(vector);
        index
    }

    /// Pop the last vector from the `VecSet`.
    pub fn pop_last(&mut self) -> Option<Vec<T>> {
        if self.data.len() >= self.dim {
            let start = self.data.len() - self.dim;
            let end = self.data.len();
            Some(self.data.drain(start..end).collect())
        } else {
            None
        }
    }

    pub fn swap_remove(&mut self, index: usize) {
        assert!(index < self.len());
        let last = self.pop_last().unwrap();
        if index < self.len() {
            self.put(index, &last);
        }
    }

    /// Convert the `VecSet` to a `VecSet` with a different scalar type.
    ///
    /// The conversion is done by casting the scalar values to `f32` and then to the target type `U`.
    pub fn to_type<U: Scalar>(&self) -> VecSet<U> {
        let data = self
            .data
            .iter()
            .map(|&x| U::cast_from_f32(x.cast_to_f32()))
            .collect::<Vec<U>>();
        VecSet::new(self.dim, data)
    }

    /// Select `k` vectors randomly from the `VecSet`.
    ///
    /// Usually used for the initialization of the centroids in K-means.
    pub fn random_sample(&self, k: usize, rng: &mut impl Rng) -> VecSet<T> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.shuffle(rng);
        let indices = &indices[..k];
        let data = indices
            .iter()
            .flat_map(|&i| self[i].iter().copied())
            .collect::<Vec<_>>();
        VecSet::new(self.dim, data)
    }
}

impl<T: Scalar> VecSet<T> {
    /// Deserialize a `VecSet` from a binary file.
    pub fn load_raw_file(
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
    pub fn save_raw_file(&self, file_path: impl AsRef<Path>) -> Result<()> {
        T::to_binary_file(&self.data, &file_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to save the binary file at {}: {}",
                file_path.as_ref().display(),
                e.to_string()
            )
        })
    }
}
impl VecSet<f32> {
    pub fn load_with(config: &VecDataConfig) -> Result<Self> {
        DynamicVecSet::load_with(config)?.try_into()
    }
}
impl VecSet<u8> {
    pub fn load_with(config: &VecDataConfig) -> Result<Self> {
        DynamicVecSet::load_with(config)?.try_into()
    }
}

/// See also `VecSet<T>` for the implementation of the methods.
///
/// This enum is used to load the vector set from a binary file,
/// where the data type is determined at runtime.
///
/// ```rust
/// use anyhow::Result;
/// use lab_1806_vec_db::{
///     config::VecDataConfig,
///     vec_set::{DynamicVecSet, VecSet},
///     scalar::Scalar,
/// };
///
/// let file_path = "config/gist_1000.toml";
/// let config = VecDataConfig::load_from_toml_file(file_path).unwrap();
/// let vec_set = DynamicVecSet::load_with(&config).unwrap();
/// // Or directly determine the data type at compile time.
/// // let vec_set = VecSet::<f32>::load_with(&config.vec_data).unwrap();
///
/// // Write your code with Generic Scalar type.
/// fn foo<T: Scalar>(vec_set: &VecSet<T>) {
///     println!("{:?}", &vec_set[0]);
/// }
///
/// // Determine the data type at runtime.
/// use DynamicVecSet::*;
/// match &vec_set {
///     Float32(vec_set) => foo(vec_set),
///     UInt8(vec_set) => foo(vec_set),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DynamicVecSet {
    Float32(VecSet<f32>),
    UInt8(VecSet<u8>),
}
impl DynamicVecSet {
    pub fn load_with(config: &VecDataConfig) -> Result<Self> {
        let dim = config.dim;
        let size = config.limit;
        let file = &config.data_path;
        use DataType::*;
        let vec_set = match config.data_type {
            Float32 => Self::Float32(VecSet::load_raw_file(dim, size, file)?),
            UInt8 => Self::UInt8(VecSet::load_raw_file(dim, size, file)?),
        };
        Ok(vec_set)
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
            _ => bail!("Failed to convert to VecSet<f32>."),
        }
    }
}
impl TryFrom<DynamicVecSet> for VecSet<u8> {
    type Error = anyhow::Error;

    fn try_from(value: DynamicVecSet) -> Result<Self> {
        match value {
            DynamicVecSet::UInt8(vec_set) => Ok(vec_set),
            _ => bail!("Failed to convert to VecSet<u8>."),
        }
    }
}

#[cfg(test)]
mod test {
    use std::{
        fs::{self, create_dir_all},
        path::PathBuf,
    };

    use anyhow::anyhow;

    use super::*;

    #[test]
    fn test_vec_set() {
        let vec_set = VecSet::new(3, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(vec_set.len(), 2);
        assert_eq!(vec_set[0], [1, 2, 3]);
        assert_eq!(vec_set[1], [4, 5, 6]);
    }

    #[test]
    #[should_panic(expected = "Failed to convert to VecSet<u8>.")]
    fn data_type_mismatched_test() {
        let file_path = "config/gist_1000.toml";
        let config = VecDataConfig::load_from_toml_file(file_path).unwrap();
        println!("Loaded config: {:#?}", config);
        VecSet::<u8>::load_with(&config).unwrap();
    }

    #[test]
    fn load_vec_set_test() -> Result<()> {
        // See also the `match` usage in the doc test of `DynamicVecSet`.

        let file_path = "config/gist_1000.toml";
        let config = VecDataConfig::load_from_toml_file(file_path).unwrap();
        println!("Loaded config: {:#?}", config);
        let vec_set = VecSet::<f32>::load_with(&config)?;

        println!("len: {}", vec_set.len());

        Ok(())
    }

    #[test]
    fn save_vec_set_test() -> Result<()> {
        let file_path = PathBuf::from("data/test_vec_set.tmp.bin");
        let dir = file_path.parent().ok_or_else(|| anyhow!("Invalid path."))?;
        create_dir_all(dir)?;

        // Save a DynamicVecSet to a binary file.
        let vec_set = VecSet::new(2, vec![0.0, 1.0, 2.0, 3.0]);
        vec_set.save_raw_file(&file_path)?;

        // Load a VecSet<f32> from the binary file.
        let loaded_vec_set = VecSet::<f32>::load_raw_file(2, None, &file_path)?;

        assert_eq!(loaded_vec_set.data, vec_set.data);
        fs::remove_file(&file_path)?;
        Ok(())
    }
}
