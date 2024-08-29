use std::{ops::Index, path::Path};

use crate::scalar::{BinaryScalar, Scalar};

use anyhow::{bail, Result};

use crate::config::{DataType, VecDataConfig};

/// The vector set with scalar type T.
/// Can be indexed to get the vector at the specified index.
/// Load and save the vector set from/to a binary file with constant extra memory.
#[derive(Debug, Clone)]
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
    pub fn new(dim: usize, data: Vec<T>) -> Self {
        assert!(
            data.len() % dim == 0,
            "Data length must be a multiple of the dimension."
        );
        Self { dim, data }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn zeros(dim: usize, len: usize) -> Self
    where
        T: Default + Clone,
    {
        Self::new(dim, vec![T::default(); dim * len])
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn put(&mut self, index: usize, vector: &[T]) {
        assert_eq!(vector.len(), self.dim);
        self.get_mut(index).clone_from_slice(vector);
    }

    pub fn get_mut(&mut self, index: usize) -> &mut [T] {
        let start = index * self.dim;
        let end = start + self.dim;
        &mut self.data[start..end]
    }

    pub fn iter(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks_exact(self.dim)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_exact_mut(self.dim)
    }

    pub fn push(&mut self, vector: &[T]) {
        assert_eq!(vector.len(), self.dim);
        self.data.extend_from_slice(vector);
    }

    pub fn pop_last(&mut self) -> Option<Vec<T>> {
        if self.data.len() >= self.dim {
            let start = self.data.len() - self.dim;
            let end = self.data.len();
            Some(self.data.drain(start..end).collect())
        } else {
            None
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
}

impl<T: BinaryScalar> VecSet<T> {
    /// Deserialize a `VecSet` from a binary file.
    pub fn load_file(dim: usize, size: Option<usize>, file_path: impl AsRef<Path>) -> Result<Self> {
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
    pub fn save_file(&self, file_path: impl AsRef<Path>) -> Result<()> {
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
///     config::DBConfig,
///     vec_set::{DynamicVecSet, VecSet},
///     scalar::Scalar,
/// };
///
/// let file_path = "config/example/db_config.toml";
/// let config = DBConfig::load_from_toml_file(file_path).unwrap();
/// let vec_set = DynamicVecSet::load_with(&config.vec_data).unwrap();
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
#[derive(Debug, Clone)]
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
            Float32 => Self::Float32(VecSet::load_file(dim, size, &file)?),
            UInt8 => Self::UInt8(VecSet::load_file(dim, size, &file)?),
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
    use std::{fs::create_dir_all, path::PathBuf};

    use anyhow::anyhow;

    use crate::config::DBConfig;

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
        let file_path = "config/example/db_config.toml";
        let config = DBConfig::load_from_toml_file(file_path).unwrap();
        println!("Loaded config: {:#?}", config);
        VecSet::<u8>::load_with(&config.vec_data).unwrap();
    }

    #[test]
    fn load_vec_set_test() -> Result<()> {
        // See also the `match` usage in the doc test of `DynamicVecSet`.

        let file_path = "config/example/db_config.toml";
        let config = DBConfig::load_from_toml_file(file_path)?;
        println!("Loaded config: {:#?}", config);
        let vec_set = VecSet::<f32>::load_with(&config.vec_data)?;

        let v0 = &vec_set[0];
        let v1 = &vec_set[1];
        println!("Distance Algorithm: {:?}", config.distance);
        let distance = config.distance.d(v0, v1);
        println!("Distance: {}", distance);
        assert!((distance - 2.3230).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn save_vec_set_test() -> Result<()> {
        let file_path = PathBuf::from("data/example/test_vec_set.test.bin");
        let dir = file_path.parent().ok_or_else(|| anyhow!("Invalid path."))?;
        create_dir_all(dir)?;

        // Save a DynamicVecSet to a binary file.
        let vec_set = VecSet::new(2, vec![0.0, 1.0, 2.0, 3.0]);
        vec_set.save_file(&file_path)?;

        // Load a VecSet<f32> from the binary file.
        let loaded_vec_set = VecSet::<f32>::load_file(2, None, &file_path)?;

        assert_eq!(loaded_vec_set.data, vec_set.data);
        Ok(())
    }
}
