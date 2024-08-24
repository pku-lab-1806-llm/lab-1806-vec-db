use std::{ops::Index, path::Path};

use crate::binary_scalar::BinaryScalar;

use anyhow::{bail, Result};

use crate::config::{DataType, VecDataConfig};

/// The vector set with scalar type T.
/// Can be indexed to get the vector at the specified index.
/// Load and save the vector set from/to a binary file with constant extra memory.
#[derive(Debug, Clone)]
pub struct VecSet<T> {
    dim: usize,
    data: Box<[T]>,
}

impl<T> Index<usize> for VecSet<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.dim;
        let end = start + self.dim;
        &self.data[start..end]
    }
}

impl<T: BinaryScalar> VecSet<T> {
    pub fn new(dim: usize, data: Box<[T]>) -> Self {
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
        Self::new(dim, vec![T::default(); dim * len].into_boxed_slice())
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

    /// Convert the `VecSet` to a `VecSet` with a different scalar type.
    ///
    /// The conversion is done by casting the scalar values to `f32` and then to the target type `U`.
    pub fn to_type<U: BinaryScalar>(&self) -> VecSet<U> {
        let data = self
            .data
            .iter()
            .map(|&x| U::cast_from_f32(x.cast_to_f32()))
            .collect::<Vec<U>>()
            .into_boxed_slice();
        VecSet::new(self.dim, data)
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

/// The reference to the vector with scalar type f32 or u8.
/// The data type is determined at runtime.
#[derive(Debug, Clone)]
pub enum DynamicVecRef<'a> {
    Float32(&'a [f32]),
    UInt8(&'a [u8]),
}
impl<'a> From<&'a [f32]> for DynamicVecRef<'a> {
    fn from(slice: &'a [f32]) -> Self {
        Self::Float32(slice)
    }
}

impl<'a> From<&'a [u8]> for DynamicVecRef<'a> {
    fn from(slice: &'a [u8]) -> Self {
        Self::UInt8(slice)
    }
}

impl<'a> TryFrom<DynamicVecRef<'a>> for &'a [f32] {
    type Error = anyhow::Error;

    fn try_from(value: DynamicVecRef<'a>) -> Result<Self> {
        match value {
            DynamicVecRef::Float32(slice) => Ok(slice),
            _ => bail!("Failed to convert to &[f32]."),
        }
    }
}

impl<'a> TryFrom<DynamicVecRef<'a>> for &'a [u8] {
    type Error = anyhow::Error;

    fn try_from(value: DynamicVecRef<'a>) -> Result<Self> {
        match value {
            DynamicVecRef::UInt8(slice) => Ok(slice),
            _ => bail!("Failed to convert to &[u8]."),
        }
    }
}

/// The dynamic vector set with scalar type f32 or u8.
/// Can be indexed to get the vector at the specified index.
/// Load and save the vector set from/to a binary file with constant extra memory.
/// The data type is determined at runtime.
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
        let size = config.limit;
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

    pub fn iter(&self) -> impl Iterator<Item = DynamicVecRef> {
        (0..self.len()).map(move |i| self.index(i))
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
        let distance = config.distance.d(&v0, &v1);
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
