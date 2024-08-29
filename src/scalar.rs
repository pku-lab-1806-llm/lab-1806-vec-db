use std::{
    fmt::Debug,
    io::Read,
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    path::Path,
};

use anyhow::Result;

/// Trait for scalar types.
/// Scalar types are used in the vector set.
/// Scalar trait contains the basic operations for scalar types.
///
/// Supported scalar types:
/// - `u8`
/// - `f32`
pub trait Scalar:
    Sized
    + Default
    + Copy
    + Debug
    // +, +=, -, -=, *, *=, /, /=
    + Add
    + AddAssign
    + Sub
    + SubAssign
    + Mul
    + MulAssign
    + Div
    + DivAssign
    // Comparison
    + PartialEq
    + PartialOrd
{
    /// Cast a float value to the scalar type. *Alias for `as`.*
    ///
    /// Casting from a float to an integer will round the float towards zero
    /// - NaN will return 0
    /// - Values larger than the maximum integer value, including INFINITY, will saturate to the maximum value of the integer type.
    /// - Values smaller than the minimum integer value, including NEG_INFINITY, will saturate to the minimum value of the integer type.
    fn cast_from_f32(value: f32) -> Self;

    /// Cast the scalar value to a float value. *Alias for `as`.*
    fn cast_to_f32(self) -> f32;
}
impl Scalar for u8 {
    fn cast_from_f32(value: f32) -> Self {
        value as u8
    }
    fn cast_to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for f32 {
    fn cast_from_f32(value: f32) -> Self {
        value
    }
    fn cast_to_f32(self) -> f32 {
        self
    }
}

/// Trait for loading data from a binary file.
/// Occupies constant space, apart from the data itself.
pub trait BinaryScalar: Scalar {
    /// Calculate the exact number of scalar values to be loaded from a binary file.
    ///
    /// limit: The maximum number of scalar values to be loaded, or `None` to load all.
    ///
    /// The return value may be less than `limit` if the file size is smaller than the limit.
    fn file_size_limit(file_path: impl AsRef<Path>, limit: Option<usize>) -> Result<usize> {
        let file_size = std::fs::metadata(file_path)?.len() as usize;
        let file_limit = file_size / mem::size_of::<Self>();
        Ok(limit.unwrap_or(usize::MAX).min(file_limit))
    }
    /// Load data from a binary file.
    /// The layout of the binary file is assumed to be a sequence of scalar values.
    /// The number of scalar values to be loaded is limited by `limit`.
    fn from_binary_file(file_path: impl AsRef<Path>, limit: Option<usize>) -> Result<Vec<Self>>;

    /// Serialize data to a binary file.
    /// The layout of the binary file is a sequence of scalar values.
    fn to_binary_file(data: &[Self], file_path: impl AsRef<Path>) -> Result<()> {
        let mut file = std::fs::File::create(&file_path)?;
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
    fn from_binary_file(file_path: impl AsRef<Path>, limit: Option<usize>) -> Result<Vec<Self>> {
        let limit = Self::file_size_limit(&file_path, limit)?;
        let mut buffer = vec![0; limit];
        let mut file = std::fs::File::open(file_path)?;
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }
}

impl BinaryScalar for f32 {
    fn from_binary_file(file_path: impl AsRef<Path>, limit: Option<usize>) -> Result<Vec<Self>> {
        let limit = Self::file_size_limit(&file_path, limit)?;
        let mut buffer = vec![0.0; limit];
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
