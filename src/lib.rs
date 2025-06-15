use mimalloc::MiMalloc;

pub mod config;
pub mod database;
pub mod distance;
pub mod index_algorithm;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(feature = "pyo3")]
mod pyo3;
pub mod scalar;
pub mod vec_set;
pub mod prelude {
    pub use crate::distance::prelude::*;
    pub use crate::index_algorithm::prelude::*;
}
