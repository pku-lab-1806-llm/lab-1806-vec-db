use crate::prelude::*;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::{scalar::Scalar, vec_set::VecSet};

pub trait GpuScalar: candle_core::WithDType {}
impl GpuScalar for u8 {}
impl GpuScalar for f32 {}

pub struct GpuVecSet {
    pub(crate) device: Device,
    pub(crate) norm_cache: Tensor,
    pub(crate) data: Tensor,
}
impl<T: Scalar> TryFrom<&VecSet<T>> for GpuVecSet {
    type Error = anyhow::Error;
    fn try_from(value: &VecSet<T>) -> Result<Self, Self::Error> {
        let dim = value.dim();
        let device = Device::cuda_if_available(0)?;
        let data = Tensor::from_slice(value.as_slice(), (value.len(), dim), &device)?;
        let norm_cache = (&data * &data)?.sum(1)?.sqrt()?;
        Ok(GpuVecSet {
            device,
            norm_cache,
            data,
        })
    }
}
impl GpuVecSet {
    pub fn parse_query<T: Scalar>(&self, query: &[T]) -> Result<Tensor> {
        let query = Tensor::from_slice(query, query.len(), &self.device)?;
        if query.dtype() == DType::F32 {
            return Ok(query);
        }
        Ok(query.to_dtype(DType::F32)?)
    }
    pub fn batch_l2_sqr(&self, query: &Tensor) -> Result<Tensor> {
        let diff = self.data.broadcast_sub(query)?;
        let diff = (&diff * &diff)?;
        Ok(diff.sum(1)?)
    }
    pub fn batch_l2(&self, query: &Tensor) -> Result<Tensor> {
        Ok(self.batch_l2_sqr(query)?.sqrt()?)
    }
    pub fn batch_cosine(&self, query: &Tensor) -> Result<Tensor> {
        let norm_query: f32 = (query * query)?.sum(0)?.sqrt()?.to_scalar()?;
        let dot_product = self.data.broadcast_mul(query)?.sum(1)?;
        Ok((dot_product / &self.norm_cache)?.affine(-1.0 / norm_query as f64, 1.0)?)
    }
    pub fn batch_distance(&self, query: &Tensor, dist: DistanceAlgorithm) -> Result<Tensor> {
        use DistanceAlgorithm::*;
        match dist {
            L2Sqr => self.batch_l2_sqr(query),
            L2 => self.batch_l2(query),
            Cosine => self.batch_cosine(query),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_gpu_cache() -> Result<()> {
        use DistanceAlgorithm::*;
        println!(
            "CUDA available: {}",
            Device::cuda_if_available(0).unwrap().is_cuda()
        );
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 20;
        let a: Vec<f32> = (0..dim * 5).map(|_| rng.gen_range(0.0..1.0)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0..1.0)).collect();

        let vec_set = VecSet::new(dim, a.clone());
        println!("Trying to create GpuCache");
        let cache = GpuVecSet::try_from(&vec_set)?;
        let query = cache.parse_query(&b)?;

        for dist in &[L2Sqr, L2, Cosine] {
            println!("dist={:?}", dist);
            let result = match dist {
                L2Sqr => cache.batch_l2_sqr(&query)?,
                L2 => cache.batch_l2(&query)?,
                Cosine => cache.batch_cosine(&query)?,
            };
            let result = result.to_vec1::<f32>()?;
            println!("result={:?}", result);
            let std = vec_set.iter().map(|v| dist.d(v, &b)).collect::<Vec<_>>();
            println!("std={:?}", std);
            for (r, s) in result.iter().zip(std.iter()) {
                assert!((r - s).abs() < 1e-6 * s.max(1.0), "r={}, s={}", r, s);
            }
        }
        Ok(())
    }
}
