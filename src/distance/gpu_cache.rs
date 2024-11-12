use candle_core::{DType, Device, IndexOp, Tensor};

use crate::{scalar::Scalar, vec_set::VecSet};

use super::DistanceAlgorithm;

#[derive(Debug)]
pub struct GpuCache {
    pub(crate) device: Device,
    pub(crate) data: Tensor,
}
pub trait GpuScalar: candle_core::WithDType {
    fn to_tensor(slice: &[Self], device: &Device) -> Tensor {
        Tensor::from_slice(slice, (slice.len(),), device).unwrap()
    }
}
impl GpuScalar for u8 {}
impl GpuScalar for f32 {}

pub trait TensorToF32 {
    fn to_f32(&self) -> f32;
}
impl TensorToF32 for Tensor {
    fn to_f32(&self) -> f32 {
        self.to_scalar().unwrap()
    }
}

impl<T: Scalar> From<&VecSet<T>> for GpuCache {
    fn from(vec_set: &VecSet<T>) -> Self {
        let dim = vec_set.dim();
        let device = Device::cuda_if_available(0).unwrap();
        let data = T::to_tensor(vec_set.as_slice(), &device);
        let data = data.reshape(((), dim)).unwrap();
        GpuCache { device, data }
    }
}
impl GpuCache {
    pub fn parse_query<T: Scalar>(&self, query: &[T]) -> Tensor {
        T::to_tensor(query, &self.device)
            .to_dtype(DType::F32)
            .unwrap()
    }
    pub fn get(&self, idx: usize) -> Tensor {
        if self.data.dtype() == DType::F32 {
            return self.data.i(idx).unwrap();
        }
        self.data.i(idx).unwrap().to_dtype(DType::F32).unwrap()
    }
    pub fn len(&self) -> usize {
        self.data.shape().dims()[0]
    }
    pub fn dim(&self) -> usize {
        self.data.shape().dims()[1]
    }
    pub fn dot_product(a: &Tensor, b: &Tensor) -> Tensor {
        (a * b).unwrap().sum(0).unwrap()
    }
    pub fn norm_vec(vec: &Tensor) -> Tensor {
        Self::dot_product(vec, vec).sqrt().unwrap()
    }
    pub fn l2_sqr_distance(&self, query: &Tensor, idx: usize) -> Tensor {
        let vec = self.get(idx);
        let diff = (query - vec).unwrap();
        Self::dot_product(&diff, &diff)
    }
    pub fn l2_distance(&self, query: &Tensor, idx: usize) -> Tensor {
        self.l2_sqr_distance(query, idx).sqrt().unwrap()
    }
    pub fn cosine_distance_cached(
        &self,
        query: &Tensor,
        idx: usize,
        norm_query: f32,
        norm_cached: f32,
    ) -> f32 {
        let vec = &self.get(idx);
        let dot_product = Self::dot_product(query, vec);
        1.0 - dot_product.to_f32() / (norm_query * norm_cached)
    }
    pub fn cosine_distance(&self, query: &Tensor, idx: usize) -> Tensor {
        let vec = &self.get(idx);
        let dot_product = Self::dot_product(query, vec);
        let norm_query = Self::norm_vec(query);
        let norm_cached = Self::norm_vec(vec);
        (1.0 - (dot_product / (norm_query * norm_cached).unwrap()).unwrap()).unwrap()
    }
    pub fn distance(&self, query: &Tensor, idx: usize, dist: DistanceAlgorithm) -> f32 {
        match dist {
            DistanceAlgorithm::L2Sqr => self.l2_sqr_distance(query, idx),
            DistanceAlgorithm::L2 => self.l2_distance(query, idx),
            DistanceAlgorithm::Cosine => self.cosine_distance(query, idx),
        }
        .to_f32()
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::prelude::DistanceAdapter;

    use super::*;

    #[test]
    fn test_gpu_cache() {
        println!(
            "CUDA available: {}",
            Device::cuda_if_available(0).unwrap().is_cuda()
        );
        let mut rng = StdRng::seed_from_u64(42);
        let a: Vec<f32> = (0..20).map(|_| rng.gen_range(0.0..1.0)).collect();
        let b: Vec<f32> = (0..20).map(|_| rng.gen_range(0.0..1.0)).collect();

        let vec_set = VecSet::new(a.len(), a.clone());
        println!("Trying to create GpuCache");
        let cache = GpuCache::from(&vec_set);
        let query = cache.parse_query(&b);

        for dist in &[
            DistanceAlgorithm::L2Sqr,
            DistanceAlgorithm::L2,
            DistanceAlgorithm::Cosine,
        ] {
            println!("dist={:?}", dist);
            let gpu_result = cache.distance(&query, 0, *dist);
            let std_result = dist.d(a.as_slice(), b.as_slice());
            assert!(
                (gpu_result - std_result).abs() < 1e-6,
                "dist={:?}, gpu_result={}, std_result={}",
                dist,
                gpu_result,
                std_result
            );
        }
    }
}
