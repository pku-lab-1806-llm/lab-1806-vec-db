use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    prelude::*,
};

use crate::{scalar::BaseScalar, vec_set::VecSet};

use super::DistanceAlgorithm;

type WgpuVec = Tensor<Wgpu, 1>;

#[derive(Debug, Clone)]
pub struct WgpuCache {
    /// The dimension of the vectors.
    pub(crate) dim: usize,
    /// The device used by the cache.
    pub(crate) device: WgpuDevice,
    /// The cached vectors. Flattened: `(len* dim,)`.
    pub(crate) data: WgpuVec,
}
impl WgpuCache {
    pub fn convert_slice<T: WgpuScalar>(&self, vec: &[T]) -> WgpuVec {
        T::to_tensor(&self.device, vec)
    }
    pub fn len(&self) -> usize {
        self.data.shape().dims[0] / self.dim
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn lazy_dot_product(a: &WgpuVec, b: &WgpuVec) -> WgpuVec {
        a.clone().mul(b.clone()).sum()
    }
    pub fn vec_norm(vec: &WgpuVec) -> f32 {
        vec.clone().prod().sqrt().into_scalar()
    }
    pub fn get(&self, idx: usize) -> WgpuVec {
        self.data.clone().narrow(0, idx * self.dim, self.dim)
    }
    pub fn l2_sqr_distance(&self, query: &WgpuVec, idx: usize) -> f32 {
        let vec = self.get(idx);
        let diff = vec.sub(query.clone());
        diff.clone().mul(diff).sum().into_scalar()
    }
    pub fn l2_distance(&self, query: &WgpuVec, idx: usize) -> f32 {
        let vec = self.get(idx);
        let diff = vec.sub(query.clone());
        diff.clone().mul(diff).sum().sqrt().into_scalar()
    }
    pub fn cosine_distance_cached(
        &self,
        query: &WgpuVec,
        idx: usize,
        norm_query: f32,
        norm_cached: f32,
    ) -> f32 {
        let vec = &self.get(idx);
        let dot_product = Self::lazy_dot_product(query, vec).into_scalar();
        1.0 - dot_product / ((norm_query) * (norm_cached))
    }
    pub fn cosine_distance(&self, query: &WgpuVec, idx: usize) -> f32 {
        let vec = &self.get(idx);
        let dot_product = Self::lazy_dot_product(query, vec);
        let norm_lhs = Self::lazy_dot_product(query, query).sqrt();
        let norm_rhs = Self::lazy_dot_product(vec, vec).sqrt();
        // Parallelize the computation of the dot product and the norms.
        let (dot_product, norm_lhs, norm_rhs) = (
            dot_product.into_scalar(),
            norm_lhs.into_scalar(),
            norm_rhs.into_scalar(),
        );
        1.0 - dot_product / (norm_lhs * norm_rhs)
    }

    pub fn distance(&self, dist: DistanceAlgorithm, query: &WgpuVec, idx: usize) -> f32 {
        use DistanceAlgorithm::*;
        match dist {
            L2Sqr => self.l2_sqr_distance(query, idx),
            L2 => self.l2_distance(query, idx),
            Cosine => self.cosine_distance(query, idx),
            #[allow(unreachable_patterns)]
            _ => unimplemented!("Distance algorithm not implemented"),
        }
    }
}

impl From<&VecSet<f32>> for WgpuCache {
    fn from(vec_set: &VecSet<f32>) -> Self {
        let dim = vec_set.dim();
        let device = WgpuDevice::DiscreteGpu(0);
        let data = f32::to_tensor(&device, &vec_set.data);
        Self { dim, device, data }
    }
}
pub trait WgpuScalar: BaseScalar {
    fn to_tensor(device: &WgpuDevice, vec: &[Self]) -> WgpuVec;
}
impl WgpuScalar for f32 {
    fn to_tensor(device: &WgpuDevice, vec: &[Self]) -> WgpuVec {
        WgpuVec::from_floats(vec, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_cache() {
        let vec_set = VecSet::new(3, vec![1.0, 2.0, 3.0]);
        let cache = WgpuCache::from(&vec_set);
        println!("{:?}", cache);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.dim(), 3);
        let query = cache.convert_slice(&[4.0, 5.0, 6.0]);
        let dist = cache.distance(DistanceAlgorithm::L2Sqr, &query, 0);
        assert!((dist - 27.0).abs() < 1e-6);
    }
}
