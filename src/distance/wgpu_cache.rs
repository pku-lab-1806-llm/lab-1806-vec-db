use burn::{backend::Wgpu, prelude::*};

use crate::vec_set::VecSet;

use super::DistanceAlgorithm;

pub struct WgpuCache<B: Backend = Wgpu> {
    pub(crate) dim: usize,
    /// The device used by the cache.
    pub(crate) device: B::Device,
    /// The cached vectors. Shape: `(len, dim)`.
    pub(crate) data: Vec<Tensor<B, 1>>,
}
impl<B: Backend> WgpuCache<B> {
    pub fn convert_slice(&self, vec: &[f32]) -> Tensor<B, 1> {
        Tensor::from_floats(vec, &self.device)
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn lazy_dot_product(a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> Tensor<B, 1> {
        a.clone().mul(b.clone()).sum()
    }
    pub fn vec_norm(vec: &Tensor<B, 1>) -> f32
    where
        B: Backend<FloatElem = f32>,
    {
        Self::lazy_dot_product(&vec, &vec).sqrt().into_scalar()
    }
    pub fn l2_sqr_distance(&self, query: &Tensor<B, 1>, idx: usize) -> f32
    where
        B: Backend<FloatElem = f32>,
    {
        let vec = self.data[idx].clone();
        let diff = vec.sub(query.clone());
        diff.clone().mul(diff).sum().into_scalar()
    }
    pub fn l2_distance(&self, query: &Tensor<B, 1>, idx: usize) -> f32
    where
        B: Backend<FloatElem = f32>,
    {
        let vec = self.data[idx].clone();
        let diff = vec.sub(query.clone());
        diff.clone().mul(diff).sum().sqrt().into_scalar()
    }
    pub fn cosine_distance_cached(
        &self,
        query: &Tensor<B, 1>,
        idx: usize,
        norm_query: f32,
        norm_cached: f32,
    ) -> f32
    where
        B: Backend<FloatElem = f32>,
    {
        let vec = &self.data[idx];
        let dot_product = Self::lazy_dot_product(query, vec).into_scalar();
        1.0 - dot_product / ((norm_query) * (norm_cached))
    }
    pub fn cosine_distance(&self, query: &Tensor<B, 1>, idx: usize) -> f32
    where
        B: Backend<FloatElem = f32>,
    {
        let vec = &self.data[idx];
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

    pub fn distance(&self, dist: DistanceAlgorithm, query: &Tensor<B, 1>, idx: usize) -> f32
    where
        B: Backend<FloatElem = f32>,
    {
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

impl<B: Backend> From<&VecSet<f32>> for WgpuCache<B> {
    fn from(vec_set: &VecSet<f32>) -> Self {
        let dim = vec_set.dim();
        let device = B::Device::default();
        let data = vec_set
            .iter()
            .map(|v| Tensor::<B, 1>::from_floats(v, &device))
            .collect();
        Self { dim, device, data }
    }
}
