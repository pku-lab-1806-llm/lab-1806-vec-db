use std::ops::Index;

use serde::{Deserialize, Serialize};

use crate::{
    distance::DistanceAlgorithm,
    scalar::{ConcatLayout, Scalar},
};

use super::{IndexBuilder, IndexIter};

/// The configuration of the HNSW (Hierarchical Navigable Small World) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWConfig {
    pub max_elements: usize,
    pub ef_construction: usize,
    pub M: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct HNSWInnerConfig {
    pub dim: usize,
    pub dist: DistanceAlgorithm,
    pub max_elements: usize,
    pub ef_construction: usize,
    pub M: usize,
    pub max_M0: usize,
    pub ef: usize,
    /// element_level0: (links_len, links, data, index): (u32, [u32; max_M0], [T; dim], u32)
    pub element_level0_layout: ConcatLayout,
}

pub struct HNSWIndex<T> {
    pub t_phantom: std::marker::PhantomData<T>,
    pub config: HNSWInnerConfig,
    pub level0_data: Box<[u8]>,
    pub len: usize,
    pub max_level: usize,
    pub enter_point: usize,
}
impl<T: Scalar> HNSWIndex<T> {
    /// Parse an element from the memory.
    ///
    /// Returns links_len, &links, &data, index.
    pub unsafe fn parse_element(&self, ptr: *const u8) -> (u32, &[u32], &[T], u32) {
        let offsets = &self.config.element_level0_layout.offsets;
        let links_len = *(ptr.add(offsets[0]) as *const u32);
        let links =
            std::slice::from_raw_parts(ptr.add(offsets[1]) as *const u32, links_len as usize);
        let data = std::slice::from_raw_parts(ptr.add(offsets[2]) as *const T, self.config.dim);
        let index = *(ptr.add(offsets[3]) as *const u32);
        (links_len, links, data, index)
    }
    /// Parse an element from the memory.
    ///
    /// Returns &mut links_len, &mut links, &mut data, &mut index.
    pub unsafe fn parse_element_mut(
        &mut self,
        ptr: *mut u8,
    ) -> (&mut u32, &mut [u32], &mut [T], &mut u32) {
        let offsets = &self.config.element_level0_layout.offsets;
        let links_len = &mut *(ptr.add(offsets[0]) as *mut u32);
        let links =
            std::slice::from_raw_parts_mut(ptr.add(offsets[1]) as *mut u32, self.config.max_M0);
        let data = std::slice::from_raw_parts_mut(ptr.add(offsets[2]) as *mut T, self.config.dim);
        let index = &mut *(ptr.add(offsets[3]) as *mut u32);
        (links_len, links, data, index)
    }

    /// Get an element from the index.
    ///
    /// Returns links_len, &links, &data, index.
    pub fn get_element(&self, index: usize) -> (u32, &[u32], &[T], u32) {
        let ptr = self.level0_data.as_ptr();
        unsafe { self.parse_element(ptr.add(index * self.config.element_level0_layout.size)) }
    }

    /// Get an element from the index. Mutable version.
    ///
    /// Returns &mut links_len, &mut links, &mut data, &mut index.
    pub fn get_element_mut(&mut self, index: usize) -> (&mut u32, &mut [u32], &mut [T], &mut u32) {
        let ptr = self.level0_data.as_mut_ptr();
        unsafe { self.parse_element_mut(ptr.add(index * self.config.element_level0_layout.size)) }
    }
}
impl<T: Scalar> Index<usize> for HNSWIndex<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let (_, _, data, _) = self.get_element(index);
        data
    }
}
impl<T: Scalar> IndexIter<T> for HNSWIndex<T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T: Scalar> IndexBuilder<T> for HNSWIndex<T> {
    type Config = HNSWConfig;
    fn dim(&self) -> usize {
        self.config.dim
    }
    fn new(dim: usize, dist: DistanceAlgorithm, config: Self::Config) -> Self {
        #[allow(non_snake_case)]
        let M = config.M;
        assert!(
            M <= 10_000,
            " M parameter exceeds 10_000 which may lead to adverse effects."
        );
        #[allow(non_snake_case)]
        let max_M0 = 2 * M;
        let max_elements = config.max_elements;

        let ef = 10;
        let ef_construction = config.ef_construction.max(config.M);

        let mut element_level0_layout = ConcatLayout::new();
        element_level0_layout.push::<u32>(1);
        element_level0_layout.push::<u32>(max_M0);
        element_level0_layout.push::<T>(dim);
        element_level0_layout.push::<u32>(1);

        let mut level0_layout = ConcatLayout::new();
        level0_layout.push_sub(&element_level0_layout, max_elements);

        let level0_data = level0_layout.alloc();

        let config = HNSWInnerConfig {
            dim,
            dist,
            max_elements,
            ef_construction,
            M,
            max_M0,
            ef,
            element_level0_layout,
        };

        Self {
            t_phantom: std::marker::PhantomData,
            config,
            level0_data,
            len: 0,
            max_level: 0,
            enter_point: 0,
        }
    }
    fn add(&mut self, _vec: &[T], _rng: &mut impl rand::Rng) -> usize {
        unimplemented!("HNSWIndex::add")
    }
}
