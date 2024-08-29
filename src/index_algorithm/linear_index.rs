use super::*;

/// Linear index for the k-nearest neighbors search.
/// The distance algorithm is configurable.
///
/// Holds a reference to the `VecSet`.
pub struct LinearIndex<'a, T> {
    vec_set: &'a VecSet<T>,
    distance: DistanceAlgorithm,
}

impl<'a, T> LinearIndex<'a, T>
where
    T: BinaryScalar,
    [T]: Distance,
{
    pub fn from_vec_set(vec_set: &'a VecSet<T>, distance: DistanceAlgorithm) -> Self {
        Self { vec_set, distance }
    }
    /// Get the precise k-nearest neighbors.
    /// Returns a vector of pairs of the index and the distance.
    /// The vector is sorted by the distance in ascending order.
    pub fn knn(&self, query: &[T], k: usize) -> Vec<ResponsePair> {
        let mut result = BTreeSet::new();
        for (i, v) in self.vec_set.iter().enumerate() {
            let dist = self.distance.d(query, v);
            if result.len() < k {
                result.insert(ResponsePair::new(i, dist));
            } else if let Some(max) = result.last() {
                if dist < max.distance {
                    result.insert(ResponsePair::new(i, dist));
                    result.pop_last();
                }
            }
        }
        result.into_iter().collect()
    }
}

/// Linear index for the k-nearest neighbors search.
///
/// Holds a reference to the `DynamicVecSet`.
///
/// The data type is determined at runtime.
pub enum DynamicLinearIndex<'a> {
    Float32(LinearIndex<'a, f32>),
    UInt8(LinearIndex<'a, u8>),
}

impl<'a> DynamicLinearIndex<'a> {
    pub fn from_dynamic_vec_set(vec_set: &'a DynamicVecSet, distance: DistanceAlgorithm) -> Self {
        use DynamicVecSet::*;
        match vec_set {
            Float32(vec_set) => Self::Float32(LinearIndex::from_vec_set(vec_set, distance)),
            UInt8(vec_set) => Self::UInt8(LinearIndex::from_vec_set(vec_set, distance)),
        }
    }
    pub fn knn(&self, query: &DynamicVecRef, k: usize) -> Vec<ResponsePair> {
        use DynamicVecRef::*;
        match (self, query) {
            (Self::Float32(index), Float32(query)) => index.knn(query, k),
            (Self::UInt8(index), UInt8(query)) => index.knn(query, k),
            _ => panic!("Mismatched types when calling knn in DynamicLinearIndex."),
        }
    }
}
