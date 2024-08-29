use super::*;
use crate::{config, distance, k_means::{DynamicKMeans, KMeans}};

pub struct IVFIndex<T> {
    k: usize, 
    vec_sets: Vec<VecSet<T>>,
    dist: DistanceAlgorithm,
    centroids: VecSet<T>,
}

impl<T> IVFIndex<T> 
where 
    T: BinaryScalar,
    [T]: Distance
{
    pub fn from_vec_set(vec_set: & VecSet<T>, dist: DistanceAlgorithm, centroids: VecSet<T>) -> Self {
        let k = centroids.len();
        let vec_sets = Vec::new();
        let mut cent_ids = vec![Vec::<usize>::new(); k];

        for (i, v) in vec_set.iter().enumerate() {
            let (_, min_idx) = centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (dist.distance(c, v), i))
                .min_by(|(d0, _), (d1, _)| d0.partial_cmp(d1).unwrap())
                .unwrap();
            cent_ids[min_idx].push(i);
        }
        
        Self {k: k, vec_sets, dist, centroids}
    }
}