use super::*;
use crate::{distance::Distance, k_means::KMeans};

pub struct IVFIndex<T> {
    pub k: usize,
    pub vec_sets: Vec<VecSet<T>>,
    pub dist: DistanceAlgorithm,
    pub centroids: VecSet<T>,
}

impl<T> IVFIndex<T>
where
    T: BinaryScalar,
    [T]: Distance,
{
    pub fn from_vec_set(
        vec_set: &VecSet<T>,
        dist: DistanceAlgorithm,
        centroids: VecSet<T>,
    ) -> Self {
        let k = centroids.len();
        let mut vec_sets = Vec::new();
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

        for i in 0..k {
            vec_sets.push(VecSet::<T>::zeros(vec_set.dim(), cent_ids[i].len()));
        }

        // 将数据复制到ivf索引中
        for (cents, v) in cent_ids.iter().zip(vec_sets.iter_mut()) {
            for (i, vec_id) in cents.iter().enumerate() {
                v.put(i, &vec_set[i]);
            }
        }

        Self {
            k: k,
            vec_sets,
            dist,
            centroids,
        }
    }
}
