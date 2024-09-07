use std::rc::Rc;

use anyhow::Result;
use lab_1806_vec_db::{
    config::DBConfig,
    distance::{
        DistanceAdapter,
        DistanceAlgorithm::{self, *},
    },
    pq_table::{PQConfig, PQTable},
    scalar::Scalar,
    vec_set::VecSet,
};
use rand::{Rng, SeedableRng};

fn pq_table_test_on_real_set_base<T: Scalar>(
    vec_set: &VecSet<T>,
    dist: DistanceAlgorithm,
) -> Result<()> {
    let start_time = std::time::Instant::now();
    let dim = vec_set.dim();
    let pq_config = Rc::new(PQConfig {
        n_bits: 4,
        m: dim / 4,
        dist,
        k_means_max_iter: 20,
        k_means_tol: 1e-6,
    });
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let pq_table = PQTable::from_vec_set(&vec_set, pq_config, &mut rng);
    let encoded_set = pq_table.encode_batch(&vec_set);

    println!("Distance Algorithm: {:?}", dist);
    let test_count = 100;
    let print_count = test_count.min(5);
    let mut errors = Vec::new();
    for i in 0..test_count {
        let i0 = rng.gen_range(0..vec_set.len());
        let i1 = rng.gen_range(0..vec_set.len());
        let v0 = &vec_set[i0];
        let v1 = &vec_set[i1];
        let e0 = &encoded_set[i0];
        let lookup = pq_table.create_lookup(v1);
        let distance = dist.d(e0, &lookup);
        let expected = dist.d(v0, v1);
        let error = (distance - expected).abs() / expected.max(1.0);
        if i < print_count {
            println!(
                "Distance: {} / Expected: {} / Error: {}",
                distance, expected, error
            );
        }
        errors.push(error);
    }
    if test_count > print_count {
        println!("...");
    }
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i95 = (errors.len() as f32 * 0.95).ceil() as usize - 1;
    let p95 = errors[i95];
    println!(
        "95% Error: {} / Elapsed Time: {:?}",
        p95,
        start_time.elapsed()
    );
    assert!(p95 < 0.25, "95% Error is too large.");
    Ok(())
}

fn main() -> Result<()> {
    let file_path = "config/example/db_config.toml";
    let config = DBConfig::load_from_toml_file(file_path)?;

    let vec_set = VecSet::<f32>::load_with(&config.vec_data)?;
    pq_table_test_on_real_set_base(&vec_set, L2Sqr)?;
    pq_table_test_on_real_set_base(&vec_set, L2)?;
    pq_table_test_on_real_set_base(&vec_set, Cosine)?;
    Ok(())
}
