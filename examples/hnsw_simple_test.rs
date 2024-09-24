use anyhow::Result;
use lab_1806_vec_db::{
    config::{DBConfig, IndexAlgorithmConfig},
    distance::DistanceAlgorithm,
    index_algorithm::{
        hnsw_index::HNSWIndex, linear_index::LinearIndex, IndexBuilder, IndexFromVecSet, IndexIter,
        IndexKNN,
    },
    vec_set::VecSet,
};
use rand::SeedableRng;
fn clip_msg(s: &str) -> String {
    if s.len() > 100 {
        format!("{}...", &s[..100])
    } else {
        s.to_string()
    }
}
fn main() -> Result<()> {
    let file_path = "config/example/db_config.toml";
    let config = DBConfig::load_from_toml_file(file_path)?;
    println!("Loaded config: {:#?}", config);
    let vec_set = VecSet::<f32>::load_with(&config.vec_data)?;
    let dist = DistanceAlgorithm::L2Sqr;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let config = match config.algorithm {
        IndexAlgorithmConfig::HNSW(config) => config,
        _ => panic!("Testing HNSWIndex with non-HNSW config."),
    };

    let construction_start = std::time::Instant::now();
    let mut index = HNSWIndex::<f32>::new(vec_set.dim(), dist, config);
    for vec in vec_set.iter() {
        index.add(vec, &mut rng);
    }
    println!(
        "Construction time: {:.2} seconds",
        construction_start.elapsed().as_secs_f64()
    );

    // Test the HNSWIndex by comparing with LinearIndex.
    let linear_index = LinearIndex::from_vec_set(vec_set, dist, (), &mut rng);

    let k = 6;
    let query_index = 200;

    println!("Query Index: {}", query_index);
    println!(
        "Query Vector: {}",
        clip_msg(&format!("{:?}", &index[query_index]))
    );

    let result = index.knn(&index[query_index], k);
    let linear_result = linear_index.knn(&linear_index[query_index], k);

    for (res, l_res) in result.iter().zip(linear_result.iter()) {
        println!("Index: {}, Distance: {}", res.index, res.distance);
        println!("Vector: {}", clip_msg(&format!("{:?}", &index[res.index])));
        assert_eq!(res.index, l_res.index, "Index mismatch");
    }
    assert_eq!(result.len(), k.min(index.len()));

    assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));
    Ok(())
}
