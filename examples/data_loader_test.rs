use anyhow::Result;
use lab_1806_vec_db::{config::DBConfig, distance::Distance, vec_set::TypedVecSet};

fn main() -> Result<()> {
    let file_path = "config/example/db_config.toml";
    let config = DBConfig::load_from_toml_file(file_path)?;
    println!("Loaded config: {:#?}", config);
    let vec_set = TypedVecSet::load_with(config.vec_data)?;
    let v0 = vec_set.i(0);
    let v1 = vec_set.i(1);
    println!("Distance Algorithm: {:?}", config.distance);
    println!("L2Sqr distance: {}", v0.distance(&v1, config.distance));
    Ok(())
}
