use std::{
    fs::File,
    io::{BufReader, Read, Write},
};

use anyhow::Result;
use clap::Parser;

/// Convert fvecs to bin
#[derive(Parser)]
struct Args {
    /// Path to the input fvecs file
    input_file: String,
    /// Path to the output bin file
    #[clap(short, long)]
    output_file: String,
    #[clap(short, long)]
    limit: Option<usize>,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let input = File::open(&args.input_file)?;
    let mut buf_reader = BufReader::new(input);
    let output = File::create(&args.output_file)?;
    let mut writer = std::io::BufWriter::new(output);

    println!("Converting fvecs to bin...");

    // fvecs format: <dimension: u32> <vector: [f32; `dimension`]> <dimension> <vector> ...
    // And usually the dimension is fixed for all vectors in the file.
    // bin format: <vectors: [f32; `dimension` * `num_vectors`]>
    let mut dim_buf = [0u8; 4];
    let mut dim = 0;
    let mut vec_buf = Vec::new();
    let mut cnt = 0;

    while buf_reader.read_exact(&mut dim_buf).is_ok() && args.limit.map(|l| cnt < l).unwrap_or(true)
    {
        if dim == 0 {
            dim = u32::from_le_bytes(dim_buf);
            println!("Dimension: {}", dim);
        }
        let dim = u32::from_le_bytes(dim_buf);
        vec_buf.resize((dim as usize) * 4, 0);
        buf_reader.read_exact(&mut vec_buf)?;
        writer.write(&vec_buf)?;
        cnt += 1;
    }
    println!("Done! {} vectors written.", cnt);
    Ok(())
}

// GIST_DIR=/path/to/gist
// cargo r -r --bin convert_fvecs -- $GIST_DIR/train.fvecs -o data/gist_10000.local.bin -l 10000
// cargo r -r --bin convert_fvecs -- $GIST_DIR/train.fvecs -o data/gist.local.bin
// cargo r -r --bin convert_fvecs -- $GIST_DIR/test.fvecs -o data/gist_test.bin
