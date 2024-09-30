# lab-1806-vec-db

Lab 1806 Vector Database.

## Development with Rust

```bash
# Install Rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

# Then install the rust-analyzer extension in VSCode.
# You may need to set "rust-analyzer.runnables.extraEnv" in VSCode Machine settings.
# The value should be like {"PATH":""} and make sure that `/home/YOUR_NAME/.cargo/bin` is in it.
# Otherwise you may fail when press the `Run test` button.

# Run tests
# Add `-r` to test with release mode
cargo test
# Or you can click the 'Run Test' button in VSCode to show output.
# Our GitHub Actions will also run the tests.
```

## Examples Binaries

See also the Binaries at `src/bin/`, and the Examples at `examples/`.

- `src/bin/convert_fvecs.rs`: Convert the fvecs format to the binary format.
- `src/bin/gen_ground_truth.rs`: Generate the ground truth for the query.
- `examples/hnsw_bench.rs`: The benchmark for the HNSW algorithm.

Check the comments at the end of the source files for the usage.

## Dataset

Download Gist1M dataset from:

- <http://corpus-texmex.irisa.fr/>

  Official but slow.
- <https://huggingface.co/datasets/fzliu/gist1m/blob/main/gist.tar.gz>

  Fast but unofficial.

`gist.tar.gz` SHA256: 01469a7f1c3768853525e543d537e2dfa1adece927616405e360952e3f67df73

Run `src/bin/convert_fvecs.rs` to convert the fvecs format to the binary format.

`gist.local.bin` SHA256: 1d9dfa933a86b1b78ef3de82eb286f85899866c428d373e42613783a13811038

Don't forget to create `config/{name}.local.toml` for each dataset file.

Then, you can run the examples to test the database.
