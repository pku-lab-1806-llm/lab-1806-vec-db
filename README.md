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
- `examples/bench.rs`: The benchmark for index algorithms.

Check the comments at the end of the source files for the usage.

## Dataset

Download Gist1M dataset from:

- Official: <http://corpus-texmex.irisa.fr/>
- Ours: **Recommended** faster, and already converted to the binary format. We also provide pre-built config file & ground truth & HNSW index.

  <https://huggingface.co/datasets/pku-lab-1806-llm/gist-for-lab-1806-vec-db>

Then, you can run the examples to test the database.
