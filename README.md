# lab-1806-vec-db

Rust Implementation for Lab 1806 Vector Database

## Development with Rust

```bash
# Install Rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

# Then install the rust-analyzer extension in VSCode.
# You may need to set "rust-analyzer.runnables.extraEnv" in VSCode Machine settings.
# Otherwise you may fail when press the `Run test` button.

# Run tests
# Add `-r` to test with release mode
cargo test
# Or you can click the 'Run Test' button in VSCode to show output

# Remember NOT to put time cost tests in test modules.
# Just create a new example file for time cost tests.

# Examples
# Put the file at `examples/some_example.rs`
cargo r -r --example some_example

# Binaries
# Put the file at `src/bin/some_binary.rs`
cargo r -r --bin some_binary
```
