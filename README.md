# lab-1806-vec-db

Rust Implementation for Lab 1806 Vector Database

## Development with Rust

```bash
# Install Rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Install Rust
. "$HOME/.cargo/env"

# Then install the rust-analyzer extension in VSCode.

# Run tests
# Or you can click the 'Run Test' button in VSCode to show output
cargo test

# Remember NOT to put time cost tests in test modules.
# Just create a new example file for time cost tests.
# So you can run them with release mode.

# Examples
# Put the file at `examples/some_example.rs`
# Then run with
cargo r -r --example some_example

# Binaries
# Put the file at `src/bin/some_binary.rs`
# Then run with
cargo r -r --bin some_binary
```
