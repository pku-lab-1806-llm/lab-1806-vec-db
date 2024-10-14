# lab-1806-vec-db

Lab 1806 Vector Database.

## Usage with Python

See <https://github.com/pku-lab-1806-llm/lab-1806-vec-db/releases/latest> for the latest release.

Install the wheel file with pip.

```py
from lab_1806_vec_db import RagVecDB

db = RagVecDB(dim=4)

db.add([1.0, 0.0, 0.0, 0.0], {"content": "a"})
db.add([1.0, 0.0, 0.0, 0.1], {"content": "aa"})

db.add([0.0, 1.0, 0.0, 0.0], {"content": "b"})
db.add([0.0, 1.0, 0.0, 0.1], {"content": "bb"})

db.add([0.0, 0.0, 1.0, 0.0], {"content": "c"})
db.add([0.0, 0.0, 1.0, 0.1], {"content": "cc"})

db.save("test_db.local.bin")

loaded_db = RagVecDB.load("test_db.local.bin")

for idx, metadata in enumerate(loaded_db.search([1.0, 0.0, 0.0, 0.0], 2)):
    print(idx, metadata["content"])
```

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

Test the python binding with `test-pyo3.py`.

```bash
# Install Python 3.10
brew install python@3.10
# or on Windows
scoop bucket add versions
scoop install python310

# Install uv.
# See https://github.com/astral-sh/uv for alternatives.
pip install uv
# or on Windows
scoop install uv

# Run the Python test
uv sync --reinstall-package lab_1806_vec_db
uv run ./test-pyo3.py

# Build the Python Wheel Release
# This will be automatically run in GitHub Actions.
uv build
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
