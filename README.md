# lab-1806-vec-db

Lab 1806 Vector Database.

See Rust source code at [GitHub lab-1806-vec-db](https://github.com/pku-lab-1806-llm/lab-1806-vec-db)

## Getting Started with Python

Get the latest version from [PyPI lab-1806-vec-db](https://pypi.org/project/lab-1806-vec-db/).

```bash
pip install lab-1806-vec-db
```

### Basic Usage

`VecDB` is recommended for most cases as a high-level API.

If the default Flat index cannot meet your performance requirements, try:

- Call `build_hnsw_index()` when creating a table or adding data. But the delete operation will clear the established HNSW index.
- Call `build_pq_index()` when all the data is added. But any write operation will clear the established PQ index.

**Warning**: All the arguments are positional, **DO NOT** use keyword arguments like `upper_bound=0.5`.

```py
from lab_1806_vec_db import VecDB

# uv sync --reinstall-package lab_1806_vec_db
# uv run -m examples.test_pyo3

db = VecDB("./tmp/vec_db")
for key in db.get_all_keys():
    db.delete_table(key)

keys = db.get_all_keys()
assert len(keys) == 0, "Test failed"

db.create_table_if_not_exists("table_1", 4)
db.add("table_1", [1.0, 0.0, 0.0, 0.0], {"content": "a"})
db.add("table_1", [0.0, 1.0, 0.0, 0.0], {"content": "b"})
db.build_hnsw_index("table_1")
db.add("table_1", [0.0, 0.0, 1.0, 0.0], {"content": "c"})
db.add("table_1", [0.0, 0.0, 1.0, 1.0], {"content": "d", "type": "oops"})
assert db.has_hnsw_index("table_1"), "Add operation should not clear HNSW index"

db.delete("table_1", {"type": "oops"})
assert db.get_len("table_1") == 3, "Test failed"
assert not db.has_hnsw_index("table_1"), (
    "HNSW index should be cleared when a vector is deleted"
)

db.build_hnsw_index("table_1")
db.build_pq_table("table_1")
result = db.search("table_1", [1.0, 0.0, 0.0, 0.0], 3, None, 0.5)
print(result)
assert len(result) == 1, "Test failed"
assert result[0][0]["content"] == "a", "Test failed"

# Manually call `db.force_save()` when you are using the database in Jupyter Notebook or FastAPI,
# since the program may not exit normally.

print("Test passed")
```

### About multi-threading

`VecDB` is thread-safe. You can use it in multiple threads, and it will handle the lock automatically.

When methods on `VecDB` is called, GIL will be temporarily released, so other threads can run Python code.

Feel free to use this in FastAPI routes or other environments with ThreadPools.

See also [multi-threading example](./examples/test_multi_threads.py).

### About auto-saving

Safe to interrupt the process on Python Level at any time with Exception or KeyboardInterrupt.

See [test_exception.py](./examples/test_exception.py) for an example.

However some platform may have timeout when shutting down, call `db.force_save()` in lifespan to ensure the data is saved in FastAPI.

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

### Test Python Bindings

Install Python `>=3.10,<3.13` and [uv](https://github.com/astral-sh/uv).

```bash
# Run the Python test
uv sync --reinstall-package lab_1806_vec_db
uv run -m examples.test_pyo3

# Build the Python Wheel Release
uv build
```

### Examples Binaries

- `src/bin/convert_fvecs.rs`: Convert the fvecs format to the binary format.
- `src/bin/gen_ground_truth.rs`: Generate the ground truth for the query.
- `examples/bench.rs`: The benchmark for index algorithms.

### Dataset

Download Gist1M dataset from:

- Official: <http://corpus-texmex.irisa.fr/>
- Ours: **Recommended** faster, and already converted to the binary format.

  <https://huggingface.co/datasets/pku-lab-1806-llm/gist-for-lab-1806-vec-db>

Then, you can run the examples to test the database.

Note that pre-built index file may be outdated. You can build it yourself.
