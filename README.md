# lab-1806-vec-db

Lab 1806 Vector Database.

## Getting Started with Python

```bash
# See https://pypi.org/project/lab-1806-vec-db/
pip install lab-1806-vec-db
```

**Warning**: All the arguments are positional, **DO NOT** use keyword arguments like `upper_bound=0.5`.

### Basic Usage

`VecDB` is recommended for most cases as a high-level API.

If the default Flat index cannot meet your performance requirements, you can call `build_hnsw_index()` when creating a table or adding data. But the delete operation will clear the established HNSW index.

```py
from lab_1806_vec_db import VecDB, calc_dist

# uv sync --reinstall-package lab_1806_vec_db
# uv run -m examples.test_pyo3

# ==== [Test] calc_dist ====
print("\n[Test] calc_dist")
a = [0.3, 0.4]
b = [0.4, 0.3]

# norm_a = sqrt(0.3^2 + 0.4^2) = 0.5
# norm_b = sqrt(0.4^2 + 0.3^2) = 0.5
# dot_product = 0.3 * 0.4 + 0.4 * 0.3 = 0.24
# cosine_dist = 1 - (a dot b) / (|a| * |b|)
#             = 1 - 0.24 / (0.5 * 0.5) = 0.04

cosine_dist = calc_dist(a, b)
print(f"{cosine_dist=}")
assert abs(cosine_dist - 0.04) < 1e-6, "Test failed"


# ==== [Test] VecDB ====
print("\n[Test] VecDB")
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
db.add("table_1", [0.0, 0.0, 1.0, 1.0], {"content": "oops"})
assert db.has_hnsw_index("table_1"), "Add operation should not clear HNSW index"

db.delete("table_1", {"content": "oops"})
assert not db.has_hnsw_index(
    "table_1"
), "HNSW index should be cleared when a vector is deleted"
db.build_hnsw_index("table_1")

result = db.search("table_1", [1.0, 0.0, 0.0, 0.0], 3, None, 0.5)
print(result)
assert len(result) == 1, "Test failed"
assert result[0][0]["content"] == "a", "Test failed"

print("Test passed")

```

### About multi-threading

`VecDB` is thread-safe. You can use it in multiple threads, and it will handle the lock automatically.

When methods on `VecDB` is called, GIL will be temporarily released, so other threads can run Python code.

```py
import random
import threading
import time

from lab_1806_vec_db import VecDB

# 1. Add `std::thread::sleep(std::time::Duration::from_secs(1));` to VecDBManager::search() before `Ok(table.search(query, k, ef, upper_bound))`.
# 2. Run `uv sync --reinstall-package lab_1806_vec_db` to reinstall the package.
# 3. Run `uv run -m examples.test_multi_threads` to test multi-threading search.

"""Sample output:
0.0s: Before search 1
0.6s: Before search 2
1.0s: After search 1
1.2s: Before search 3
1.6s: After search 2
1.8s: Before search 4
2.2s: After search 3
2.8s: After search 4
"""

db = VecDB("tmp/vec_db")

db.create_table_if_not_exists("table1", 1)
db.build_hnsw_index("table1")
db.batch_add(
    "table1",
    [[random.random()] for _ in range(100)],
    [{"id": str(i)} for i in range(100)],
)

count = 0
start = time.time()


def worker():
    global count

    count += 1
    id = f"{count}"
    print(f"{time.time()-start:0.1f}s: Before search {id}")
    db.search("table1", [random.random()], 1)
    print(f"{time.time()-start:0.1f}s: After search {id}")


threads = []
for _ in range(4):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()
    time.sleep(0.6)

for t in threads:
    t.join()

```

### About low-level APIs

Low-level APIs are also provided. But before using them, make sure you know what you are doing.

`BareVecTable` is a low-level API designed for a single table without auto-saving or multi-threading support.

`calc_dist` is a helper function to calculate the distance between two vectors. It supports "cosine" and "l2sqr", default to "cosine". To make sure smaller is closer, we make `cosine_dist = 1 - cosine_similarity`.

### About auto-saving

Safe to interrupt the process on Python Level at any time with Exception or KeyboardInterrupt.

See [test_exception.py](./examples/test_exception.py) for an example.

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

Test the python binding with `test_pyo3.py`.

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
uv run -m examples.test_pyo3

# Build the Python Wheel Release
# This will be automatically run in GitHub Actions.
uv build
```

### Examples Binaries

See also the Binaries at `src/bin/`, and the Examples at `examples/`.

- `src/bin/convert_fvecs.rs`: Convert the fvecs format to the binary format.
- `src/bin/gen_ground_truth.rs`: Generate the ground truth for the query.
- `examples/bench.rs`: The benchmark for index algorithms.

Check the comments at the end of the source files for the usage.

### Dataset

Download Gist1M dataset from:

- Official: <http://corpus-texmex.irisa.fr/>
- Ours: **Recommended** faster, and already converted to the binary format.

  <https://huggingface.co/datasets/pku-lab-1806-llm/gist-for-lab-1806-vec-db>

Then, you can run the examples to test the database.

Note that pre-built index file may be outdated and failed to load. You can build it yourself.
