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

Low-level APIs are also provided. But before using them, make sure you know what you are doing.

`BareVecTable` is a low-level API designed for a single table without auto-saving or multi-threading support.

`calc_dist` is a helper function to calculate the distance between two vectors. It supports "cosine" and "l2sqr", default to "cosine". To make sure smaller is closer, we make `cosine_dist = 1 - cosine_similarity`.

```py
import os

from lab_1806_vec_db import BareVecTable, VecDB, calc_dist


def test_calc_dist():
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


test_calc_dist()


def test_bare_vec_table():
    print("\n[Test] BareVecTable")
    table = BareVecTable(dim=4)
    table.add([1.0, 0.0, 0.0, 0.0], {"content": "a"})
    table.add([0.0, 1.0, 0.0, 0.0], {"content": "b"})
    table.add([0.0, 0.0, 1.0, 0.0], {"content": "c"})

    table.batch_add(
        [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 1.0, 0.1]],
        [{"content": x} for x in ["aa", "bb", "cc"]],
    )
    # Save and load <<<<
    table.save("test_table.local.db")
    table = BareVecTable.load("test_table.local.db")
    os.remove("test_table.local.db")
    # Save and load >>>>

    results = table.search([1.0, 0.0, 0.0, 0.0], 2)
    contents: list[str] = []
    for metadata, d in results:
        print(metadata["content"], d)
        contents.append(metadata["content"])
    assert (contents[0], contents[1]) == ("a", "aa"), "Test failed"
    print("Test passed")


test_bare_vec_table()


def test_vec_db():
    print("\n[Test] VecDB")
    db = VecDB("./tmp/vec_db")
    for key in db.get_all_keys():
        db.delete_table(key)

    keys = db.get_all_keys()
    assert len(keys) == 0, "Test failed"

    db.create_table_if_not_exists("table_1", 4)
    db.add("table_1", [1.0, 0.0, 0.0, 0.0], {"content": "a"})
    db.add("table_1", [0.0, 1.0, 0.0, 0.0], {"content": "b"})
    db.add("table_1", [0.0, 0.0, 1.0, 0.0], {"content": "c"})

    db.create_table_if_not_exists("table_2", 4)
    db.batch_add(
        "table_2",
        [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 1.0, 0.1]],
        [{"content": x} for x in ["aa", "bb", "cc"]],
    )

    result = db.search("table_1", [1.0, 0.0, 0.0, 0.0], 3, None, 0.5)
    print(result)
    assert len(result) == 1, "Test failed"
    assert result[0][0]["content"] == "a", "Test failed"

    results = db.join_search({"table_1", "table_2"}, [1.0, 0.0, 0.0, 0.0], 2)

    for key, metadata, d in results:
        print(key, metadata["content"], d)

    assert len(results) == 2, "Test failed"
    assert results[0][0] == "table_1", "Test failed"
    assert results[0][1]["content"] == "a", "Test failed"
    assert results[1][0] == "table_2", "Test failed"
    assert results[1][1]["content"] == "aa", "Test failed"
    print("Test passed")


test_vec_db()
```

### About auto-saving

Safe to interrupt the process on Python Level at any time with Exception or KeyboardInterrupt.

```py
import os

from lab_1806_vec_db import VecDB

if os.path.exists("./tmp/vec_db"):
    for file in os.listdir("./tmp/vec_db"):
        os.remove(f"./tmp/vec_db/{file}")
# Wait for the user to see the empty dir
input("Press Enter to continue...")


# Create the database
db = VecDB("./tmp/vec_db")
db.create_table_if_not_exists("table_1", 1)
db.add("table_1", [0.0], {"content": "0"})


# Data will be written to the disk every 30 seconds
# Here we can see the dir contains a lock file.
# And after 5 seconds, the `brief.toml` will be created
# And after 30 seconds, `*.db` files will be created

# Try interrupting the process at different stages
# And check if the data appears in the disk at last

# Cases:
# - Wait for 30 seconds without doing anything
# - Enter to exit normally
# - Type "raise" to raise an exception
# - Type "dim" to do a wrong operation
# - Press Ctrl+C to interrupt the process

cmd = input("Type to choose the action: ")
if cmd == "":
    exit(0)  # Exit normally
elif cmd == "raise":
    raise Exception("Deliberate exception")
elif cmd == "dim":
    # Dimension mismatch
    db.add("table_1", [0.0, 1.0], {"content": "0, 1"})
elif cmd == "len":
    # Length mismatch
    db.batch_add("table_1", [[0.0], [1.0]], [{"content": "0"}])

# File appears before the program exits in all cases, even with Exception or KeyboardInterrupt
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
- Ours: **Recommended** faster, and already converted to the binary format. We also provide pre-built config file & ground truth & HNSW index.

  <https://huggingface.co/datasets/pku-lab-1806-llm/gist-for-lab-1806-vec-db>

Then, you can run the examples to test the database.
