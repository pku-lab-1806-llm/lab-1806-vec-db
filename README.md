# lab-1806-vec-db

Lab 1806 Vector Database.

## Usage with Python

```bash
# See https://pypi.org/project/lab-1806-vec-db/
pip install lab-1806-vec-db
```

Example usage:

```py
import os

from lab_1806_vec_db import BareVecTable, VecDB, calc_dist


def test_calc_dist():
    print("\n[Test] calc_dist")
    dist0 = calc_dist([1.0, 0.0], [0.0, 1.0])  # default: "cosine"
    print(f"{dist0=}")
    assert abs(dist0 - 1.0) < 1e-6, "Test failed"
    print("Test passed")

    print("\n[Test] calc_dist with invalid metric")
    try:
        dist1 = calc_dist([1.0, 0.0], [0.0, 1.0], "euclidean")
        print(f"{dist1=}")
        assert False, "Test failed"
    except ValueError as e:
        print(f"Got expected exception: {e}")
        print("Test passed")


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

**Warning**: All the arguments are positional, do not use keyword arguments like `upper_bound=0.5`.

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
