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
assert db.get_table_info("table_1") == (
    4,
    3,
    "cosine",
), "Test failed"  # (dim, len, dist)

result = db.search("table_1", [1.0, 0.0, 0.0, 0.0], 3, None, 0.5)
print(result)
assert len(result) == 1, "Test failed"
assert result[0][0]["content"] == "a", "Test failed"

print("Test passed")
