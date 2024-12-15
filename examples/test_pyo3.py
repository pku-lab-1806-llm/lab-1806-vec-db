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
assert not db.has_hnsw_index(
    "table_1"
), "HNSW index should be cleared when a vector is deleted"

db.build_hnsw_index("table_1")
db.build_pq_table("table_1")
result = db.search("table_1", [1.0, 0.0, 0.0, 0.0], 3, None, 0.5)
print(result)
assert len(result) == 1, "Test failed"
assert result[0][0]["content"] == "a", "Test failed"

print("Test passed")
