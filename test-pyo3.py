import os

from lab_1806_vec_db import RagMultiVecDB, RagVecDB, calc_dist

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

print("\n[Test] RagVecDB and RagMultiVecDB")

# Create and add <<<<
db_a = RagVecDB(dim=4)

db_a.add([1.0, 0.0, 0.0, 0.0], {"content": "a"})
db_a.add([0.0, 1.0, 0.0, 0.0], {"content": "b"})
db_a.add([0.0, 0.0, 1.0, 0.0], {"content": "c"})

db_b = RagVecDB(dim=4)

db_a.batch_add(
    [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 1.0, 0.1]],
    [{"content": x} for x in ["aa", "bb", "cc"]],
)
# Create and add >>>>

# Save and load <<<<
db_a.save("test_db.local.bin")

db_a = RagVecDB.load("test_db.local.bin")

os.remove("test_db.local.bin")
# Save and load >>>>

multi_db = RagMultiVecDB([db_a, db_b])

result = multi_db.search([1.0, 0.0, 0.0, 0.0], 2)

for idx, metadata in enumerate(result):
    print(idx, metadata["content"])

assert result[0]["content"] == "a" and result[1]["content"] == "aa", "Test failed"

print("Test passed")
