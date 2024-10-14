import os

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

os.remove("test_db.local.bin")

result = loaded_db.search([1.0, 0.0, 0.0, 0.0], 2)

for idx, metadata in enumerate(result):
    print(idx, metadata["content"])

assert result[0]["content"] == "a" and result[1]["content"] == "aa", "Test failed"

print("Test passed")
