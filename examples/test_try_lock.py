from lab_1806_vec_db import VecDB

db1 = VecDB("./tmp/vec_db")
db2 = VecDB("./tmp/vec_db")  # RuntimeError: Failed to acquire lock for VecDBManager
