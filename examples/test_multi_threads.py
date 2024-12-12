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
