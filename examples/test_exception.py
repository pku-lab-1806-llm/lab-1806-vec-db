import os

from lab_1806_vec_db import VecDB

# uv sync --reinstall-package lab_1806_vec_db
# uv run -m examples.test_exception

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
