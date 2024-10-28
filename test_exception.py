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
# - Press Ctrl+C to interrupt the process

cmd = input("Type `raise` to raise an exception: ")
if cmd == "":
    exit(0)  # Exit normally
elif cmd == "raise":
    raise Exception("Deliberate exception")

# File appears before the program exits in all cases, even with KeyboardInterrupt.
