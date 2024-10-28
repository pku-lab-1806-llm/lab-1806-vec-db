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
