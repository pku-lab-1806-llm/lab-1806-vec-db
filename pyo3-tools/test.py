from lab_1806_vec_db import RagVecDB


def t(content: str) -> dict[str, str]:
    return {"content": content}


db = RagVecDB(dim=4)

db.add([1.0, 0.0, 0.0, 0.0], t("a"))
db.add([1.0, 0.0, 0.0, 0.1], t("aa"))

db.add([0.0, 1.0, 0.0, 0.0], t("b"))
db.add([0.0, 1.0, 0.0, 0.1], t("bb"))

db.add([0.0, 0.0, 1.0, 0.0], t("c"))
db.add([0.0, 0.0, 1.0, 0.1], t("cc"))

for idx, metadata in enumerate(db.search([1.0, 0.0, 0.0, 0.0], 2)):
    print(idx, metadata["content"])
