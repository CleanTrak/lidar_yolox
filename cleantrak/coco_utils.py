import pycocotools.coco


def get_labels_list_from_coco(coco: pycocotools.coco.COCO) -> list[str]:
    cat_map = dict()
    for cat in coco.cats.values():
        id_ = cat["id"]
        assert id_ not in cat_map, f"Internal error, id {id_} found twice in coco categories"
        cat_map[id_] = cat["name"]
    cat_list = []
    for i, (id_, value) in enumerate(sorted(cat_map.items())):
        assert id_ == i + 1, "Internal error, unexpected input. Expected categories' ids consequent starting with 1"
        cat_list.append(value)
    return cat_list
