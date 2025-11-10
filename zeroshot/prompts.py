from typing import List, Dict

TEMPLATES_ALL = [
    "a photo of {name}.",
    "a picture of {name}.",
    "herbarium specimen of {name}.",
    "field photo of {name}.",
    "plant species {name}.",
    "macro photo of {name}.",
]

def build_prompts(class_ids: List[str], id2name: Dict[str, str], templates: list[str]) -> list[str]:
    out: list[str] = []
    for cid in class_ids:
        name = id2name.get(cid, cid)
        for t in templates:
            out.append(t.format(name=name))
    return out
