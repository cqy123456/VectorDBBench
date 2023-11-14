from vectordb_bench.models import CaseType

passwordKeys = ["password", "api_key"]


def inputIsPassword(key: str) -> bool:
    return key.lower() in passwordKeys


def get_all_combinations(items: list[any]):
    k = len(items)
    flags = [2**j for j in range(k)]
    combinations = [[items[j] for j in range(k) if i & flags[j]] for i in range(2**k)]
    return combinations


colors = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]


def getColor(i: int):
    if i < len(colors):
        return colors[i]
    return "#000"
