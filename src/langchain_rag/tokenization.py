import re


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.casefold())
