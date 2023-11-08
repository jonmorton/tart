import itertools
from typing import Iterable

size_ordinary = 256

SPECIAL_TOKENS = {
    "<!eod!>": size_ordinary,
    "<!pad!>": size_ordinary + 1,
    "<!unk!>": size_ordinary + 2,
}

size = size_ordinary + len(SPECIAL_TOKENS)

SPECIAL_TO_STR = {v: k for k, v in SPECIAL_TOKENS.items()}
SPECIAL_IDS = list(SPECIAL_TOKENS.values())

EOD = SPECIAL_TOKENS["<!eod!>"]


def encode(string: str):
    return list(string.encode("utf-8"))


def encode_doc(doc):
    t = encode(doc)
    t.append(EOD)
    return t


def decode(tokens: Iterable[int]):
    bytes_ = bytearray()
    for t in tokens:
        if t in SPECIAL_IDS:
            bytes_.extend(SPECIAL_TO_STR[t].encode("utf-8"))
        else:
            bytes_.append(t)
    return bytes_.decode("utf-8", errors="ignore")
