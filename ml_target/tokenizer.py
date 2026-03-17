"""OpenAI CLIP BPE tokenizer reimplemented with stdlib `re` only (no `regex` dep)."""

import gzip
import re
from typing import Dict, List, Tuple

import numpy as np


def _bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: Tuple[str, ...]) -> set:
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class SimpleTokenizer:
    """CLIP BPE tokenizer using the official merges file (bpe_simple_vocab_16e6.txt.gz).

    Uses only stdlib `re` — no third-party `regex` dependency.
    Works well for typical English smart-search queries.
    """

    def __init__(self, bpe_path: str):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges = self._load_merges(bpe_path)

        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for a, b in merges:
            vocab.append(a + b)
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = {tok: i for i, tok in enumerate(vocab)}
        self.decoder = {i: tok for tok, i in self.encoder.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.cache: Dict[str, str] = {}

        self.pat = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|[A-Za-z]+|\d+|[^\sA-Za-z\d]+",
            re.IGNORECASE,
        )

        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    def _load_merges(self, bpe_path: str) -> List[Tuple[str, str]]:
        with gzip.open(bpe_path, "rt", encoding="utf-8") as f:
            lines = f.read().splitlines()
        merges = [
            tuple(line.split())
            for line in lines[1:]
            if line and not line.startswith("#")
        ]
        return [(a, b) for a, b in merges[:48894]]

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, 1e10))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> List[int]:
        text = text.strip().lower()
        bpe_tokens: List[int] = []
        for token in re.findall(self.pat, text):
            token_bytes = token.encode("utf-8")
            token_trans = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe = self.bpe(token_trans).split(" ")
            bpe_tokens.extend(self.encoder.get(b, 0) for b in bpe)
        return bpe_tokens

    def tokenize(self, text: str, context_length: int = 77) -> np.ndarray:
        tokens = [self.sot] + self.encode(text) + [self.eot]
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = self.eot
        else:
            tokens = tokens + [0] * (context_length - len(tokens))
        return np.array(tokens, dtype=np.int32)
