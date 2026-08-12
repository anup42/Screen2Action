"""Small deterministic tokenizer used by the CPU pipeline.

The paper does not publish its tokenizer or vocabulary. This implementation is
an intentionally simple, train-only vocabulary boundary that can later be
replaced by SentencePiece/BPE without changing model interfaces.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class VocabularyTokenizer:
    """Deterministic whitespace/wordpiece-like tokenizer."""

    token_to_id: Mapping[str, int]
    pad_token: str = "<pad>"
    unknown_token: str = "<unk>"
    begin_token: str = "<bos>"
    end_token: str = "<eos>"

    def __post_init__(self) -> None:
        required = (self.pad_token, self.unknown_token, self.begin_token, self.end_token)
        missing = [token for token in required if token not in self.token_to_id]
        if missing:
            raise ValueError(f"tokenizer vocabulary is missing special tokens: {missing}")
        ids = list(self.token_to_id.values())
        if len(ids) != len(set(ids)) or min(ids, default=0) < 0:
            raise ValueError("tokenizer IDs must be unique and non-negative")

    @property
    def vocab_size(self) -> int:
        """Return the embedding vocabulary size."""

        return max(self.token_to_id.values(), default=-1) + 1

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unknown_id(self) -> int:
        return self.token_to_id[self.unknown_token]

    @property
    def begin_id(self) -> int:
        return self.token_to_id[self.begin_token]

    @property
    def end_id(self) -> int:
        return self.token_to_id[self.end_token]

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Tokenize text with Unicode case folding and deterministic punctuation."""

        if not text.strip():
            return ()
        return tuple(_TOKEN_PATTERN.findall(text.casefold()))

    def encode(self, text: str, max_length: int = 64) -> tuple[int, ...]:
        """Encode and pad one command to a fixed length."""

        if max_length < 2:
            raise ValueError("max_length must allow BOS and EOS")
        pieces = self.tokenize(text)
        pieces = pieces[: max_length - 2]
        ids = [self.begin_id]
        ids.extend(self.token_to_id.get(piece, self.unknown_id) for piece in pieces)
        ids.append(self.end_id)
        ids.extend([self.pad_id] * (max_length - len(ids)))
        return tuple(ids)

    def encode_batch(
        self, texts: Iterable[str], max_length: int = 64
    ) -> tuple[tuple[int, ...], ...]:
        """Encode a batch while preserving input order."""

        return tuple(self.encode(text, max_length=max_length) for text in texts)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe vocabulary snapshot."""

        return {
            "token_to_id": dict(self.token_to_id),
            "pad_token": self.pad_token,
            "unknown_token": self.unknown_token,
            "begin_token": self.begin_token,
            "end_token": self.end_token,
        }

    @classmethod
    def from_corpus(cls, corpus: Iterable[str], vocab_size: int = 16_384) -> VocabularyTokenizer:
        """Build a deterministic vocabulary from training-only text."""

        if vocab_size < 4:
            raise ValueError("vocab_size must leave room for four special tokens")
        counts = Counter(
            token for text in corpus for token in _TOKEN_PATTERN.findall(text.casefold())
        )
        specials = ("<pad>", "<unk>", "<bos>", "<eos>")
        token_to_id = {token: index for index, token in enumerate(specials)}
        available = vocab_size - len(specials)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:available]
        token_to_id.update(
            {token: index + len(specials) for index, (token, _) in enumerate(ranked)}
        )
        return cls(token_to_id)

    @classmethod
    def default(cls) -> VocabularyTokenizer:
        """Return the minimal vocabulary for an empty or synthetic corpus."""

        return cls.from_corpus(())
