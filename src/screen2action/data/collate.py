"""Small typed collate helpers for command batches."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from screen2action.data.tokenizer import VocabularyTokenizer


def collate_commands(
    commands: Iterable[str],
    tokenizer: VocabularyTokenizer,
    *,
    max_length: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded command IDs and valid-token masks."""

    encoded = tokenizer.encode_batch(commands, max_length=max_length)
    ids = torch.tensor(encoded, dtype=torch.long)
    return ids, ids != tokenizer.pad_id
