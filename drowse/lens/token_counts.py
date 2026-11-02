"""Token counts for the S read's mask: the catalog's, else the model's corpus counts.

An S read ranks every token its mask keeps (:func:`~drowse.lens.footprint.token_mask`),
and a token the model almost never saw in text has an undertrained embedding
that tops early S reads.  The mask's counts come from, in order:

1. a bound neuron-lens catalog (``drowse unit fetch``): its activation
   corpora's counts (:attr:`~drowse.units.catalog.UnitCatalog.token_counts`),
   kept at :data:`~drowse.lens.footprint.MIN_TOKEN_COUNT`;
2. the model's **corpus counts**, a derived cache this module owns:
   ``models/<id>/tokens/counts.safetensors`` (``counts``, ``[ids]`` int64)
   with ``counts.json`` (:data:`TOKEN_COUNTS_FORMAT`, the corpus spec, the
   documents and tokens counted, the threshold, the vocabulary size and the
   counting tokenizer's :func:`tokenizer_fingerprint`; a session whose
   tokenizer has another fingerprint does not read them and says so).
   ``drowse lens counts MODEL`` writes it, streaming the S lens's recorded
   corpus (a local fit's or a fetched lens's ``corpus_spec``; the default lens
   corpus otherwise) through the tokenizer alone (no model load), up to
   :data:`DEFAULT_COUNT_DOCUMENTS` documents.  A lens fit's own documents are
   too few for a count (64 by default), so the count is its own pass over
   more of the same corpus, and it lives with the model, not in a lens pack;
3. none: the tokenizer's rules alone, surfaced wherever the mask is
   described (:func:`~drowse.lens.footprint.mask_detail`).

A record of another format, model or vocabulary size is not read (count
again); there is no migration.  :class:`TokenCountSource` is the session's
view: it reopens when either file's on-disk identity changes, so a ``unit
fetch`` or a ``lens counts`` beside a running server reaches it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from drowse.base.atomic import artifact_lock, write_json_atomic
from drowse.base.paths import model_dir
from drowse.lens.footprint import TokenCounts, count_threshold, mask_detail
from drowse.lens.key import DEFAULT_COUNT_DOCUMENTS

if TYPE_CHECKING:
    from drowse.units.catalog import UnitCatalog

_log = logging.getLogger(__name__)

#: ``counts.json``'s format; another is not read.
TOKEN_COUNTS_FORMAT = 1
#: The fewest tokens a count may cover: below it the mask would drop most of
#: the vocabulary (a 1M-token count keeps 62% of neuron-lens's kept set).
MIN_COUNT_TOKENS = 1_000_000
_COUNTS_FILE = "counts.safetensors"
_RECORD_FILE = "counts.json"


def counts_dir(model_id: str) -> Path:
    """``$DROWSE_HOME/models/<safe id>/tokens``."""
    return model_dir(model_id) / "tokens"


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """The identity counts are indexed by: the vocabulary size and a digest
    of every ``(piece, id)`` pair (added tokens included)."""
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: (item[1], item[0]))
    digest = hashlib.sha256(json.dumps(vocab, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"{len(tokenizer)}:{digest[:16]}"


def read_corpus_counts_record(model_id: str) -> dict[str, Any] | None:
    """The model's corpus-counts record when it is current, else ``None``."""
    path = counts_dir(model_id) / _RECORD_FILE
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(record, dict)
        or record.get("format") != TOKEN_COUNTS_FORMAT
        or record.get("model_id") != model_id
        or not (counts_dir(model_id) / _COUNTS_FILE).is_file()
    ):
        return None
    return record


def count_tokens(
    tokenizer: Any, documents: Iterable[str], *, batch: int = 256,
) -> tuple[torch.Tensor, int, int]:
    """``(counts [len(tokenizer)] int64, documents, tokens)`` over
    ``documents``, each tokenized whole without special tokens."""
    size = len(tokenizer)
    counts = torch.zeros(size, dtype=torch.int64)
    n_docs = n_tokens = 0
    chunk: list[str] = []

    def flush() -> None:
        nonlocal n_tokens
        if not chunk:
            return
        ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        flat = torch.tensor([i for row in ids for i in row], dtype=torch.int64)
        flat = flat[(flat >= 0) & (flat < size)]
        counts.add_(torch.bincount(flat, minlength=size))
        n_tokens += int(flat.numel())
        chunk.clear()

    for text in documents:
        if not text:
            continue
        chunk.append(text)
        n_docs += 1
        if len(chunk) >= batch:
            flush()
    flush()
    return counts, n_docs, n_tokens


def write_corpus_counts(
    model_id: str,
    counts: torch.Tensor,
    *,
    corpus: str,
    documents: int,
    tokens: int,
    tokenizer: str,
) -> dict[str, Any]:
    """Store the model's corpus counts (atomic per file, under the folder's
    artifact lock); returns the record.  ``tokenizer`` is the counting
    tokenizer's :func:`tokenizer_fingerprint`.  Fewer than
    :data:`MIN_COUNT_TOKENS` tokens is a :class:`~drowse.errors.RequestError`."""
    from safetensors.torch import save_file

    from drowse.errors import RequestError

    if int(tokens) < MIN_COUNT_TOKENS:
        raise RequestError(
            f"the corpus gave {tokens:,} tokens; a token count needs at least "
            f"{MIN_COUNT_TOKENS:,} (stream more documents)"
        )
    target = counts_dir(model_id)
    target.mkdir(parents=True, exist_ok=True)
    record = {
        "format": TOKEN_COUNTS_FORMAT,
        "model_id": model_id,
        "corpus": corpus,
        "documents": int(documents),
        "tokens": int(tokens),
        "min_count": count_threshold(int(tokens)),
        "vocabulary": int(counts.numel()),
        "tokenizer": tokenizer,
        "counted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with artifact_lock(target):
        staged = target / f".{_COUNTS_FILE}.tmp"
        save_file({"counts": counts.to(torch.int64).contiguous()}, str(staged))
        staged.replace(target / _COUNTS_FILE)
        write_json_atomic(target / _RECORD_FILE, record)
    return record


def remove_corpus_counts(model_id: str) -> bool:
    """Drop the model's corpus counts; whether there were any."""
    target = counts_dir(model_id)
    with artifact_lock(target):
        found = False
        for name in (_RECORD_FILE, _COUNTS_FILE):
            path = target / name
            if path.exists():
                path.unlink()
                found = True
        return found


def load_corpus_counts(model_id: str, *, tokenizer: str | None = None) -> TokenCounts | None:
    """The model's corpus counts as :class:`TokenCounts`, or ``None``;
    ``tokenizer`` (a :func:`tokenizer_fingerprint`), when given, must be the
    one the counts were taken with (another tokenizer indexes other ids)."""
    from safetensors.torch import load_file

    record = read_corpus_counts_record(model_id)
    if record is None:
        return None
    if tokenizer is not None and record.get("tokenizer") != tokenizer:
        _log.warning(
            "the corpus counts for %s were taken with another tokenizer (%s, "
            "this session's is %s); the S mask uses the tokenizer's rules alone "
            "(count again with `drowse lens counts`)",
            model_id, record.get("tokenizer"), tokenizer,
        )
        return None
    counts = load_file(str(counts_dir(model_id) / _COUNTS_FILE))["counts"]
    if int(counts.numel()) != int(record.get("vocabulary", -1)):
        return None
    return TokenCounts(
        counts=counts,
        source="corpus",
        detail=(
            f"corpus counts over {int(record['tokens']):,} tokens of "
            f"{record.get('corpus')}"
        ),
        min_count=int(record["min_count"]),
    )


_Identity = tuple[int, int, int] | None


def _identity(path: Path) -> _Identity:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)


class TokenCountSource:
    """The session's token counts: the bound catalog's, else the model's
    corpus counts, else ``None`` (the tokenizer mask).

    :meth:`get` returns the same :class:`TokenCounts` object while neither
    source changed on disk (the catalog binding's identity is its own; the
    corpus counts' is ``counts.json``'s), so a consumer can key a cache on it.
    Disk only.
    """

    def __init__(
        self, model_id: str, catalog: Callable[[], "UnitCatalog | None"],
        *, tokenizer: Any = None,
    ) -> None:
        self.model_id = model_id
        self._catalog = catalog
        # The session's tokenizer: corpus counts taken with another are not
        # read (its fingerprint is taken once, on first use).
        self._tokenizer = tokenizer
        self._tokenizer_id: str | None = None
        self._lock = threading.Lock()
        self._corpus_identity: _Identity = None
        self._corpus: TokenCounts | None = None
        self._corpus_loaded = False

    def _corpus_counts(self) -> TokenCounts | None:
        identity = _identity(counts_dir(self.model_id) / _RECORD_FILE)
        with self._lock:
            if not self._corpus_loaded or identity != self._corpus_identity:
                if self._tokenizer is not None and self._tokenizer_id is None:
                    self._tokenizer_id = tokenizer_fingerprint(self._tokenizer)
                self._corpus = None if identity is None else load_corpus_counts(
                    self.model_id, tokenizer=self._tokenizer_id,
                )
                self._corpus_identity = identity
                self._corpus_loaded = True
            return self._corpus

    def get(self) -> TokenCounts | None:
        catalog = self._catalog()
        if catalog is not None:
            counts = catalog.token_counts
            if counts is not None:
                return counts
        return self._corpus_counts()

    def describe(self) -> dict[str, str]:
        """``{"source", "detail"}`` of the mask these counts make (disk only)."""
        source, detail = mask_detail(self.get())
        return {"source": source, "detail": detail}


__all__ = [
    "DEFAULT_COUNT_DOCUMENTS",
    "MIN_COUNT_TOKENS",
    "TOKEN_COUNTS_FORMAT",
    "TokenCountSource",
    "count_tokens",
    "counts_dir",
    "load_corpus_counts",
    "read_corpus_counts_record",
    "remove_corpus_counts",
    "tokenizer_fingerprint",
    "write_corpus_counts",
]
