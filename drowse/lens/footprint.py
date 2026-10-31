"""Footprint z: the S-family read of a residual over the token vocabulary.

An S (or K) lens holds ``K_L = E[Σ_{s≤t} ∂h_L(t)/∂e(s)]``.  A token ``v``'s
**footprint** at layer ``L`` is ``K_L ẽ_v``, the residual direction the token
leaves there, with ``ẽ_v`` its centred embedding as it enters the residual
(``scale · (W_E[v] − mean)``; Gemma's scale is ``√d``).  Reading a residual
``h`` asks which tokens' footprints it carries: pull ``h`` back to the
embedding basis, ``u_L = K_Lᵀ h``, and score every token by ``u_L · ẽ_v``.

The score's scale follows ``h`` and the layer, so it is reported as a z over
the vocabulary::

    z_L(v) = u_L · ẽ_v / sqrt(u_Lᵀ Σ_E u_L)

with ``Σ_E`` the covariance of ``ẽ`` over the token mask.  Since ``ẽ`` is
centred over the mask, ``u_Lᵀ Σ_E u_L`` is exactly the variance of the scores
across the masked vocabulary, so ``z`` is each score's vocabulary z-score: a
per-vector affine of the raw score, whose top-k is the raw read's.  ``Σ_E``
makes a one-token read (a probe, a gate) cost ``O(d²)`` per layer instead of a
pass over the vocabulary.

**The token mask** is decided by the tokenizer alone.  It drops special
tokens, every added token (special or not: chat and tool control pieces such
as Qwen3's ``<think>``, ``<tool_call>`` and ``<|fim_*|>``, Gemma's reserved
``<unusedN>``), byte-fallback pieces (a vocabulary piece spelled ``<0xNN>``,
including the ASCII ones that decode to a control character, and any token
that decodes alone to U+FFFD, a partial UTF-8 sequence; out of distribution
on their own and on top of many early reads) and embedding rows past the
tokenizer (padding).  Masked tokens never rank and do not enter the mean or
``Σ_E``.

Relation to neuron-lens (``scripts/08_reads.py``, ``neuron_lens/catalog.py``):
the centred embedding and the read are the same (``s_tops`` over ``emb −
emb.mean(0)`` with Gemma's ``× √d``), so top-k matches its raw ``read_S`` /
``write_S`` wherever both keep a token.  The differences:

- Its kept set is the tokens seen at least five times in its activation
  corpora, which drops control, byte and reserved pieces by never seeing
  them.  A lens corpus is too small for a count, so drowse masks by the
  tokenizer: the two sets differ on rare base-vocabulary tokens (drowse keeps
  them) and on added tokens a corpus does contain (drowse drops them all,
  Gemma 2's added whitespace runs among them).
- It reports the raw score of a unit covector where drowse reports the
  vocabulary z (same order, a scale comparable across vectors and layers).
- Its ``_z`` reads are layer-contrastive (z across a layer's neurons), which
  needs a population of vectors, not one.
- It scores in fp16; drowse in fp32.

Torch only: the instrument builds :class:`TokenEmbeddings` once per model and
every S read, probe, gate, steering atom and the ``tokens-in`` dictionary use
it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

#: Rows per chunk when a pass runs over the vocabulary (an fp32 ``[chunk, d]``
#: block of the embedding).
VOCAB_CHUNK = 8192
#: The smallest score variance a z divides by: a zero read (``u = 0``) gives
#: ``z = 0`` everywhere rather than NaN.
_MIN_VARIANCE = 1e-30
REPLACEMENT_CHARACTER = "�"
#: A sentencepiece byte-fallback piece as the vocabulary spells it.
_BYTE_PIECE = re.compile(r"<0x[0-9A-Fa-f]{2}>")


def token_mask(tokenizer: Any, rows: int) -> torch.Tensor:
    """``[rows]`` bool: the tokens an S read ranks.

    Drops the tokenizer's special ids (``all_special_ids``), every added
    token (``added_tokens_decoder``, special or not: control and reserved
    pieces such as Qwen3's ``<think>``/``<tool_call>`` and Gemma's
    ``<unusedN>``), byte-fallback pieces (a vocabulary piece spelled
    ``<0xNN>``, whatever it decodes to, and any id that decodes alone to a
    string containing U+FFFD, a partial UTF-8 sequence), and embedding rows
    at or past the tokenizer's length (padding rows a model reserves beyond
    its vocabulary).
    """
    size = rows
    try:
        size = min(rows, len(tokenizer))
    except TypeError:
        pass
    keep = torch.zeros(rows, dtype=torch.bool)
    keep[:size] = True
    dropped = set(int(i) for i in getattr(tokenizer, "all_special_ids", ()) or ())
    dropped.update(
        int(i) for i in (getattr(tokenizer, "added_tokens_decoder", None) or {})
    )
    for token_id in dropped:
        if 0 <= token_id < rows:
            keep[token_id] = False
    ids = list(range(size))
    batch_decode: Any = getattr(tokenizer, "batch_decode", None)
    pieces: list[Any] = (
        list(batch_decode([[i] for i in ids])) if batch_decode is not None
        else [tokenizer.decode([i]) for i in ids]
    )
    to_tokens: Any = getattr(tokenizer, "convert_ids_to_tokens", None)
    spelled: list[Any] = list(to_tokens(ids)) if to_tokens is not None else [None] * size
    for token_id, piece, raw in zip(ids, pieces, spelled, strict=True):
        if REPLACEMENT_CHARACTER in str(piece) or (
            isinstance(raw, str) and _BYTE_PIECE.fullmatch(raw)
        ):
            keep[token_id] = False
    return keep


@dataclass(frozen=True, eq=False)
class TokenEmbeddings:
    """The token embeddings as they enter the residual, centred over the mask.

    ``weight`` is the model's own ``[V, d]`` embedding (shared, model dtype,
    never copied); ``scale`` the factor the model applies on lookup.  ``keep``,
    ``mean`` (``[d]``) and ``cov`` (``Σ_E``, ``[d, d]``) live on the weight's
    device, the statistics in fp32.
    """

    weight: torch.Tensor
    scale: float
    keep: torch.Tensor
    mean: torch.Tensor
    cov: torch.Tensor
    n_kept: int

    @classmethod
    def build(
        cls, weight: torch.Tensor, scale: float, keep: torch.Tensor,
        *, chunk: int = VOCAB_CHUNK,
    ) -> "TokenEmbeddings":
        """Mean and covariance of ``scale · W[keep]`` in fp32, two chunked passes."""
        device = weight.device
        keep = keep.to(device=device, dtype=torch.bool)
        n_kept = int(keep.sum())
        if n_kept < 2:
            raise ValueError("the token mask keeps fewer than two tokens")
        d = int(weight.shape[1])
        total = torch.zeros(d, dtype=torch.float32, device=device)
        for start in range(0, int(weight.shape[0]), chunk):
            rows = weight[start:start + chunk].detach().to(torch.float32)
            total += rows[keep[start:start + chunk]].sum(dim=0)
        mean = total * (float(scale) / n_kept)
        cov = torch.zeros(d, d, dtype=torch.float32, device=device)
        for start in range(0, int(weight.shape[0]), chunk):
            rows = weight[start:start + chunk].detach().to(torch.float32)
            centred = rows[keep[start:start + chunk]] * float(scale) - mean
            cov += centred.T @ centred
        cov /= n_kept
        return cls(weight, float(scale), keep, mean, cov, n_kept)

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def vocab(self) -> int:
        return int(self.weight.shape[0])

    @property
    def d_emb(self) -> int:
        return int(self.weight.shape[1])

    def centred(self, token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """``ẽ_v = scale · W_E[v] − mean`` for each id, fp32 ``[k, d]``."""
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).reshape(-1)
        return self.weight.index_select(0, ids).detach().to(torch.float32) * self.scale - self.mean

    def variance(self, covectors: torch.Tensor) -> torch.Tensor:
        """``uᵀ Σ_E u`` per row of ``[..., d]`` (the score variance over the mask)."""
        u = covectors.to(device=self.device, dtype=torch.float32)
        return ((u @ self.cov) * u).sum(dim=-1)

    def scores(self, covectors: torch.Tensor, *, chunk: int = VOCAB_CHUNK) -> torch.Tensor:
        """Raw scores ``u · ẽ_v`` for every token, ``[n, V]`` fp32.

        Masked tokens score ``-inf``, so they never rank and a nonnegative
        pursuit never selects them.
        """
        u = covectors.to(device=self.device, dtype=torch.float32).reshape(-1, self.d_emb)
        offset = u @ self.mean                                       # [n]
        out = torch.empty(u.shape[0], self.vocab, dtype=torch.float32, device=self.device)
        for start in range(0, self.vocab, chunk):
            rows = self.weight[start:start + chunk].detach().to(torch.float32)
            out[:, start:start + chunk] = (u @ rows.T) * self.scale
        out -= offset[:, None]
        out[:, ~self.keep] = -torch.inf
        return out

    def z(self, covectors: torch.Tensor) -> torch.Tensor:
        """Footprint z of every token, ``[n, V]``: the scores over their std."""
        u = covectors.to(device=self.device, dtype=torch.float32).reshape(-1, self.d_emb)
        return self.scores(u) / self.variance(u).clamp_min(_MIN_VARIANCE).sqrt()[:, None]

    def z_of(self, covectors: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Footprint z of selected tokens only, ``[n, k]``, in ``O(d²)`` per row."""
        u = covectors.to(device=self.device, dtype=torch.float32).reshape(-1, self.d_emb)
        rows = self.centred(token_ids)                               # [k, d]
        return (u @ rows.T) / self.variance(u).clamp_min(_MIN_VARIANCE).sqrt()[:, None]


def pullback_rows(matrices: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    """Batched ``u_L = K_Lᵀ h_L`` for ``[L, d, d_emb]`` matrices and ``[L, d]`` rows."""
    hidden = residuals.to(device=matrices.device, dtype=torch.float32)
    return torch.bmm(hidden.unsqueeze(1), matrices).squeeze(1)


def depth_moments_positive(
    values: torch.Tensor, depths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth center and spread per column of ``[L, k]``, mass ``max(z, 0)``.

    ``depths`` is ``[L, 1]``.  A column with no positive mass reads center 0
    and spread 0.
    """
    mass = values.clamp_min(0.0)
    total = mass.sum(dim=0).clamp_min(1e-12)
    center = (mass * depths).sum(dim=0) / total
    var = (mass * (depths - center.unsqueeze(0)) ** 2).sum(dim=0) / total
    return center, var.clamp_min(0.0).sqrt()


__all__ = [
    "TokenEmbeddings",
    "VOCAB_CHUNK",
    "depth_moments_positive",
    "pullback_rows",
    "token_mask",
]
