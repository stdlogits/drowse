"""The read of a vector at a place: one R-family and one S-family half.

The four-lens rule.  The **site** comes from where the object sits: ``out``
reads a block's output residual, an MLP neuron's write and an attention head's
inputs (at ``L − 1``); ``mid`` reads a neuron's read direction (the MLP's
input) and a head's write.  The **kind** is a depth question: the S family
(K/S) is the faithful read early, the R family (J/R) late, with both shown
inside a crossover band (:mod:`drowse.lens.depth`).

So one read returns both halves, each carrying its depth weight and the lens
it read through:

- **R** (``unit="probability"``): ``softmax(W_U · norm(J_L x))`` through the
  model's own final path (norm, projection, logit scale, softcap), the top-k
  per-layer probabilities ``p_L(v)`` every lens readout reports.
- **S** (``unit="footprint_z"``): the footprint z of every masked token,
  ``u · ẽ_v / sqrt(uᵀ Σ_E u)`` with ``u = K_Lᵀ x`` (:mod:`drowse.lens.footprint`).

A half with no lens, no lens at this site, or no matrix at this layer comes
back absent with the reason; it is never filled from the other site or the
other family.  The weights are the depth rule's ``(w_S, w_R)`` at the layer
whether or not a half is present, so a consumer can fade a light half or say
that a heavy one is missing.

This is the primitive token anatomy (residual, attention write, MLP write)
and the unit views ("writes", "fires on") read through;
:meth:`~drowse.lens.instrument.LensInstrument.read` is its session face.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from drowse.lens.depth import LensBand, kind_weights, layer_depth
from drowse.lens.footprint import TokenEmbeddings
from drowse.lens.jlens import JacobianLens, LensNotFittedError, topk_probabilities
from drowse.lens.key import LENS_SITES

LensFamily = Literal["R", "S"]
ReadUnit = Literal["probability", "footprint_z"]


@dataclass(frozen=True)
class TokenScore:
    """One ranked token: its id, decoded piece and score in the half's unit."""

    token_id: int
    token: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {"token_id": self.token_id, "token": self.token, "value": round(self.value, 6)}


@dataclass(frozen=True)
class FamilyRead:
    """One family's half of a read.

    ``lens`` labels the lens read through (``"R:mid"``, ``"J:out"``,
    ``"S:out"``) and ``absent`` is ``None`` when the half was read; otherwise
    ``lens`` may be ``None`` (no lens of the family is selected), ``tokens``
    is empty and ``absent`` says why.  ``weight`` is the depth rule's weight
    for this family at the layer either way.
    """

    family: LensFamily
    weight: float
    unit: ReadUnit
    lens: str | None = None
    tokens: tuple[TokenScore, ...] = ()
    absent: str | None = None

    @property
    def present(self) -> bool:
        return self.absent is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "weight": round(self.weight, 6),
            "unit": self.unit,
            "lens": self.lens,
            "tokens": [token.to_dict() for token in self.tokens],
            "absent": self.absent,
        }


@dataclass(frozen=True)
class LensRead:
    """The read of one vector at ``(layer, site)``: its R and S halves.

    ``depth`` is ``layer / (n_layers − 1)``; ``r.weight + s.weight == 1``.
    """

    layer: int
    site: str
    depth: float
    r: FamilyRead
    s: FamilyRead = field(repr=False)

    @property
    def halves(self) -> tuple[FamilyRead, FamilyRead]:
        """``(s, r)``: shallow-faithful first, as the depth rule orders them."""
        return (self.s, self.r)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "site": self.site,
            "depth": round(self.depth, 6),
            "r": self.r.to_dict(),
            "s": self.s.to_dict(),
        }


#: ``(logits) -> logits``: the model's final path from the last residual
#: (:class:`~drowse.model.readout.LogitReadout`).
Readout = Callable[[torch.Tensor], torch.Tensor]


def _matrix(lens: JacobianLens, layer: int, site: str, device: torch.device) -> torch.Tensor:
    return lens.matrix(layer, site).to(device=device, dtype=torch.float32)


def read_r(
    vectors: torch.Tensor,
    matrix: torch.Tensor,
    readout: Readout,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R-family top-k for rows ``[n, d]``: ``(p [n, k], ids [n, k])``.

    ``matrix`` is ``J_L`` at the site; rows transport as ``x @ J_Lᵀ`` and the
    readout supplies the final norm and the vocabulary head.
    """
    transported = vectors.to(device=matrix.device, dtype=torch.float32) @ matrix.T
    logits = readout(transported)
    return topk_probabilities(logits, min(int(k), int(logits.shape[-1])))


def read_s(
    vectors: torch.Tensor,
    matrix: torch.Tensor,
    embeddings: TokenEmbeddings,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """S-family top-k for rows ``[n, d]``: ``(z [n, k], ids [n, k])``.

    ``matrix`` is ``K_L`` at the site; rows pull back as ``u = x @ K_L``.
    """
    u = vectors.to(device=matrix.device, dtype=torch.float32) @ matrix
    z = embeddings.z(u)
    return z.topk(min(int(k), embeddings.n_kept), dim=-1)


def read_vectors(
    vectors: torch.Tensor,
    layer: int,
    site: str,
    *,
    r_lens: JacobianLens | None,
    s_lens: JacobianLens | None,
    readout: Callable[[], Readout],
    embeddings: Callable[[], TokenEmbeddings],
    decode: Callable[[int], str],
    n_layers: int,
    device: torch.device,
    k: int = 8,
    band: LensBand | None = None,
    no_r_lens: str = "no R-family lens is selected for this model",
    no_s_lens: str = "no S-family lens is selected for this model",
) -> list[LensRead]:
    """Read every row of ``vectors`` (``[d]`` or ``[n, d]``) at ``(layer, site)``.

    ``readout`` and ``embeddings`` are thunks, so a half that is absent never
    pays for the vocabulary head or the embedding statistics.  The rows of
    each half leave the device in one transfer.
    """
    if site not in LENS_SITES:
        raise ValueError(f"site must be one of {', '.join(LENS_SITES)}, got {site!r}")
    if not 0 <= int(layer) < int(n_layers):
        raise ValueError(f"layer {layer} is outside the model's {n_layers} layers")
    if vectors.ndim not in (1, 2):
        raise ValueError(f"read takes a [d] vector or [n, d] rows, got {tuple(vectors.shape)}")
    for lens, side in ((r_lens, "output"), (s_lens, "input")):
        if lens is not None and lens.key.side != side:
            raise ValueError(f"a {lens.kind} lens cannot read the {side}-family half")
    rows = vectors.reshape(1, -1) if vectors.ndim == 1 else vectors
    w_s, w_r = kind_weights(n_layers, band)[int(layer)]
    depth = layer_depth(int(layer), int(n_layers))
    n = int(rows.shape[0])

    def half(
        family: LensFamily, lens: JacobianLens | None, weight: float, unit: ReadUnit,
        missing: str, run: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> list[FamilyRead]:
        if lens is None:
            return [FamilyRead(family, weight, unit, absent=missing)] * n
        label = lens.site_key(site).label
        try:
            matrix = _matrix(lens, int(layer), site, device)
        except LensNotFittedError as exc:
            return [FamilyRead(family, weight, unit, lens=label, absent=str(exc))] * n
        values, ids = run(matrix)
        host = torch.cat([values.float(), ids.float()], dim=0).detach().cpu()
        out: list[FamilyRead] = []
        for row in range(n):
            tokens = tuple(
                TokenScore(int(host[n + row, j]), decode(int(host[n + row, j])), float(host[row, j]))
                for j in range(int(values.shape[1]))
            )
            out.append(FamilyRead(family, weight, unit, lens=label, tokens=tokens))
        return out

    r_halves = half(
        "R", r_lens, w_r, "probability", no_r_lens,
        lambda matrix: read_r(rows, matrix, readout(), k=k),
    )
    s_halves = half(
        "S", s_lens, w_s, "footprint_z", no_s_lens,
        lambda matrix: read_s(rows, matrix, embeddings(), k=k),
    )
    return [
        LensRead(int(layer), site, depth, r=r_halves[row], s=s_halves[row])
        for row in range(n)
    ]


__all__ = [
    "FamilyRead",
    "LensFamily",
    "LensRead",
    "ReadUnit",
    "TokenScore",
    "read_r",
    "read_s",
    "read_vectors",
]
