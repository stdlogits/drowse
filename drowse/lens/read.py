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

**What is read** is part of the request (:data:`ReadOf`), and never has a
default.  A **residual** (the stream at a block's output or ``mid``) carries
the large common component every position shares, and the footprint of that
component swamps the position's own tokens, so the S half reads ``h − μ``,
``μ`` the neutral layer mean at the read's site (``out``: the monitor's and
whitener's centring mean; ``mid``: the neutral cache's mid means).  A
**direction** (a steering vector, a neuron or head write, an attention or MLP
write delta) has no such component and is read as given.  The R half is
never centred: the final norm already discards the common scale, and the
readout is the model's own.  A residual read with no mean to centre on raises
(the ``centre`` callable's error); it is never read uncentred.

A half with no lens, no lens at this site, or no matrix at this layer comes
back absent with the reason; it is never filled from the other site or the
other family.  One layer needs no matrix: an output kind's target is the last
block's output, so at that layer's ``out`` its transport is the identity
(``J = I`` exactly, under the standard and the RelP rules alike: no block lies
between the site and the target).  An output fit therefore stops a layer short
there (:func:`~drowse.lens.key.site_layers`), and the R half at the last
``out`` is the logit lens, labelled ``logit`` (:func:`output_transport`).  The weights are the depth rule's ``(w_S, w_R)`` at the layer
whether or not a half is present, so a consumer can fade a light half or say
that a heavy one is missing.

Every request of a batch is computed on the device and leaves it in **one**
transfer (:func:`plan_reads` → :class:`PendingReads`, which a caller may fold
into a larger packed transfer of its own).  Token anatomy (residual,
attention write, MLP write) and the unit views ("writes", "fires on") read
through this; :meth:`~drowse.lens.instrument.LensInstrument.read` is its
session face.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from drowse.lens.depth import LensBand, kind_weights, layer_depth
from drowse.lens.footprint import TokenEmbeddings
from drowse.lens.jlens import JacobianLens, LensNotFittedError, topk_probabilities
from drowse.lens.key import LENS_SITES, LensFamily

ReadUnit = Literal["probability", "footprint_z"]
#: What a read reads: a ``residual`` (its S half is centred on the neutral
#: mean at its site) or a ``direction`` (read as given).
ReadOf = Literal["residual", "direction"]
READ_OF: tuple[ReadOf, ...] = ("residual", "direction")


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
    for this family at the layer either way.  ``centred`` is True when the
    half read ``x − μ`` (the S half of a residual read).
    """

    family: LensFamily
    weight: float
    unit: ReadUnit
    lens: str | None = None
    tokens: tuple[TokenScore, ...] = ()
    absent: str | None = None
    centred: bool = False

    @property
    def present(self) -> bool:
        return self.absent is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "weight": round(self.weight, 6),
            "unit": self.unit,
            "lens": self.lens,
            "centred": self.centred,
            "tokens": [token.to_dict() for token in self.tokens],
            "absent": self.absent,
        }


@dataclass(frozen=True)
class LensRead:
    """The read of one vector at ``(layer, site)``: its R and S halves.

    ``depth`` is ``layer / (n_layers − 1)``; ``r.weight + s.weight == 1``.
    ``of`` says whether a residual or a direction was read.
    """

    layer: int
    site: str
    depth: float
    r: FamilyRead
    s: FamilyRead = field(repr=False)
    of: ReadOf = field(kw_only=True)

    @property
    def halves(self) -> tuple[FamilyRead, FamilyRead]:
        """``(s, r)``: shallow-faithful first, as the depth rule orders them."""
        return (self.s, self.r)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "site": self.site,
            "depth": round(self.depth, 6),
            "of": self.of,
            "r": self.r.to_dict(),
            "s": self.s.to_dict(),
        }


@dataclass(frozen=True)
class VectorRead:
    """One read request: ``rows`` (``[d]`` or ``[n, d]``) at ``(layer, site)``,
    read as a residual or a direction (no default: the caller says which)."""

    rows: torch.Tensor
    layer: int
    site: str
    of: ReadOf

    def __post_init__(self) -> None:
        if self.of not in READ_OF:
            raise ValueError(f"a read reads a residual or a direction, got of={self.of!r}")
        if self.site not in LENS_SITES:
            raise ValueError(f"site must be one of {', '.join(LENS_SITES)}, got {self.site!r}")
        if self.rows.ndim not in (1, 2):
            raise ValueError(f"read takes a [d] vector or [n, d] rows, got {tuple(self.rows.shape)}")

    @property
    def matrix_rows(self) -> torch.Tensor:
        return self.rows.reshape(1, -1) if self.rows.ndim == 1 else self.rows


#: ``(logits) -> logits``: the model's final path from the last residual
#: (:class:`~drowse.model.readout.LogitReadout`).
Readout = Callable[[torch.Tensor], torch.Tensor]
#: ``(site, layer) -> [d]``: the neutral mean a residual read centres its S
#: half on (raises when there is none).
Centre = Callable[[str, int], torch.Tensor]


#: The R half's lens label at the last block's ``out``, where the transport is
#: the identity and the read is the logit lens (:func:`output_transport`).
LOGIT_LENS = "logit"


def _matrix(lens: JacobianLens, layer: int, site: str, device: torch.device) -> torch.Tensor:
    return lens.matrix(layer, site).to(device=device, dtype=torch.float32)


def output_transport(
    lens: JacobianLens, layer: int, site: str, n_layers: int, device: torch.device,
) -> tuple[torch.Tensor | None, str]:
    """An output-kind (J/R) lens's transport at ``(layer, site)`` on ``device``
    and the label its read carries: the fitted ``J_L`` and the lens's label,
    or, at the last block's ``out``, ``None`` (the identity: that output is
    the lens's own target) and :data:`LOGIT_LENS`.  The identity needs a lens
    that holds the ``out`` site; anything else the lens lacks raises
    :class:`~drowse.lens.jlens.LensNotFittedError`, as :meth:`JacobianLens.matrix`
    does.  The one place a read meets the last layer: :func:`plan_reads` and
    the unit reads' population statistics both ask here."""
    if lens.key.family != "R":
        raise ValueError(f"a {lens.kind} lens has no output transport")
    if site == "out" and int(layer) == int(n_layers) - 1:
        lens.matrices("out")
        return None, LOGIT_LENS
    return _matrix(lens, int(layer), site, device), lens.site_key(site).label


def read_r(
    vectors: torch.Tensor,
    matrix: torch.Tensor | None,
    readout: Readout,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R-family top-k for rows ``[n, d]``: ``(p [n, k], ids [n, k])``.

    ``matrix`` is ``J_L`` at the site, or ``None`` for the identity (the last
    block's ``out``, :func:`output_transport`); rows transport as
    ``x @ J_Lᵀ`` and the readout supplies the final norm and the vocabulary
    head.
    """
    transported = vectors.to(dtype=torch.float32)
    if matrix is not None:
        transported = transported.to(device=matrix.device) @ matrix.T
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
    Centring is the caller's (:func:`plan_reads` subtracts the neutral mean
    from a residual before it gets here).
    """
    u = vectors.to(device=matrix.device, dtype=torch.float32) @ matrix
    z = embeddings.z(u)
    return z.topk(min(int(k), embeddings.n_kept), dim=-1)


@dataclass(frozen=True)
class _Half:
    """One half of one request: either read (``offset`` into the packed
    tensor, ``width`` = k) or absent."""

    family: LensFamily
    weight: float
    unit: ReadUnit
    lens: str | None
    centred: bool
    absent: str | None = None
    offset: int = 0
    width: int = 0


@dataclass(frozen=True)
class _Planned:
    request: VectorRead
    n: int
    depth: float
    r: _Half
    s: _Half


class PendingReads:
    """A batch of reads still on the device: :attr:`tensor` is every read
    half's ``[values, ids]`` flattened into one fp32 vector; :meth:`finish`
    unpacks the host copy into one ``list[LensRead]`` per request."""

    def __init__(
        self, tensor: torch.Tensor, planned: list[_Planned],
        decode: Callable[[int], str],
    ) -> None:
        self.tensor = tensor
        self._planned = planned
        self._decode = decode

    def finish(self, host: torch.Tensor) -> list[list[LensRead]]:
        flat = host.reshape(-1).tolist()
        out: list[list[LensRead]] = []
        for plan in self._planned:
            out.append([
                LensRead(
                    int(plan.request.layer), plan.request.site, plan.depth,
                    r=self._half(plan.r, flat, plan.n, row),
                    s=self._half(plan.s, flat, plan.n, row),
                    of=plan.request.of,
                )
                for row in range(plan.n)
            ])
        return out

    def _half(self, half: _Half, flat: list[float], n: int, row: int) -> FamilyRead:
        if half.absent is not None:
            return FamilyRead(
                half.family, half.weight, half.unit, lens=half.lens, absent=half.absent,
            )
        k = half.width
        values = half.offset + row * k
        ids = half.offset + n * k + row * k
        tokens = tuple(
            TokenScore(int(flat[ids + j]), self._decode(int(flat[ids + j])), float(flat[values + j]))
            for j in range(k)
        )
        return FamilyRead(
            half.family, half.weight, half.unit, lens=half.lens, tokens=tokens,
            centred=half.centred,
        )


def plan_reads(
    requests: Sequence[VectorRead],
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
    centre: Centre | None = None,
    no_r_lens: str = "no R-family lens is selected for this model",
    no_s_lens: str = "no S-family lens is selected for this model",
) -> PendingReads:
    """Compute every request's halves on the device, packed for one transfer.

    ``readout``, ``embeddings`` are thunks, so a half that is absent never
    pays for the vocabulary head or the embedding statistics; ``centre`` is
    called only for the S half of a residual read, and a residual read with
    an S half and no ``centre`` is a broken invariant (``ValueError``).
    """
    for lens, family in ((r_lens, "R"), (s_lens, "S")):
        if lens is not None and lens.key.family != family:
            raise ValueError(f"a {lens.kind} lens cannot read the {family}-family half")
    weights = kind_weights(n_layers, band)
    parts: list[torch.Tensor] = []
    planned: list[_Planned] = []
    offset = 0

    for request in requests:
        layer = int(request.layer)
        if not 0 <= layer < int(n_layers):
            raise ValueError(f"layer {layer} is outside the model's {n_layers} layers")
        rows = request.matrix_rows
        n = int(rows.shape[0])
        w_s, w_r = weights[layer]

        def half(
            family: LensFamily, lens: JacobianLens | None, weight: float, unit: ReadUnit,
            missing: str, run: Callable[[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]],
            *, centred: bool = False,
        ) -> _Half:
            nonlocal offset
            if lens is None:
                return _Half(family, weight, unit, None, False, absent=missing)
            label = lens.site_key(request.site).label
            matrix: torch.Tensor | None
            try:
                if family == "R":
                    matrix, label = output_transport(lens, layer, request.site, n_layers, device)
                else:
                    matrix = _matrix(lens, layer, request.site, device)
            except LensNotFittedError as exc:
                return _Half(family, weight, unit, label, False, absent=str(exc))
            values, ids = run(matrix)
            width = int(values.shape[1])
            parts.append(values.float().reshape(-1))
            parts.append(ids.float().reshape(-1))
            planned_half = _Half(family, weight, unit, label, centred, offset=offset, width=width)
            offset += 2 * n * width
            return planned_half

        def s_run(matrix: torch.Tensor | None, *, rows: torch.Tensor = rows) -> tuple[torch.Tensor, torch.Tensor]:
            assert matrix is not None, "an S read always has its K_L"
            x = rows.to(device=matrix.device, dtype=torch.float32)
            if request.of == "residual":
                if centre is None:
                    raise ValueError(
                        "a residual read needs the neutral means its S half centres on"
                    )
                mean = centre(request.site, layer).to(device=matrix.device, dtype=torch.float32)
                x = x - mean.reshape(1, -1)
            return read_s(x, matrix, embeddings(), k=k)

        r_half = half(
            "R", r_lens, w_r, "probability", no_r_lens,
            lambda matrix, rows=rows: read_r(rows.to(device), matrix, readout(), k=k),
        )
        s_half = half(
            "S", s_lens, w_s, "footprint_z", no_s_lens, s_run,
            centred=request.of == "residual",
        )
        planned.append(_Planned(request, n, layer_depth(layer, int(n_layers)), r_half, s_half))

    tensor = (
        torch.cat(parts) if parts
        else torch.zeros(0, dtype=torch.float32, device=device)
    )
    return PendingReads(tensor, planned, decode)


def read_vectors(
    vectors: torch.Tensor,
    layer: int,
    site: str,
    *,
    of: ReadOf,
    r_lens: JacobianLens | None,
    s_lens: JacobianLens | None,
    readout: Callable[[], Readout],
    embeddings: Callable[[], TokenEmbeddings],
    decode: Callable[[int], str],
    n_layers: int,
    device: torch.device,
    k: int = 8,
    band: LensBand | None = None,
    centre: Centre | None = None,
    no_r_lens: str = "no R-family lens is selected for this model",
    no_s_lens: str = "no S-family lens is selected for this model",
) -> list[LensRead]:
    """Read every row of ``vectors`` (``[d]`` or ``[n, d]``) at ``(layer, site)``
    as a residual or a direction, in one transfer (:func:`plan_reads`)."""
    pending = plan_reads(
        [VectorRead(vectors, int(layer), site, of)],
        r_lens=r_lens, s_lens=s_lens, readout=readout, embeddings=embeddings,
        decode=decode, n_layers=n_layers, device=device, k=k, band=band,
        centre=centre, no_r_lens=no_r_lens, no_s_lens=no_s_lens,
    )
    return pending.finish(pending.tensor.detach().cpu())[0]


__all__ = [
    "Centre",
    "FamilyRead",
    "LensFamily",
    "LOGIT_LENS",
    "LensRead",
    "PendingReads",
    "READ_OF",
    "ReadOf",
    "ReadUnit",
    "TokenScore",
    "VectorRead",
    "output_transport",
    "plan_reads",
    "read_r",
    "read_s",
    "read_vectors",
]
