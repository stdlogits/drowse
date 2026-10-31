"""Static unit reads: what a neuron or a head is about, from its weights.

Every read follows the four-lens rule (:mod:`drowse.lens.read`): the **site**
comes from where the vector sits and the **kind** from the depth, so each
place returns an S half and an R half with the depth rule's weights (the
light half is still returned, for the UI to fade).

**Neurons** (``neuron/L8n344``), per pole (``+``: ``act > 0``, ``−``: ``act <
0``):

- **writes** — the pole's write ``±w`` (``s · γ_out ⊙ W_down[:, n]``,
  :func:`~drowse.model.units.neuron_write`) read at ``out`` of its layer
  through :meth:`~drowse.lens.instrument.LensInstrument.read` as a direction
  (uncentred).  The last block's ``out`` is the final residual, where the R
  read is the logit lens exactly (``J = I``), labelled ``logit``: the lens
  read's own rule (:func:`~drowse.lens.read.output_transport`), which the
  population statistics here share.
- **fires on** — the pole's read direction ``â ± b̂`` (unit gate and up rows
  times the MLP input norm's gain) read at ``mid``.

**Hub tokens** top many raw reads at once from the lens's geometry.  With
``hide_hubs`` each half ranks by neuron-lens's layer-contrastive ``_z``
instead: each token's score z-scored across the layer's whole population
(:class:`ContrastCache`, a derived per-layer cache under
``models/<id>/units/contrast/``).  As neuron-lens computes them:

- R: the lens logit (after the model's final norm and softcap) over the
  layer's ``+`` writes (the ``−`` pole's z is minus it), and over each pole's
  read directions separately;
- S: the raw score ``û · ẽ_v`` of unit covectors ``û = u/‖u‖``, ``u = x K_L``,
  over the ``+`` writes (``−`` is minus it) and over both poles' read
  directions together.

Scalars, as neuron-lens: ``w_norm`` ``‖w‖``, ``cos_ab`` ``â · b̂``,
``null_share`` (the write's squared-norm share in the bottom 1% of
right-singular directions of ``W_U diag(γ_final)``), and per pole the skew
and kurtosis of the write's logit-lens logits over the token mask.

**Heads** (``head/L20h17``), as neuron-lens's head reads (its findings §11):
a head reads token ``v`` as its S footprint at ``L − 1``'s output ``X_v =
K^{out}_{L−1} ẽ_v`` (``ẽ_v`` itself at L0) and writes at ``mid`` of ``L``.

- **copies / suppresses** — the OV map ``M[u, v] = A_u · B_v`` with ``B_v =
  W_V^h X_v`` and ``A_u`` the output token ``u``'s read of ``W_O^h`` (R:
  ``(γ_f ⊙ W_U[u]) J^{mid}_L W_O^h``; S: ``ẽ_u K^{mid⊤}_L W_O^h``): each
  token's own-column z ``(M[v,v] − mean_u M[u,v]) / std_u M[u,v]`` from the
  mean and covariance of ``A`` (the ``V × V`` map is never built).  The top
  ``k`` copy, the bottom ``k`` suppress.  The copy score is Elhage et al.'s
  ``Σλ/Σ|λ|`` of ``Bᵀ A``.  Hidden hubs z-score each token across the
  layer's heads.
- **inputs** — what each of the head's query, key and value projections
  picks up at ``L − 1``'s output: token ``v`` by the share of its unit
  direction the projection keeps, ``‖W^h x̂_v‖`` (S: ``x_v`` the footprint;
  R: ``x_v = J^{out⊤}_{L−1}(γ_f ⊙ W_U[v])``, what a residual carries when it
  is about to say ``v``), as a z over the token mask.  neuron-lens reads the
  QK circuit from attention statistics instead (``qk_S``); this read needs
  none.
- **head norms** — a block with per-head norms (Qwen3's and Gemma's q/k
  norms, Gemma 4's weightless value norm) reads each projection through its
  norm's true derivative averaged over the residuals entering the block, as
  neuron-lens applies them (``norm_derivative``: ``diag(g)·E[w(r I − r³ x
  xᵀ/hd)]``, each position weighted by the input norm's scale ``1/ρ``): ``B``
  through the value norm's, each input through its own.  The residuals are
  the neutral ones at ``L − 1``'s output (the whitener's rows; at L0 the
  kept tokens' embeddings).

Every product takes the small factors first (``J^{mid}_L W_O^h``,
``K^{out⊤}_{L−1} W_V^{h⊤}``: ``[d, hd]``) and streams the vocabulary rows
through them in chunks, so a head read holds ``[V, hd]`` at most and never a
``[V, d]`` tensor (one is 2.5 GB on Qwen3-8B); the OV z takes two streamed
passes (``A``'s mean and scatter merged chunk by chunk, then each token's z).

Where drowse differs from neuron-lens: the token mask is the tokenizer's
(:func:`~drowse.lens.footprint.token_mask`; neuron-lens keeps tokens seen five
times in its corpora); S raw reads report footprint z (same order as its raw
score); the R head reads hold the final norm's scale constant (as it does);
the head norms' derivative averages over drowse's neutral residuals (one
pooled position per neutral statement) where neuron-lens averages over every
position of its activation corpora; fp32 throughout where it used fp16.

Every read runs inside the model-exclusive section under ``inference_mode``,
moves each result off the device once, and is fp32.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from drowse.base.atomic import artifact_lock, write_bytes_atomic
from drowse.base.paths import ensure_within, model_dir
from drowse.errors import RequestError, WhitenerError
from drowse.lens.depth import kind_weights, layer_depth
from drowse.lens.jlens import LensNotFittedError
from drowse.lens.read import LOGIT_LENS, FamilyRead, TokenScore, output_transport
from drowse.model.blocks import BlockSpec, NormSpec
from drowse.model.units import head_read_rows, head_write_matrix, neuron_read_rows, neuron_writes
from drowse.naming import HeadRef, NeuronRef, UnitRef, parse_unit_ref
from drowse.units.catalog import LABEL_CREDIT
from drowse.units.runtime import UnitsRuntime, UnitsUnavailableError

if TYPE_CHECKING:
    from drowse.geometry.whitener import LayerWhitener
    from drowse.lens.footprint import TokenEmbeddings
    from drowse.lens.instrument import LensInstrument
    from drowse.units.catalog import CatalogBinding

Pole = Literal["+", "-"]
Family = Literal["R", "S"]
#: The unit a half's token scores are in.
HalfUnit = Literal["probability", "footprint_z", "layer_z", "own_token_z", "energy_z"]
#: What a top query ranks by: a pole's write or its read direction.
TopBy = Literal["writes", "fires"]
#: The vocabulary rows a head read streams: centred embeddings or gained
#: unembeddings.
RowKind = Literal["embed", "unembed"]

#: The format of a contrast cache shard; a mismatch recomputes (derived cache).
CONTRAST_FORMAT = 1
#: Rows per chunk in a population pass (an fp32 ``[chunk, V]`` logit block).
ROW_CHUNK = 512
#: Vocabulary rows per chunk when a pass runs over the embedding or unembedding.
VOCAB_CHUNK = 8192
#: The most elements one chunk's product holds in a head read's streamed
#: passes (fewer rows per chunk for a wide factor: a whole layer's heads).
_CHUNK_ELEMENTS = 1 << 24
#: neuron-lens's floors: the R logit std (``clamp_min(1e-8).sqrt()``), the S
#: score std (``clamp_min(1e-12).sqrt()``) and the across-heads std.
_R_STD_FLOOR = 1e-4
_S_STD_FLOOR = 1e-6
_HEAD_STD_FLOOR = 1e-6
#: The bottom share of ``W_U diag(γ)``'s right-singular directions the null
#: share measures (Stolfo et al. 2024's entropy neurons; neuron-lens 1%).
NULL_SHARE = 0.01
#: The largest ``k`` a read lists.
MAX_K = 256
_HEAD_ROLES = ("query", "key", "value")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitHalf:
    """One family's read of a unit vector at its place (the shape of
    :class:`~drowse.lens.read.FamilyRead`, with the unit widened to the unit
    reads' scores)."""

    family: Family
    weight: float
    unit: HalfUnit
    lens: str | None = None
    tokens: tuple[TokenScore, ...] = ()
    absent: str | None = None

    @property
    def present(self) -> bool:
        return self.absent is None

    @classmethod
    def of(cls, half: FamilyRead) -> "UnitHalf":
        return cls(half.family, half.weight, half.unit, half.lens, half.tokens, half.absent)

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
class PlaceRead:
    """One unit vector read at its place: ``(layer, site)`` with its S and R
    halves.  ``hubs`` says whether hub tokens are ``shown`` (raw reads) or
    ``hidden`` (layer-contrastive ``_z`` reads)."""

    layer: int
    site: str
    depth: float
    s: UnitHalf
    r: UnitHalf
    hubs: Literal["shown", "hidden"] = "shown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "site": self.site,
            "depth": round(self.depth, 6),
            "hubs": self.hubs,
            "s": self.s.to_dict(),
            "r": self.r.to_dict(),
        }


@dataclass(frozen=True)
class Firing:
    """How a unit (or a pole) fires: the neutral statistics drowse measured
    and the catalog's activation table, each present or absent with why.

    ``rate`` is the share of tokens the unit fires on (the catalog's
    ``|act| > act_top / 4`` over its corpora); drowse's neutral statistics
    carry no rate, so without a bound catalog it is absent.
    """

    neutral: dict[str, float] | None = None
    neutral_absent: str | None = None
    catalog: dict[str, Any] | None = None
    catalog_absent: str | None = None

    @property
    def rate(self) -> float | None:
        if self.catalog is None:
            return None
        value = self.catalog.get("rate")
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        rate = self.rate
        return {
            "rate": None if rate is None else round(rate, 6),
            "rate_absent": None if rate is not None else (
                self.catalog_absent or "the catalog row has no firing rate"
            ),
            "neutral": self.neutral,
            "neutral_absent": self.neutral_absent,
            "catalog": self.catalog,
            "catalog_absent": self.catalog_absent,
        }


@dataclass(frozen=True)
class NeuronPoleRead:
    """One pole of a neuron: its writes, what it fires on, its logit-lens
    shape, how it fires and its catalog label."""

    pole: Pole
    writes: PlaceRead
    fires_on: PlaceRead
    vocab_skew: float
    vocab_kurt: float
    firing: Firing
    label: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pole": self.pole,
            "writes": self.writes.to_dict(),
            "fires_on": self.fires_on.to_dict(),
            "vocab_skew": round(self.vocab_skew, 6),
            "vocab_kurt": round(self.vocab_kurt, 6),
            "firing": self.firing.to_dict(),
            "label": self.label,
        }


@dataclass(frozen=True)
class NeuronRead:
    """A neuron from its weights: scalars and both poles."""

    ref: NeuronRef
    depth: float
    w_norm: float
    cos_ab: float
    null_share: float
    mean_act: float | None
    poles: tuple[NeuronPoleRead, NeuronPoleRead]
    catalog: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.ref.qualified,
            "kind": "neuron",
            "layer": self.ref.layer,
            "index": self.ref.index,
            "depth": round(self.depth, 6),
            "scalars": {
                "w_norm": round(self.w_norm, 6),
                "cos_ab": round(self.cos_ab, 6),
                "null_share": round(self.null_share, 6),
                "mean_act": None if self.mean_act is None else round(self.mean_act, 6),
            },
            "poles": [pole.to_dict() for pole in self.poles],
            "catalog": self.catalog,
        }


@dataclass(frozen=True)
class HeadRead:
    """A head from its weights: its OV write (copies and suppresses), its
    inputs per projection, the copy scores, how it fires and its label."""

    ref: HeadRef
    depth: float
    kv_head: int
    copies: PlaceRead
    suppresses: PlaceRead
    inputs: Mapping[str, PlaceRead]
    copy_score: Mapping[str, float | None]
    firing: Firing
    label: dict[str, Any] | None
    catalog: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.ref.qualified,
            "kind": "head",
            "layer": self.ref.layer,
            "index": self.ref.index,
            "kv_head": self.kv_head,
            "depth": round(self.depth, 6),
            "copies": self.copies.to_dict(),
            "suppresses": self.suppresses.to_dict(),
            "inputs": {role: place.to_dict() for role, place in self.inputs.items()},
            "copy_score": {
                family: None if value is None else round(float(value), 6)
                for family, value in self.copy_score.items()
            },
            "firing": self.firing.to_dict(),
            "label": self.label,
            "catalog": self.catalog,
        }


UnitDetail = NeuronRead | HeadRead


@dataclass(frozen=True)
class TopHit:
    """One neuron pole ranked by how specifically it writes (or fires on) a
    token: ``score`` is the depth-weighted layer-contrastive z over the
    halves present; each half reports its own z, its raw read of the token
    (R probability, S footprint z) and its weight."""

    ref: NeuronRef
    pole: Pole
    score: float
    r: dict[str, Any]
    s: dict[str, Any]
    label: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.ref.qualified,
            "layer": self.ref.layer,
            "index": self.ref.index,
            "pole": self.pole,
            "score": round(self.score, 6),
            "r": self.r,
            "s": self.s,
            "label": self.label,
        }


@dataclass(frozen=True)
class TopResult:
    """``top``'s ranked poles with what each layer read through."""

    token_id: int
    token: str
    by: TopBy
    hits: tuple[TopHit, ...]
    layers: Mapping[int, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "token": self.token,
            "by": self.by,
            "hits": [hit.to_dict() for hit in self.hits],
            "layers": {str(layer): info for layer, info in sorted(self.layers.items())},
        }


@dataclass(frozen=True)
class FindResult:
    """A direction split over each layer's neuron writes: per layer the share
    of its Mahalanobis variance the ``k`` neurons carry and each neuron with
    its signed coefficient (the sign is the pole) and label."""

    target: str
    k: int
    layers: Mapping[int, dict[str, Any]]
    catalog: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        shares = [row["share"] for row in self.layers.values()]
        return {
            "target": self.target,
            "k": self.k,
            "mean_share": round(sum(shares) / len(shares), 6) if shares else 0.0,
            "layers": {str(layer): row for layer, row in sorted(self.layers.items())},
            "catalog": dict(self.catalog),
        }


# ---------------------------------------------------------------------------
# Small math
# ---------------------------------------------------------------------------

def own_token_z(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``[V]``: each token's z in its own column of ``M[u, v] = A_u · B_v``
    (``A``, ``B``: ``[V, k]``), from the mean and covariance of ``A``'s rows
    (neuron-lens ``heads.own_token_z`` for one head)."""
    abar = A.mean(0)
    centred = A - abar
    cov = centred.T @ centred / A.shape[0]
    var = ((B @ cov) * B).sum(-1)
    return ((A * B).sum(-1) - B @ abar) / var.clamp_min(1e-30).sqrt()


def copy_score(A: torch.Tensor, B: torch.Tensor) -> float:
    """Elhage et al.'s copying score ``Σλ / Σ|λ|`` of ``A Bᵀ`` from the
    eigenvalues of ``Bᵀ A`` (its nonzero ones): +1 copies, −1 suppresses."""
    return _copy_score_of(B.T @ A)


def _copy_score_of(bta: torch.Tensor) -> float:
    """:func:`copy_score` from ``Bᵀ A`` ``[k, k]`` itself."""
    # To the host first: MPS has no float64.
    lam = torch.linalg.eigvals(bta.detach().cpu().double())
    denom = float(lam.abs().sum())
    return float(lam.real.sum()) / denom if denom > 0 else 0.0


def logit_moments(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Skew and kurtosis per row of ``[n, V]`` (neuron-lens ``layer_reads``)."""
    centred = logits - logits.mean(-1, keepdim=True)
    var = centred.pow(2).mean(-1).clamp_min(1e-30)
    return centred.pow(3).mean(-1) / var.pow(1.5), centred.pow(4).mean(-1) / var.pow(2)


def _final_hidden(readout: Any, rows: torch.Tensor) -> torch.Tensor:
    """The final norm (and projection) of :class:`~drowse.model.readout.LogitReadout`,
    in the model's dtype as the model runs it, then fp32."""
    weight = readout.head.weight
    hidden = readout.norm(rows.to(weight.dtype))
    if readout.projection is not None:
        hidden = readout.projection(hidden)
    return hidden.float()


def _logit_tail(readout: Any, logits: torch.Tensor) -> torch.Tensor:
    """LogitReadout's elementwise tail (scale, divisor, softcap) in fp32."""
    if readout.scale != 1.0:
        logits = logits * readout.scale
    if readout.divisor != 1.0:
        logits = logits / readout.divisor
    if readout.softcap is not None:
        logits = (logits / readout.softcap).tanh() * readout.softcap
    return logits


def full_logits(readout: Any, rows: torch.Tensor, *, chunk: int = VOCAB_CHUNK) -> torch.Tensor:
    """The model's logits ``[n, V]`` for final residuals ``rows``, the
    vocabulary product in fp32 (chunked over the head's rows), so a
    population's ``logsumexp`` and a single token's logit agree exactly."""
    weight = readout.head.weight
    hidden = _final_hidden(readout, rows)
    vocab = int(weight.shape[0])
    out = torch.empty(hidden.shape[0], vocab, dtype=torch.float32, device=hidden.device)
    bias = getattr(readout.head, "bias", None)
    for start in range(0, vocab, chunk):
        block = hidden @ weight[start:start + chunk].detach().float().T
        if bias is not None:
            block = block + bias[start:start + chunk].float()
        out[:, start:start + chunk] = block
    return _logit_tail(readout, out)


def token_logits(readout: Any, rows: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """The model's logits of ``token_ids`` only, ``[n, m]`` fp32: the
    :func:`full_logits` path without the full vocabulary product."""
    weight = readout.head.weight
    hidden = _final_hidden(readout, rows)
    ids = token_ids.to(device=weight.device, dtype=torch.long)
    logits = hidden @ weight.index_select(0, ids).detach().float().T
    bias = getattr(readout.head, "bias", None)
    if bias is not None:
        logits = logits + bias.index_select(0, ids).float()
    return _logit_tail(readout, logits)


def _drain(device: torch.device) -> None:
    """Commit the MPS queue (a long pass must not exhaust it) and free the
    allocator's cache; nothing elsewhere."""
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# The contrast cache
# ---------------------------------------------------------------------------

def contrast_dir(model_id: str) -> Path:
    """``$DROWSE_HOME/models/<safe id>/units/contrast``."""
    return model_dir(model_id) / "units" / "contrast"


@dataclass(frozen=True, eq=False)
class ContrastStats:
    """One layer's population statistics for one family: ``{name: tensor}``
    fp32 on the device (``<population>.mean`` / ``.std`` ``[V]`` and the R
    family's per-row ``<population>.lse`` ``[n]``)."""

    tensors: Mapping[str, torch.Tensor]

    def mean_std(self, population: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tensors[f"{population}.mean"], self.tensors[f"{population}.std"]

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class ContrastCache:
    """Per-layer population statistics on disk, keyed by what they depend on.

    A shard is ``<key>/L<layer>.safetensors`` with a metadata record (format,
    the key's parts, the mask digest); a shard that is missing, unreadable or
    names anything else is recomputed (a derived cache, never migrated).  In
    memory the latest few shards stay resident.
    """

    def __init__(self, model_id: str, *, resident: int = 8) -> None:
        self._model_id = model_id
        self._resident = int(resident)
        self._memory: dict[tuple[str, int], ContrastStats] = {}
        self._lock = threading.Lock()

    def _path(self, key: str, layer: int) -> Path:
        return ensure_within(contrast_dir(self._model_id), key, f"L{int(layer):03d}.safetensors")

    def get(
        self, key: str, layer: int, meta: Mapping[str, Any], device: torch.device,
        build: Callable[[], Mapping[str, torch.Tensor]],
    ) -> ContrastStats:
        """The stats for ``(key, layer)``: memory, then disk when ``meta``
        matches, else ``build()`` (then saved)."""
        from safetensors.torch import load_file, save

        slot = (key, int(layer))
        with self._lock:
            hit = self._memory.get(slot)
        if hit is not None:
            return hit
        record = {"format": CONTRAST_FORMAT, "layer": int(layer), **dict(meta)}  # meta carries the weights
        path = self._path(key, layer)
        tensors: Mapping[str, torch.Tensor] | None = None
        if path.is_file():
            try:
                from safetensors import safe_open

                with safe_open(str(path), framework="pt") as handle:
                    stored = json.loads((handle.metadata() or {}).get("drowse_contrast", "null"))
                if stored == record:
                    tensors = load_file(str(path))
            except Exception:  # noqa: BLE001 - a derived cache recomputes
                tensors = None
        if tensors is None:
            built = {name: value.detach().float().cpu().contiguous() for name, value in build().items()}
            with artifact_lock(path):
                write_bytes_atomic(path, save(built, metadata={"drowse_contrast": json.dumps(record)}))
            tensors = built
        stats = ContrastStats({name: value.to(device) for name, value in tensors.items()})
        with self._lock:
            if len(self._memory) >= self._resident:
                self._memory.pop(next(iter(self._memory)))
            self._memory[slot] = stats
        return stats

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _Lenses:
    """The two selected lenses at read time and their cache identities."""

    r: Any
    s: Any
    r_identity: str | None
    s_identity: str | None


class UnitReader:
    """Static reads of neurons and heads through the session's lenses.

    ``lens`` is the session's lens instrument (the selected R and S sources,
    their read primitive and the token embeddings); ``runtime`` the units
    runtime (block descriptions, neutral unit statistics); ``catalog`` the
    model's neuron-lens catalog binding; ``weights`` the loaded weights'
    fingerprint (it keys the contrast cache); ``whitener`` the neutral
    baseline's whitener getter, whose neutral residuals a head's norms are
    averaged over (``None``: a block with head norms refuses head reads).
    """

    def __init__(
        self,
        runtime: UnitsRuntime,
        lens: "LensInstrument",
        *,
        model_id: str,
        n_layers: int,
        exclusive: Callable[..., Any],
        catalog: "CatalogBinding",
        weights: Callable[[], str],
        whitener: "Callable[[], LayerWhitener | None] | None" = None,
    ) -> None:
        self.runtime = runtime
        self.lens = lens
        self.model_id = model_id
        self.n_layers = int(n_layers)
        self._exclusive = exclusive
        self.catalog = catalog
        self._weights_id = weights
        self._whitener = whitener
        self.contrast = ContrastCache(model_id)
        self._null: torch.Tensor | None = None
        self._kept_ids: torch.Tensor | None = None
        #: Per block, the head norms' mean derivatives and what they were
        #: averaged over (the whitener or the token embeddings).
        self._norm_cache: dict[int, tuple[object, dict[str, torch.Tensor | None]]] = {}

    # ------------------------------------------------------------ plumbing

    @property
    def device(self) -> torch.device:
        return self.runtime.device

    def _section(self, what: str) -> Any:
        return self._exclusive(
            f"{what} called while another model use is in flight; retry shortly",
            phase_msg=f"{what} called while a generation is in flight",
        )

    def _lenses(self) -> _Lenses:
        r = self.lens.resident()
        s = self.lens.resident_input()
        r_id = self.lens.runtime.identity if r is not None else None
        s_id = self.lens.input_runtime.identity if s is not None else None
        return _Lenses(r, s, None if r_id is None else r_id[:16], None if s_id is None else s_id[:16])

    def _readout(self) -> Any:
        return self.lens.runtime.readout_modules()[1]

    def _unembed(self) -> torch.Tensor:
        return self.lens.runtime.readout_modules()[0]

    def _embeddings(self) -> "TokenEmbeddings":
        return self.lens.token_embeddings()

    def _final_gain(self) -> torch.Tensor:
        return self.runtime.spec().final_norm.gain(device=self.device)

    def _decode(self, token_id: int) -> str:
        return self.lens.runtime.decode_id(int(token_id))

    def _matrix(self, lens: Any, layer: int, site: str) -> torch.Tensor | None:
        """``lens``'s matrix at ``(layer, site)`` on the device, or ``None``."""
        if lens is None or site not in lens.sites:
            return None
        try:
            return lens.matrix(int(layer), site).to(device=self.device, dtype=torch.float32)
        except LensNotFittedError:
            return None

    def _r_matrix(self, lenses: _Lenses, layer: int, site: str) -> tuple[torch.Tensor | None, str | None]:
        """The R transport at ``(layer, site)`` and its label, by the lens
        read's rule (:func:`~drowse.lens.read.output_transport`): the lens's
        matrix, at the last block's ``out`` ``None`` (the identity, the logit
        lens), and ``(None, None)`` where the lens has none."""
        if lenses.r is None or site not in lenses.r.sites:
            return None, None
        try:
            return output_transport(lenses.r, layer, site, self.n_layers, self.device)
        except LensNotFittedError:
            return None, None

    def _r_key(self, lenses: _Lenses, layer: int, site: str) -> str:
        matrix_label = self._r_matrix(lenses, layer, site)[1]
        if matrix_label == LOGIT_LENS:
            return "R-logit"
        return f"R-{lenses.r_identity}"

    def _mask(self, rows: int) -> torch.Tensor:
        """The S read's token mask over ``rows`` vocabulary ids (padded with
        ``False`` past the embedding)."""
        keep = self._embeddings().keep
        if int(keep.shape[0]) >= rows:
            return keep[:rows]
        pad = torch.zeros(rows - int(keep.shape[0]), dtype=torch.bool, device=keep.device)
        return torch.cat([keep, pad])

    def _meta(self, **parts: Any) -> dict[str, Any]:
        """A contrast shard's record: ``parts`` plus the token mask digest and
        the loaded weights' fingerprint (both change what a population reads)."""
        import hashlib

        keep = self._embeddings().keep.to("cpu", torch.uint8).numpy().tobytes()
        return {
            **parts,
            "mask": hashlib.sha256(keep).hexdigest()[:16],
            "weights": self._weights_id()[:16],
        }

    def _kept(self) -> torch.Tensor:
        """The kept token ids (the mask's), on the device."""
        if self._kept_ids is None:
            self._kept_ids = self._embeddings().keep.nonzero().squeeze(1).to(self.device)
        return self._kept_ids

    def _check_k(self, k: int) -> int:
        if not 1 <= int(k) <= MAX_K:
            raise RequestError(f"k must be in [1, {MAX_K}]")
        return int(k)

    def _weights(self, layer: int) -> tuple[float, float]:
        """``(w_S, w_R)`` at ``layer`` by the lens instrument's band."""
        return kind_weights(self.n_layers, self.lens.band)[int(layer)]

    # ----------------------------------------------------------- R and S rows

    def _r_logits(self, rows: torch.Tensor, matrix: torch.Tensor | None) -> torch.Tensor:
        """Full-vocabulary R logits ``[n, V]`` fp32 for residual rows."""
        x = rows.to(device=self.device, dtype=torch.float32)
        if matrix is not None:
            x = x @ matrix.T
        return full_logits(self._readout(), x)

    def _s_covectors(self, rows: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """``u = x K_L`` ``[n, d_emb]`` fp32."""
        return rows.to(device=self.device, dtype=torch.float32) @ matrix

    def _raw_scores(self, unit_u: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """``û · ẽ_v`` for vocabulary ids ``[start, stop)``, unmasked ``[n, stop−start]``."""
        emb = self._embeddings()
        rows = emb.weight[start:stop].detach().to(torch.float32)
        return (unit_u @ rows.T) * emb.scale - (unit_u @ emb.mean)[:, None]

    # --------------------------------------------------------- contrast stats

    def _r_population_stats(
        self, populations: Mapping[str, torch.Tensor], matrix: torch.Tensor | None,
        *, lse: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Per population ``[n, d]``: the R logits' per-token mean and std
        across rows, and with ``lse`` each row's ``logsumexp`` over the
        vocabulary (and of the negated logits, as ``.lse_neg``)."""
        out: dict[str, torch.Tensor] = {}
        for name, rows in populations.items():
            n = int(rows.shape[0])
            total = sq = None
            lses: list[torch.Tensor] = []
            negs: list[torch.Tensor] = []
            for start in range(0, n, ROW_CHUNK):
                logits = self._r_logits(rows[start:start + ROW_CHUNK], matrix)
                total = logits.sum(0) if total is None else total + logits.sum(0)
                sq = logits.pow(2).sum(0) if sq is None else sq + logits.pow(2).sum(0)
                if lse:
                    lses.append(logits.logsumexp(-1))
                    negs.append((-logits).logsumexp(-1))
                del logits
            assert total is not None and sq is not None
            mean = total / n
            out[f"{name}.mean"] = mean
            out[f"{name}.std"] = (sq / n - mean.pow(2)).clamp_min(_R_STD_FLOOR ** 2).sqrt()
            if lse:
                out[f"{name}.lse"] = torch.cat(lses)
                out[f"{name}.lse_neg"] = torch.cat(negs)
            _drain(self.device)
        return out

    def _s_population_stats(
        self, populations: Mapping[str, torch.Tensor], matrix: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Per population ``[n, d]``: the raw scores of the rows' unit
        covectors, per-token mean and std across rows (masked tokens included;
        they never rank)."""
        emb = self._embeddings()
        vocab = emb.vocab
        out: dict[str, torch.Tensor] = {}
        for name, rows in populations.items():
            u = self._s_covectors(rows, matrix)
            unit_u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            n = int(unit_u.shape[0])
            total = torch.zeros(vocab, dtype=torch.float32, device=self.device)
            sq = torch.zeros_like(total)
            for vstart in range(0, vocab, VOCAB_CHUNK):
                vstop = min(vstart + VOCAB_CHUNK, vocab)
                for start in range(0, n, ROW_CHUNK * 8):
                    scores = self._raw_scores(unit_u[start:start + ROW_CHUNK * 8], vstart, vstop)
                    total[vstart:vstop] += scores.sum(0)
                    sq[vstart:vstop] += scores.pow(2).sum(0)
            mean = total / n
            out[f"{name}.mean"] = mean
            out[f"{name}.std"] = (sq / n - mean.pow(2)).clamp_min(_S_STD_FLOOR ** 2).sqrt()
            _drain(self.device)
        return out

    def _neuron_populations(self, block: BlockSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(writes [f, d], reads+ [f, d], reads− [f, d])`` on the device."""
        writes = neuron_writes(block).T.to(self.device)
        a, b = neuron_read_rows(block)
        a, b = a.to(self.device), b.to(self.device)
        return writes, a + b, a - b

    def _neuron_contrast(self, lenses: _Lenses, layer: int, family: Family, place: str) -> ContrastStats | None:
        """The layer's neuron statistics for ``family`` at ``place``
        (``writes`` at out, ``fires`` at mid), or ``None`` without the lens."""
        block = self.runtime.block(layer)
        if family == "R":
            site = "out" if place == "writes" else "mid"
            matrix, label = self._r_matrix(lenses, layer, site)
            if label is None:
                return None
            key = self._r_key(lenses, layer, site)

            def build_r() -> dict[str, torch.Tensor]:
                writes, plus, minus = self._neuron_populations(block)
                if place == "writes":
                    return self._r_population_stats({"writes": writes}, matrix)
                return self._r_population_stats({"fires+": plus, "fires-": minus}, matrix)

            return self.contrast.get(
                f"{key}-neurons-{place}", layer,
                self._meta(family="R", lens=label, place=place),
                self.device, build_r,
            )
        site = "out" if place == "writes" else "mid"
        matrix = self._matrix(lenses.s, layer, site)
        if matrix is None:
            return None

        def build_s() -> dict[str, torch.Tensor]:
            writes, plus, minus = self._neuron_populations(block)
            if place == "writes":
                return self._s_population_stats({"writes": writes}, matrix)
            return self._s_population_stats({"fires": torch.cat([plus, minus])}, matrix)

        return self.contrast.get(
            f"S-{lenses.s_identity}-neurons-{place}", layer,
            self._meta(family="S", lens=lenses.s.site_key(site).label, place=place),
            self.device, build_s,
        )

    # ----------------------------------------------------------- top-k lists

    def _top_tokens(self, values: torch.Tensor, k: int) -> list[tuple[TokenScore, ...]]:
        """Top-``k`` per row of ``[n, V]`` (masked entries ``-inf``), one
        device-to-host transfer."""
        top, ids = values.topk(min(int(k), int(values.shape[-1])), dim=-1)
        host = torch.cat([top.float(), ids.float()], dim=0).cpu()
        n = int(values.shape[0])
        out: list[tuple[TokenScore, ...]] = []
        for row in range(n):
            out.append(tuple(
                TokenScore(int(host[n + row, j]), self._decode(int(host[n + row, j])), float(host[row, j]))
                for j in range(int(top.shape[1]))
                if math.isfinite(float(host[row, j]))
            ))
        return out

    def _masked(self, values: torch.Tensor) -> torch.Tensor:
        keep = self._mask(int(values.shape[-1]))
        return values.masked_fill(~keep, -torch.inf)

    # ---------------------------------------------------------- neuron place

    def _place(
        self, rows: torch.Tensor, layer: int, site: str, *, k: int, lenses: _Lenses,
    ) -> list[PlaceRead]:
        """Raw reads of ``rows`` at ``(layer, site)`` through the lens
        instrument's read (at the last block's ``out`` the R half is the logit
        lens, by the read's own rule)."""
        reads = self.lens.read(rows, int(layer), site, of="direction", k=k)
        assert isinstance(reads, list)
        return [
            PlaceRead(read.layer, read.site, read.depth, UnitHalf.of(read.s), UnitHalf.of(read.r))
            for read in reads
        ]

    def _hubless(
        self,
        places: Sequence[PlaceRead],
        rows: torch.Tensor,
        signs: Sequence[float],
        *,
        k: int,
        lenses: _Lenses,
        populations: tuple[Sequence[str], Sequence[str]],
        place: str,
    ) -> list[PlaceRead]:
        """``places`` re-ranked by the layer-contrastive z.  ``populations``
        names each row's R and S population, ``signs`` its sign against it."""
        layer, site = places[0].layer, places[0].site
        r_names, s_names = populations
        r_halves = [place_.r for place_ in places]
        s_halves = [place_.s for place_ in places]
        if any(half.present for half in r_halves):
            stats = self._neuron_contrast(lenses, layer, "R", place)
            if stats is not None:
                matrix, label = self._r_matrix(lenses, layer, site)
                logits = self._r_logits(rows, matrix)
                z = torch.stack([
                    (logits[i] - sign * stats.mean_std(name)[0]) / stats.mean_std(name)[1]
                    for i, (name, sign) in enumerate(zip(r_names, signs, strict=True))
                ])
                lists = self._top_tokens(self._masked(z), k)
                r_halves = [
                    UnitHalf("R", half.weight, "layer_z", half.lens or label, tokens)
                    if half.present else half
                    for half, tokens in zip(r_halves, lists, strict=True)
                ]
        if any(half.present for half in s_halves):
            stats = self._neuron_contrast(lenses, layer, "S", place)
            matrix = self._matrix(lenses.s, layer, site)
            if stats is not None and matrix is not None:
                u = self._s_covectors(rows, matrix)
                unit_u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                scores = self._raw_scores(unit_u, 0, self._embeddings().vocab)
                z = torch.stack([
                    (scores[i] - sign * stats.mean_std(name)[0]) / stats.mean_std(name)[1]
                    for i, (name, sign) in enumerate(zip(s_names, signs, strict=True))
                ])
                lists = self._top_tokens(self._masked(z), k)
                s_halves = [
                    UnitHalf("S", half.weight, "layer_z", half.lens, tokens) if half.present else half
                    for half, tokens in zip(s_halves, lists, strict=True)
                ]
        return [
            PlaceRead(p.layer, p.site, p.depth, s, r, hubs="hidden")
            for p, s, r in zip(places, s_halves, r_halves, strict=True)
        ]

    # --------------------------------------------------------------- neurons

    def _null_basis(self) -> torch.Tensor:
        """The bottom :data:`NULL_SHARE` right-singular directions of
        ``W_U diag(γ_final)``, ``[d, m]`` on the device (built once)."""
        null = self._null
        if null is None:
            unembed = self._unembed()
            gain = self._final_gain()
            d = int(unembed.shape[1])
            gram = torch.zeros(d, d, dtype=torch.float32, device=self.device)
            for start in range(0, int(unembed.shape[0]), VOCAB_CHUNK):
                rows = unembed[start:start + VOCAB_CHUNK].detach().float() * gain
                gram += rows.T @ rows
            vectors = torch.linalg.eigh(gram.cpu().double())[1]  # ascending
            keep = max(1, int(NULL_SHARE * d))
            null = self._null = vectors[:, :keep].float().to(self.device)
        return null

    def neuron(self, ref: NeuronRef, *, k: int = 8, hide_hubs: bool = False) -> NeuronRead:
        """Read one neuron (both poles) from its weights."""
        k = self._check_k(k)
        self.runtime.check(ref)
        with self._section("unit read"), torch.inference_mode():
            lenses = self._lenses()
            block = self.runtime.block(ref.layer)
            from drowse.model.units import neuron_read, neuron_write

            w = neuron_write(block, ref.index).to(self.device)
            plus = neuron_read(block, ref.index, "+").to(self.device)
            minus = neuron_read(block, ref.index, "-").to(self.device)
            # ‖â + b̂‖² − ‖â − b̂‖² = 4 â·b̂
            cos_ab = (plus.pow(2).sum() - minus.pow(2).sum()) / 4
            writes_rows = torch.stack([w, -w])
            fires_rows = torch.stack([plus, minus])
            writes = self._place(writes_rows, ref.layer, "out", k=k, lenses=lenses)
            fires = self._place(fires_rows, ref.layer, "mid", k=k, lenses=lenses)
            if hide_hubs:
                writes = self._hubless(
                    writes, writes_rows, (1.0, -1.0), k=k, lenses=lenses,
                    populations=(("writes", "writes"), ("writes", "writes")), place="writes",
                )
                fires = self._hubless(
                    fires, fires_rows, (1.0, 1.0), k=k, lenses=lenses,
                    populations=(("fires+", "fires-"), ("fires", "fires")), place="fires",
                )
            logits = self._masked_logit_lens(w)
            skew, kurt = logit_moments(logits[None])
            null = self._null_basis()
            scalars = torch.stack([
                w.norm(), cos_ab, (null.T @ w).pow(2).sum() / w.pow(2).sum().clamp_min(1e-30),
                skew[0], kurt[0],
            ]).cpu().tolist()
        w_norm, cos_ab_v, null_share, skew_v, kurt_v = (float(x) for x in scalars)
        firing = self._neuron_firing(ref)
        labels = self._neuron_labels(ref)
        poles = tuple(
            NeuronPoleRead(
                pole, writes[i], fires[i],
                vocab_skew=skew_v if pole == "+" else -skew_v,
                vocab_kurt=kurt_v,
                firing=firing[i],
                label=labels[i],
            )
            for i, pole in enumerate(("+", "-"))
        )
        return NeuronRead(
            ref=ref,
            depth=layer_depth(ref.layer, self.n_layers),
            w_norm=w_norm, cos_ab=cos_ab_v, null_share=null_share,
            mean_act=self._neutral_mean(ref),
            poles=(poles[0], poles[1]),
            catalog=self._catalog_block(ref),
        )

    def _masked_logit_lens(self, w: torch.Tensor) -> torch.Tensor:
        """The write's logit-lens logits over the token mask, ``[V_kept]``."""
        logits = self._r_logits(w[None], None)[0]
        kept = self._kept()
        return logits.index_select(0, kept[kept < logits.shape[0]])

    # --------------------------------------------------------- firing / labels

    def _stats(self) -> tuple[Any, str | None]:
        try:
            return self.runtime.stats(), None
        except UnitsUnavailableError as exc:
            return None, str(exc)

    def _neutral_mean(self, ref: NeuronRef) -> float | None:
        stats, _ = self._stats()
        if stats is None:
            return None
        return float(stats.layer(ref.layer).neuron_mean[ref.index])

    def _neuron_firing(self, ref: NeuronRef) -> tuple[Firing, Firing]:
        stats, reason = self._stats()
        catalog, status = self.catalog.view()
        cat_absent = status["reason"]
        out: list[Firing] = []
        for row, pole in enumerate((1, -1)):
            neutral = None
            if stats is not None:
                layer = stats.layer(ref.layer)
                neutral = {
                    "act_top": round(float(layer.neuron_scale[row, ref.index]), 6),
                    # What this pole's readings divide by (stats.soft_scale).
                    "scale": round(float(layer.neuron_reading_scale[row, ref.index]), 6),
                    "mean": round(float(layer.neuron_mean[ref.index]), 6),
                    "documents": float(stats.n_documents),
                }
            cat_row: dict[str, Any] | None = None
            absent = cat_absent
            if catalog is not None:
                cat_row = catalog.neuron_acts(ref.layer, ref.index, pole)
                if cat_row is None:
                    absent = "the catalog has no activation table for this unit"
            out.append(Firing(neutral, reason, cat_row, absent))
        return out[0], out[1]

    def _neuron_labels(self, ref: NeuronRef) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        catalog = self.catalog.get()
        if catalog is None:
            return None, None
        plus = catalog.neuron_label(ref.layer, ref.index, 1)
        minus = catalog.neuron_label(ref.layer, ref.index, -1)
        return plus, minus

    def _catalog_summary(self) -> dict[str, Any]:
        """The binding without a unit: bound, slug, credit or why not (one
        view of the binding, so ``bound`` and ``reason`` always agree)."""
        catalog, status = self.catalog.view()
        return {
            "bound": catalog is not None,
            "slug": status.get("slug"),
            "credit": LABEL_CREDIT if catalog is not None else None,
            "reason": status["reason"],
        }

    def _catalog_block(self, ref: UnitRef) -> dict[str, Any]:
        """The catalog binding as a unit view shows it: bound or why not, the
        unit's neuron-lens.com page and the label credit."""
        catalog, status = self.catalog.view()
        return {
            "bound": catalog is not None,
            "slug": status.get("slug"),
            "url": None if catalog is None else catalog.url(ref),
            "credit": LABEL_CREDIT if catalog is not None else None,
            "reason": status["reason"],
        }

    # ------------------------------------------------------------------ heads
    #
    # A head's reads are products of [V, d] vocabulary rows with a few
    # [d, d] lens matrices and its [d, hd] slices.  Every product here takes
    # the small factors first (``J W_O^h``, ``K^ᵀ W_V^{h⊤}``: [d, hd]) and
    # streams the vocabulary rows through them in chunks, so a read holds
    # [V, hd] at most, never [V, d] (2.5 GB per tensor on Qwen3-8B).

    def _head_kept(self) -> torch.Tensor:
        """The kept token ids a head read ranks (the mask's, inside the
        unembedding the R side reads), on the device."""
        kept = self._kept()
        return kept[kept < int(self._unembed().shape[0])]

    def _chunks(self, ids: torch.Tensor, width: int) -> Iterator[torch.Tensor]:
        """``ids`` in chunks of :data:`VOCAB_CHUNK`, fewer when a chunk's
        ``[rows, width]`` product would pass :data:`_CHUNK_ELEMENTS`."""
        step = max(1, min(VOCAB_CHUNK, _CHUNK_ELEMENTS // max(int(width), 1)))
        for start in range(0, int(ids.shape[0]), step):
            yield ids[start:start + step]

    def _vocab_rows(self, kind: RowKind, ids: torch.Tensor, gain: torch.Tensor | None = None) -> torch.Tensor:
        """``[n, d]`` fp32 rows of the vocabulary ``ids``: the centred
        embeddings ``ẽ_v`` (``embed``) or ``γ_f ⊙ W_U[v]`` (``unembed``, with
        ``gain = γ_f``)."""
        if kind == "embed":
            return self._embeddings().centred(ids)
        assert gain is not None
        return self._unembed().index_select(0, ids).detach().float() * gain

    # ------------------------------------------------------------ head norms

    def _block_inputs(self, layer: int) -> tuple[object, Callable[[], Iterator[torch.Tensor]]]:
        """``(identity, chunks)``: the residuals entering block ``layer`` that
        its head norms' derivative is averaged over.  The neutral residuals
        at ``layer − 1``'s output (the whitener's rows: every neutral
        statement at its pooled position); at L0, where a block reads the
        token embedding, every kept token's embedding.  Raises
        :class:`~drowse.errors.WhitenerError` without the neutral rows."""
        if layer == 0:
            emb = self._embeddings()
            ids = self._head_kept()

            def embedded() -> Iterator[torch.Tensor]:
                for chunk in self._chunks(ids, emb.d_emb):
                    yield emb.weight.index_select(0, chunk).detach().float() * emb.scale

            return emb, embedded
        whitener = None if self._whitener is None else self._whitener()
        if whitener is None or (layer - 1) not in whitener:
            raise WhitenerError(
                f"a head of block {layer} reads through its head norms, whose derivative "
                f"is averaged over the neutral residuals at L{layer - 1}; the neutral "
                f"baseline does not cover that layer"
            )
        centred, _small, _ridge = whitener.woodbury_factors(layer - 1, device=self.device, dtype=torch.float32)
        rows = centred + whitener.layer_means[layer - 1].to(device=self.device, dtype=torch.float32)
        return whitener, lambda: iter((rows,))

    def _head_norms(self, block: BlockSpec) -> dict[str, torch.Tensor | None]:
        """Per projection (``query``, ``key``, ``value``) its head norm's true
        derivative averaged over the residuals entering the block, ``[heads,
        hd, hd]`` (query heads for ``query``, KV heads otherwise), or ``None``
        where the block has no such norm.

        As neuron-lens's ``heads.norm_derivative``: with ``x`` the projection's
        output before its norm and ``r = rsqrt(mean(x²) + ε)``, ``diag(g)·E[w(r I
        − r³ x xᵀ / hd)]``, each position weighted by the input norm's scale
        ``w = 1/ρ`` (what a footprint direction is multiplied by before the
        projection).  Kept per block until the neutral baseline changes."""
        attn = block.attn
        norms: dict[str, NormSpec | None] = {"query": attn.q_norm, "key": attn.k_norm, "value": attn.v_norm}
        if all(norm is None for norm in norms.values()):
            return dict.fromkeys(_HEAD_ROLES)
        identity, inputs = self._block_inputs(block.layer)
        hit = self._norm_cache.get(block.layer)
        if hit is not None and hit[0] is identity:
            return hit[1]
        hd = attn.head_dim
        in_norm = block.attn_in_norm
        gain_in = in_norm.gain(device=self.device)
        sources = {"query": attn.q, "key": attn.k, "value": attn.k if attn.v is None else attn.v}
        weights = {
            role: (
                sources[role].weight.detach().to(device=self.device, dtype=torch.float32) * gain_in,
                None if sources[role].bias is None
                else sources[role].bias.detach().to(device=self.device, dtype=torch.float32),
            )
            for role, norm in norms.items() if norm is not None
        }
        first: dict[str, torch.Tensor] = {}
        third: dict[str, torch.Tensor] = {}
        count = 0
        for rows in inputs():
            inv_rho = torch.rsqrt(rows.pow(2).mean(-1) + in_norm.eps)            # [n]
            normed = rows * inv_rho[:, None]
            for role, (weight, bias) in weights.items():
                x = normed @ weight.T
                if bias is not None:
                    x = x + bias
                x = x.view(int(rows.shape[0]), -1, hd)
                norm = norms[role]
                assert norm is not None
                r = torch.rsqrt(x.pow(2).mean(-1) + norm.eps)                     # [n, heads]
                w = r * inv_rho[:, None]
                s1 = w.sum(0)
                s3 = torch.einsum("nh,nhi,nhj->hij", w * r.pow(2), x, x)
                first[role] = s1 if role not in first else first[role] + s1
                third[role] = s3 if role not in third else third[role] + s3
            count += int(rows.shape[0])
        eye = torch.eye(hd, device=self.device)
        out: dict[str, torch.Tensor | None] = {}
        for role, norm in norms.items():
            if norm is None:
                out[role] = None
                continue
            gain = norm.gain(device=self.device)
            out[role] = gain[None, :, None] * (first[role][:, None, None] * eye - third[role] / hd) / count
        self._norm_cache[block.layer] = (identity, out)
        return out

    def _norms_digest(self, norms: Mapping[str, torch.Tensor | None]) -> str:
        """What a head population depends on through the head norms."""
        import hashlib

        if all(value is None for value in norms.values()):
            return "none"
        digest = hashlib.sha256()
        for role in _HEAD_ROLES:
            value = norms.get(role)
            if value is not None:
                digest.update(role.encode())
                digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()[:16]

    def _projection(
        self, block: BlockSpec, head: int, role: str, norms: Mapping[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Head ``head``'s ``role`` rows times ``γ_attn`` through its head
        norm's mean derivative, ``[hd, d]`` on the device."""
        rows = getattr(head_read_rows(block, head), role).to(self.device)
        derivative = norms[role]
        if derivative is None:
            return rows
        index = head if role == "query" else block.attn.kv_head(head)
        return derivative[index] @ rows

    # --------------------------------------------------------------- OV map

    def _footprint_map(self, lenses: _Lenses, layer: int) -> tuple[torch.Tensor | None, bool]:
        """``(T, present)``: a token's footprint at ``layer``'s output is
        ``ẽ_v T``, ``T = K^{out⊤}_{layer}`` ``[d_emb, d]``; at the embedding
        (``layer < 0``) the footprint is ``ẽ_v`` itself (``T = None``)."""
        if layer < 0:
            return None, True
        matrix = self._matrix(lenses.s, layer, "out")
        return (None, False) if matrix is None else (matrix.T, True)

    def _b_factor(
        self, lenses: _Lenses, block: BlockSpec, heads: Sequence[int],
        norms: Mapping[str, torch.Tensor | None],
    ) -> torch.Tensor | None:
        """``[d_emb, H'·hd]``, so that ``B = ẽ ·`` it: per head
        ``K^{out⊤}_{L−1} W_V^{h⊤} D_v^{h⊤}`` (the footprint through the value
        rows and the value norm's mean derivative), or ``None`` without the S
        matrix at ``L − 1``."""
        through, present = self._footprint_map(lenses, block.layer - 1)
        if not present:
            return None
        factor = torch.cat([self._projection(block, h, "value", norms).T for h in heads], dim=1)
        return factor if through is None else through @ factor

    def _a_factor(
        self, lenses: _Lenses, block: BlockSpec, heads: Sequence[int], family: Family,
    ) -> tuple[RowKind, torch.Tensor, str] | None:
        """``(rows, factor, label)``: ``A = rows · factor`` reads each output
        token in the heads' writes at ``mid`` (R: ``γ_f ⊙ W_U[u]`` times
        ``J^{mid}_L W_O^h``; S: ``ẽ_u`` times ``K^{mid⊤}_L W_O^h``), or
        ``None`` without that lens's ``mid`` matrix."""
        write = torch.cat([head_write_matrix(block, h) for h in heads], dim=1).to(self.device)
        if family == "R":
            j_mid = self._matrix(lenses.r, block.layer, "mid")
            if j_mid is None:
                return None
            return "unembed", j_mid @ write, lenses.r.site_key("mid").label
        k_mid = self._matrix(lenses.s, block.layer, "mid")
        if k_mid is None:
            return None
        return "embed", k_mid.T @ write, lenses.s.site_key("mid").label

    def _ov_maps(
        self,
        sides: Mapping[Family, tuple[RowKind, torch.Tensor, str]],
        b_factor: torch.Tensor,
        ids: torch.Tensor,
        hd: int,
    ) -> dict[Family, tuple[torch.Tensor, torch.Tensor]]:
        """Per family each head's own-token z ``[H', V]`` and ``Bᵀ A``
        ``[H', hd, hd]`` of its OV map ``M[u, v] = A_u · B_v``.

        :func:`own_token_z` and :func:`copy_score` over the vocabulary in two
        streamed passes: the first merges ``A``'s mean and scatter chunk by
        chunk (Chan et al.'s pairwise update) and sums ``Bᵀ A``, the second
        takes each token's z from them.  ``A`` and ``B`` exist one chunk at a
        time."""
        n_heads = int(b_factor.shape[1]) // hd
        width = int(b_factor.shape[1]) * (1 + len(sides))
        gain = self._final_gain() if any(kind == "unembed" for kind, _f, _l in sides.values()) else None

        def products(chunk: torch.Tensor) -> tuple[torch.Tensor, dict[Family, torch.Tensor]]:
            embedded = self._vocab_rows("embed", chunk)
            unembedded: torch.Tensor | None = None
            B = (embedded @ b_factor).view(-1, n_heads, hd)
            out: dict[Family, torch.Tensor] = {}
            for family, (kind, factor, _label) in sides.items():
                if kind == "unembed":
                    if unembedded is None:
                        unembedded = self._vocab_rows("unembed", chunk, gain)
                    rows = unembedded
                else:
                    rows = embedded
                out[family] = (rows @ factor).view(-1, n_heads, hd)
            return B, out

        zeros = torch.zeros(n_heads, hd, dtype=torch.float32, device=self.device)
        count = 0
        mean = dict.fromkeys(sides, zeros)
        scatter = dict.fromkeys(sides, zeros[:, :, None] * zeros[:, None, :])
        bta = dict(scatter)
        for chunk in self._chunks(ids, width):
            B, As = products(chunk)
            n = int(B.shape[0])
            total = count + n
            for family, A in As.items():
                chunk_mean = A.mean(0)
                centred = A - chunk_mean
                delta = chunk_mean - mean[family]
                mean[family] = mean[family] + delta * (n / total)
                scatter[family] = (
                    scatter[family] + torch.einsum("vhi,vhj->hij", centred, centred)
                    + delta[:, :, None] * delta[:, None, :] * (count * n / total)
                )
                bta[family] = bta[family] + torch.einsum("vhi,vhj->hij", B, A)
            count = total
        cov = {family: scatter[family] / count for family in sides}
        zs: dict[Family, list[torch.Tensor]] = {family: [] for family in sides}
        for chunk in self._chunks(ids, width):
            B, As = products(chunk)
            for family, A in As.items():
                own = (A * B).sum(-1) - (B * mean[family]).sum(-1)
                var = torch.einsum("vhi,hij,vhj->vh", B, cov[family], B)
                zs[family].append(own / var.clamp_min(1e-30).sqrt())
        return {family: (torch.cat(zs[family]).T.contiguous(), bta[family]) for family in sides}

    def _head_contrast(
        self, lenses: _Lenses, block: BlockSpec, family: Family,
        norms: Mapping[str, torch.Tensor | None],
    ) -> ContrastStats:
        """Mean and (unbiased) std across the layer's heads of each token's
        OV own-token z, as neuron-lens's head ``_z`` lists (one streamed
        pass over every head of the layer)."""
        layer = block.layer
        identity = lenses.r_identity if family == "R" else lenses.s_identity

        def build() -> dict[str, torch.Tensor]:
            heads = range(block.n_heads)
            side = self._a_factor(lenses, block, heads, family)
            b_factor = self._b_factor(lenses, block, heads, norms)
            assert side is not None and b_factor is not None
            per_head = self._ov_maps({family: side}, b_factor, self._head_kept(), block.attn.head_dim)[family][0]
            _drain(self.device)
            return {
                "ov.mean": per_head.mean(0),
                "ov.std": per_head.std(0).clamp_min(_HEAD_STD_FLOOR),
            }

        return self.contrast.get(
            f"{family}-{identity}-S-{lenses.s_identity}-heads-ov", layer,
            self._meta(family=family, what="ov", norms=self._norms_digest(norms)), self.device, build,
        )

    def head(self, ref: HeadRef, *, k: int = 8, hide_hubs: bool = False) -> HeadRead:
        """Read one head from its weights: OV copies and suppressions at
        ``mid``, its inputs at ``L − 1`` out, the copy scores."""
        k = self._check_k(k)
        self.runtime.check(ref)
        layer, head = ref.layer, ref.index
        hubs: Literal["shown", "hidden"] = "hidden" if hide_hubs else "shown"
        depth = layer_depth(layer, self.n_layers)
        with self._section("unit read"), torch.inference_mode():
            lenses = self._lenses()
            block = self.runtime.block(layer)
            ids = self._head_kept()
            ids_host = ids.cpu().tolist()
            norms = self._head_norms(block)
            w_s, w_r = self._weights(layer)
            b_factor = self._b_factor(lenses, block, (head,), norms)
            b_absent = None if b_factor is not None else (
                f"a head's input side reads through the S lens at L{layer - 1} out, "
                f"and {'no S-family lens is selected' if lenses.s is None else 'it holds no such matrix'}"
            )
            sides: dict[Family, tuple[RowKind, torch.Tensor, str]] = {}
            for family in ("R", "S"):
                side = self._a_factor(lenses, block, (head,), family)
                if side is not None:
                    sides[family] = side
            maps = {} if b_factor is None or not sides else self._ov_maps(
                sides, b_factor, ids, block.attn.head_dim,
            )
            copies: dict[Family, UnitHalf] = {}
            suppresses: dict[Family, UnitHalf] = {}
            scores: dict[str, float | None] = {"R": None, "S": None}
            families: tuple[tuple[Family, float], ...] = (("R", w_r), ("S", w_s))
            for family, weight in families:
                label = sides[family][2] if family in sides else None
                if family not in maps:
                    why = b_absent if b_factor is None else self._no_lens(lenses, family, layer, "mid")
                    copies[family] = suppresses[family] = UnitHalf(
                        family, weight, "own_token_z", label, absent=why,
                    )
                    continue
                per_head, bta = maps[family]
                z = per_head[0]
                scores[family] = _copy_score_of(bta[0])
                unit: HalfUnit = "own_token_z"
                if hide_hubs:
                    mean, std = self._head_contrast(lenses, block, family, norms).mean_std("ov")
                    z = (z - mean) / std
                    unit = "layer_z"
                lists = self._top_tokens(torch.stack([z, -z]), k)
                copies[family] = UnitHalf(family, weight, unit, label, self._remap(lists[0], ids_host))
                suppresses[family] = UnitHalf(
                    family, weight, unit, label,
                    tuple(TokenScore(t.token_id, t.token, -t.value) for t in self._remap(lists[1], ids_host)),
                )
            copies_place = PlaceRead(layer, "mid", depth, copies["S"], copies["R"], hubs)
            suppress_place = PlaceRead(layer, "mid", depth, suppresses["S"], suppresses["R"], hubs)
            inputs = self._head_inputs(lenses, block, head, norms, k=k, ids=ids, ids_host=ids_host)
            _drain(self.device)
        return HeadRead(
            ref=ref,
            depth=depth,
            kv_head=int(block.attn.kv_head(head)),
            copies=copies_place,
            suppresses=suppress_place,
            inputs=inputs,
            copy_score=scores,
            firing=self._head_firing(ref),
            label=self._head_label(ref),
            catalog=self._catalog_block(ref),
        )

    def _remap(self, tokens: Iterable[TokenScore], ids_host: Sequence[int]) -> tuple[TokenScore, ...]:
        """Kept-row indices back to vocabulary ids."""
        return tuple(
            TokenScore(int(ids_host[t.token_id]), self._decode(int(ids_host[t.token_id])), t.value)
            for t in tokens
        )

    def _no_lens(self, lenses: _Lenses, family: Family, layer: int, site: str) -> str:
        lens = lenses.r if family == "R" else lenses.s
        if lens is None:
            return f"no {family}-family lens is selected for {self.model_id}"
        return f"the {lens.kind} lens holds no {site} matrix at L{layer}"

    # --------------------------------------------------------------- inputs

    def _input_energies(
        self, kind: RowKind, through: torch.Tensor | None, projections: torch.Tensor,
        ids: torch.Tensor, hd: int,
    ) -> torch.Tensor:
        """``[m, V]``: per projection (``projections`` ``[d, m·hd]``, side by
        side) the vocabulary z of ``‖P x̂_v‖``, ``x_v = rows_v · through``
        (``through`` ``[d_in, d]``; ``None``: the rows themselves), streamed
        over vocabulary chunks."""
        gain = self._final_gain() if kind == "unembed" else None
        factor = projections if through is None else through @ projections
        m = int(factor.shape[1]) // hd
        width = int(factor.shape[1]) + (0 if through is None else int(through.shape[1]))
        parts: list[torch.Tensor] = []
        for chunk in self._chunks(ids, width):
            rows = self._vocab_rows(kind, chunk, gain)
            norm = (rows if through is None else rows @ through).norm(dim=-1).clamp_min(1e-12)
            parts.append((rows @ factor).view(-1, m, hd).norm(dim=-1) / norm[:, None])
        energy = torch.cat(parts).T
        centred = energy - energy.mean(-1, keepdim=True)
        return centred / energy.std(-1, correction=0, keepdim=True).clamp_min(1e-12)

    def _head_inputs(
        self, lenses: _Lenses, block: BlockSpec, head: int,
        norms: Mapping[str, torch.Tensor | None], *, k: int,
        ids: torch.Tensor, ids_host: Sequence[int],
    ) -> dict[str, PlaceRead]:
        """What each projection picks up at ``L − 1``'s output (the token
        embedding at L0): the S half over footprints, the R half over the
        directions that say each token, each projection through its head
        norm's mean derivative where the block has one.

        Always the vocabulary z (hubs shown): a layer contrast across heads
        saturates here, since the heads of one KV group share their key and
        value rows (with ``g`` of ``n`` heads alike, every token tops out at
        the same z), and a few KV heads are too small a population."""
        layer = block.layer
        at = layer - 1
        site_layer = max(at, 0)
        w_s, w_r = self._weights(site_layer)
        depth = layer_depth(site_layer, self.n_layers)
        hd = block.attn.head_dim
        projections = torch.cat(
            [self._projection(block, head, role, norms).T for role in _HEAD_ROLES], dim=1,
        )
        energies: dict[Family, torch.Tensor] = {}
        through, s_present = self._footprint_map(lenses, at)
        if s_present:
            energies["S"] = self._input_energies("embed", through, projections, ids, hd)
        j_out = None if at < 0 else self._matrix(lenses.r, at, "out")
        if j_out is not None:
            # x_v = J^{out⊤}_{L−1}(γ_f ⊙ W_U[v]): as a row, (γ_f ⊙ W_U[v]) J.
            energies["R"] = self._input_energies("unembed", j_out, projections, ids, hd)
        labels: dict[Family, str | None] = {
            "S": "embedding" if at < 0 else (
                None if lenses.s is None else lenses.s.site_key("out").label
            ),
            "R": None if lenses.r is None or at < 0 else lenses.r.site_key("out").label,
        }
        families: tuple[tuple[Family, float], ...] = (("S", w_s), ("R", w_r))
        out: dict[str, PlaceRead] = {}
        for index, role in enumerate(_HEAD_ROLES):
            halves: dict[Family, UnitHalf] = {}
            for family, weight in families:
                if family not in energies:
                    why = (
                        "the embedding has no R read (a head at L0 reads the token embeddings)"
                        if family == "R" and at < 0 else self._no_lens(lenses, family, at, "out")
                    )
                    halves[family] = UnitHalf(family, weight, "energy_z", labels[family], absent=why)
                    continue
                z = energies[family][index]
                tokens = self._remap(self._top_tokens(z[None], k)[0], ids_host)
                halves[family] = UnitHalf(family, weight, "energy_z", labels[family], tokens)
            out[role] = PlaceRead(site_layer, "out", depth, halves["S"], halves["R"])
        return out

    def _head_firing(self, ref: HeadRef) -> Firing:
        stats, reason = self._stats()
        neutral = None
        if stats is not None:
            layer = stats.layer(ref.layer)
            neutral = {
                "write_top": round(float(layer.head_scale[ref.index]), 6),
                "scale": round(float(layer.head_reading_scale[ref.index]), 6),
                "mean_z_norm": round(float(layer.head_mean[ref.index].norm()), 6),
                "documents": float(stats.n_documents),
            }
        catalog, status = self.catalog.view()
        if catalog is None:
            return Firing(neutral, reason, None, status["reason"])
        row = catalog.head_acts(ref.layer, ref.index)
        return Firing(
            neutral, reason, row,
            None if row is not None else "the catalog has no activation table for this unit",
        )

    def _head_label(self, ref: HeadRef) -> dict[str, Any] | None:
        catalog = self.catalog.get()
        return None if catalog is None else catalog.head_label(ref.layer, ref.index)

    # ------------------------------------------------------------------ read

    def read(self, unit: str | UnitRef, *, k: int = 8, hide_hubs: bool = False) -> UnitDetail:
        """``neuron/L8n344`` or ``head/L20h17`` (or a parsed reference)."""
        ref = self._parse(unit)
        if isinstance(ref, NeuronRef):
            return self.neuron(ref, k=k, hide_hubs=hide_hubs)
        assert isinstance(ref, HeadRef)
        return self.head(ref, k=k, hide_hubs=hide_hubs)

    def labels(self, units: Sequence[str | UnitRef]) -> dict[str, Any]:
        """The catalog's labels for ``units`` (disk only: no lens and no model
        work), one row per neuron pole (``+`` then ``-``) and one per head, in
        the order asked, each unit once; ``label`` is ``None`` where the
        catalog has no row.  ``catalog`` is the binding as the unit views show
        it; unbound, there are no rows and ``reason`` says why."""
        refs = list(dict.fromkeys(self._parse(unit) for unit in units))
        catalog, status = self.catalog.view()
        rows: list[dict[str, Any]] = []
        if catalog is not None:
            for ref in refs:
                if isinstance(ref, NeuronRef):
                    rows.extend(
                        {"unit": ref.qualified, "pole": pole,
                         "label": catalog.neuron_label(ref.layer, ref.index, sign)}
                        for sign, pole in ((1, "+"), (-1, "-"))
                    )
                else:
                    rows.append({"unit": ref.qualified, "pole": None,
                                 "label": catalog.head_label(ref.layer, ref.index)})
        return {
            "catalog": {
                "bound": catalog is not None,
                "slug": status.get("slug"),
                "credit": LABEL_CREDIT if catalog is not None else None,
                "reason": status["reason"],
            },
            "labels": rows,
        }

    def _parse(self, unit: str | UnitRef) -> UnitRef:
        if isinstance(unit, (NeuronRef, HeadRef)):
            return self.runtime.check(unit)
        namespace, sep, name = str(unit).partition("/")
        if not sep or namespace not in (NeuronRef.namespace, HeadRef.namespace):
            raise RequestError(f"a unit is neuron/L<layer>n<index> or head/L<layer>h<index>, got {unit!r}")
        return self.runtime.check(parse_unit_ref(namespace, name))

    def stats(self, unit: str | UnitRef) -> dict[str, Any]:
        """How a unit fires: neutral statistics and the catalog's activation
        row (per pole for a neuron), without a lens read."""
        ref = self._parse(unit)
        if isinstance(ref, NeuronRef):
            firing = self._neuron_firing(ref)
            labels = self._neuron_labels(ref)
            return {
                "unit": ref.qualified, "kind": "neuron", "layer": ref.layer, "index": ref.index,
                "poles": [
                    {"pole": pole, "firing": firing[i].to_dict(), "label": labels[i]}
                    for i, pole in enumerate(("+", "-"))
                ],
                "catalog": self._catalog_block(ref),
            }
        assert isinstance(ref, HeadRef)
        return {
            "unit": ref.qualified, "kind": "head", "layer": ref.layer, "index": ref.index,
            "firing": self._head_firing(ref).to_dict(),
            "label": self._head_label(ref),
            "catalog": self._catalog_block(ref),
        }

    # ------------------------------------------------------------------- top

    def top(
        self,
        token: str,
        *,
        by: TopBy = "writes",
        layers: "Sequence[int] | str | None" = None,
        k: int = 20,
    ) -> TopResult:
        """Neuron poles that write (``by="writes"``) or fire on
        (``by="fires"``) ``token`` most specifically, over ``layers``
        (default every layer).

        Each pole scores ``Σ w_f z_f / Σ w_f`` over the families present at
        its place, ``z_f`` the token's layer-contrastive z there (the hub-free
        read, comparable across families and layers), ``w_f`` the depth
        rule's weight.  The first read of a layer builds its population
        statistics (cached on disk).
        """
        from drowse.model.layers import resolve_layer_selector

        if by not in ("writes", "fires"):
            raise RequestError(f"by must be 'writes' or 'fires', got {by!r}")
        k = self._check_k(k)
        selected = resolve_layer_selector(
            _selector(layers), list(range(self.n_layers)), n_layers=self.n_layers, allow_sample=False,
        )
        for layer in selected:
            if not 0 <= int(layer) < self.n_layers:
                raise RequestError(f"layer {layer} is outside the model's {self.n_layers} layers")
        token_id = self.lens.resolve_token(token, side="input")
        site = "out" if by == "writes" else "mid"
        candidates: list[tuple[float, int, int, int, dict[str, Any], dict[str, Any]]] = []
        info: dict[int, dict[str, Any]] = {}
        with self._section("unit top"), torch.inference_mode():
            lenses = self._lenses()
            emb = self._embeddings()
            if not bool(emb.keep[token_id]):
                raise RequestError(
                    f"{token!r} is token {token_id}, which the token mask drops "
                    f"(special, added and byte-fallback pieces never rank)"
                )
            ids = torch.tensor([token_id], device=self.device)
            for layer in sorted({int(layer) for layer in selected}):
                scored = self._top_layer(lenses, layer, site, by, ids, emb)
                info[layer] = scored["info"]
                candidates.extend(scored["rows"])
                _drain(self.device)
        candidates.sort(key=lambda row: -row[0])
        catalog = self.catalog.get()
        hits = []
        for score, layer, index, pole_row, r, s in candidates[:k]:
            ref = NeuronRef(layer=layer, index=index)
            pole_sign = 1 if pole_row == 0 else -1
            hits.append(TopHit(
                ref, "+" if pole_sign > 0 else "-", score, r, s,
                None if catalog is None else catalog.neuron_label(layer, index, pole_sign),
            ))
        return TopResult(token_id, self._decode(token_id), by, tuple(hits), info)

    def _top_layer(
        self, lenses: _Lenses, layer: int, site: str, by: TopBy,
        ids: torch.Tensor, emb: "TokenEmbeddings",
    ) -> dict[str, Any]:
        """One layer's pole scores for token ``ids[0]``: ``k`` best rows of
        each layer would bias across layers, so every pole of the layer is
        scored and the best few kept (one host transfer)."""
        block = self.runtime.block(layer)
        writes, plus, minus = self._neuron_populations(block)
        f = int(writes.shape[0])
        w_s, w_r = self._weights(layer)
        r_z = r_p = s_z = s_fz = None
        r_label = s_label = None
        place = "writes" if by == "writes" else "fires"
        r_stats = self._neuron_contrast(lenses, layer, "R", place)
        if r_stats is not None:
            matrix, r_label = self._r_matrix(lenses, layer, site)
            token = int(ids[0])
            if by == "writes":
                lg = self._token_logit_rows(writes, matrix, ids)[:, 0]          # [f]
                mean, std = r_stats.mean_std("writes")
                z_plus = (lg - mean[token]) / std[token]
                lse, lse_neg = r_stats.get("writes.lse"), r_stats.get("writes.lse_neg")
                r_z = torch.stack([z_plus, -z_plus])
                r_p = torch.stack([(lg - lse).exp(), (-lg - lse_neg).exp()])
            else:
                rows_z, rows_p = [], []
                for name, rows in (("fires+", plus), ("fires-", minus)):
                    lg = self._token_logit_rows(rows, matrix, ids)[:, 0]
                    mean, std = r_stats.mean_std(name)
                    rows_z.append((lg - mean[token]) / std[token])
                    rows_p.append((lg - r_stats.get(f"{name}.lse")).exp())
                r_z, r_p = torch.stack(rows_z), torch.stack(rows_p)
        s_stats = self._neuron_contrast(lenses, layer, "S", place)
        s_matrix = self._matrix(lenses.s, layer, site)
        if s_stats is not None and s_matrix is not None:
            s_label = lenses.s.site_key(site).label
            token = int(ids[0])
            e_v = emb.centred(ids)[0]
            pole_rows = (writes, -writes) if by == "writes" else (plus, minus)
            name_sign = (("writes", 1.0), ("writes", -1.0)) if by == "writes" else (("fires", 1.0), ("fires", 1.0))
            zs, fzs = [], []
            for rows, (name, sign) in zip(pole_rows, name_sign, strict=True):
                u = self._s_covectors(rows, s_matrix)
                norm = u.norm(dim=-1).clamp_min(1e-12)
                raw = u @ e_v
                mean, std = s_stats.mean_std(name)
                zs.append((raw / norm - sign * mean[token]) / std[token])
                fzs.append(raw / emb.variance(u).clamp_min(1e-30).sqrt())
            s_z, s_fz = torch.stack(zs), torch.stack(fzs)
        present_r = r_z is not None
        present_s = s_z is not None
        info = {
            "weights": {"S": round(w_s, 6), "R": round(w_r, 6)},
            "lens": {"S": s_label, "R": r_label},
            "absent": {
                "S": None if present_s else self._no_lens(lenses, "S", layer, site),
                "R": None if present_r else self._no_lens(lenses, "R", layer, site),
            },
        }
        if not present_r and not present_s:
            return {"info": info, "rows": []}
        wr = w_r if present_r else 0.0
        ws = w_s if present_s else 0.0
        total = wr + ws
        if total <= 0.0:  # the present half has no weight here: fall to it alone
            wr, ws, total = float(present_r), float(present_s), float(present_r) + float(present_s)
        score = torch.zeros(2, f, device=self.device)
        if r_z is not None:
            score = score + wr * r_z
        if s_z is not None:
            score = score + ws * s_z
        score = score / total
        keep = min(MAX_K, 2 * f)
        best, flat = score.reshape(-1).topk(keep)
        cols = [best]
        for tensor in (r_z, r_p, s_z, s_fz):
            cols.append(
                torch.full_like(best, float("nan")) if tensor is None
                else tensor.reshape(-1).index_select(0, flat)
            )
        host = torch.stack([*cols, flat.float()]).cpu().tolist()
        rows_out = []
        for j in range(len(host[0])):
            position = int(host[5][j])
            pole_row, index = divmod(position, f)
            rz, rp, sz, sfz = (host[i][j] for i in range(1, 5))
            rows_out.append((
                float(host[0][j]), layer, index, pole_row,
                {"z": _num(rz), "probability": _num(rp), "weight": round(w_r, 6), "lens": r_label},
                {"z": _num(sz), "footprint_z": _num(sfz), "weight": round(w_s, 6), "lens": s_label},
            ))
        return {"info": info, "rows": rows_out}

    def _token_logit_rows(
        self, rows: torch.Tensor, matrix: torch.Tensor | None, ids: torch.Tensor,
    ) -> torch.Tensor:
        """``[n, m]`` R logits of ``ids`` for residual rows, chunked."""
        out = []
        for start in range(0, int(rows.shape[0]), ROW_CHUNK * 4):
            x = rows[start:start + ROW_CHUNK * 4].to(torch.float32)
            if matrix is not None:
                x = x @ matrix.T
            out.append(token_logits(self._readout(), x, ids))
        return torch.cat(out)

    # ------------------------------------------------------------------ find

    def find(
        self,
        target: "str | torch.Tensor | Mapping[int, torch.Tensor]",
        *,
        layer: int | None = None,
        k: int = 16,
        layers: "Sequence[int] | str | None" = None,
    ) -> "FindResult":
        """Decompose a steerable selector (``honest``, ``jlens/Paris``,
        ``0.5 personas%pirate`` is not: one selector), a ``[d]`` vector at
        ``layer`` or a per-layer mapping over each layer's neuron writes
        (the ``neurons`` dictionary, Mahalanobis), with each neuron's pole
        and label."""
        k = self._check_k(k)
        if isinstance(target, str):
            label = target
            out = self.lens.decompose(target, k=k, layers=_selector(layers), dictionary="neurons")
        else:
            if isinstance(target, torch.Tensor):
                if layer is None:
                    raise RequestError("a vector needs the layer it lives at")
                if target.ndim != 1:
                    raise RequestError(f"find takes a [d] vector, got shape {tuple(target.shape)}")
                directions = {int(layer): target}
            else:
                directions = {int(key): value for key, value in target.items()}
            width = self.runtime.spec().d_model
            for value in directions.values():
                if value.ndim != 1 or int(value.shape[0]) != width:
                    raise RequestError(
                        f"find takes residual-width vectors (d = {width}), got shape {tuple(value.shape)}"
                    )
            label = "vector"
            out = self.lens.decompose_neurons(directions, k=k, layers=_selector(layers), label=label)
        catalog = self.catalog.get()
        rows: dict[int, dict[str, Any]] = {}
        for at, (share, atoms) in sorted(out.items()):
            neurons = []
            for name, coeff in atoms:
                ref = parse_unit_ref(NeuronRef.namespace, name.split("/", 1)[1])
                sign = 1 if coeff > 0 else -1
                neurons.append({
                    "unit": name,
                    "index": ref.index,
                    "pole": "+" if sign > 0 else "-",
                    "coeff": round(float(coeff), 6),
                    "label": None if catalog is None else catalog.neuron_label(ref.layer, ref.index, sign),
                })
            rows[int(at)] = {"share": round(float(share), 6), "neurons": neurons}
        return FindResult(label, k, rows, self._catalog_summary())

    # ------------------------------------------------------------- population

    def layer_contrast(self, layer: int) -> dict[str, Any]:
        """Build (or load) one layer's neuron contrast statistics for both
        families and places: the whole-layer read ``hide_hubs`` and ``top``
        rest on.  Returns which halves it covered."""
        with self._section("unit contrast"), torch.inference_mode():
            lenses = self._lenses()
            self.runtime.block(layer)
            covered = {
                f"{family}:{place}": self._neuron_contrast(lenses, layer, family, place) is not None
                for family in ("R", "S") for place in ("writes", "fires")
            }
            _drain(self.device)
        return covered


def _selector(layers: "Sequence[int] | str | None") -> Any:
    """A layer selector as the one parser reads it: a string spec
    (``"20-31"``, ``"3,7"``, ``workspace``) is parsed, anything else passes."""
    if not isinstance(layers, str):
        return None if layers is None else [int(layer) for layer in layers]
    from drowse.model.layers import parse_layer_selector

    try:
        return parse_layer_selector(layers)
    except ValueError as exc:
        raise RequestError(f"bad layers {layers!r}: {exc}") from None


def _num(value: float) -> float | None:
    return None if value != value else round(float(value), 6)  # NaN → None


__all__ = [
    "CONTRAST_FORMAT",
    "ContrastCache",
    "ContrastStats",
    "FindResult",
    "Firing",
    "HeadRead",
    "NeuronPoleRead",
    "NeuronRead",
    "PlaceRead",
    "TopHit",
    "TopResult",
    "UnitDetail",
    "UnitHalf",
    "UnitReader",
    "contrast_dir",
    "copy_score",
    "logit_moments",
    "full_logits",
    "own_token_z",
    "token_logits",
]
