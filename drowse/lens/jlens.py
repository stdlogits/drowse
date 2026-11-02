"""Jacobian lens artifact + readout: transport, readout math, decomposition.

Implements the Jacobian lens of Gurnee et al., "Verbalizable Representations
Form a Global Workspace in Language Models" (Transformer Circuits, 2026):
per-layer matrices ``J_l = E[∂h_final/∂h_l]`` — the average first-order effect
of a layer-l residual on the final-layer residual, over positions and a text
corpus.  ``lens(h) = softmax(W_U · norm(J_l h))`` ranks
the vocabulary by what an activation is disposed to make the model say; the
J-lens *direction* for vocab id v at layer l is ``W_U[v] @ J_l`` — a per-layer
d-vector with the same shape contract as any drowse steering direction, which
is what lets a ``jlens/<word>`` term ride the ordinary profile registry.

This module holds everything on a runtime path: the :class:`JacobianLens`
artifact (transport, token directions, select/union) and the readout math
every probe / live-workspace / token-replay surface calls per decode step.
The J-space decomposition lives in :mod:`drowse.lens.decompose`.  It imports
nothing beyond torch and the error taxonomy, so a per-step import costs a
``sys.modules`` lookup.

The **estimator** (transport, :mod:`drowse.lens.transport`, driven by
:mod:`drowse.lens.fit`) lives apart; importing this module never drags it in.

One unit runs through every readout surface: the per-layer softmax probability
``p_l(v)``.  ``readout_probabilities`` is the single calibration primitive; a
token's ``strength`` is ``mean_l p_l(v)`` and its depth center of mass is
weighted by the same ``p_l``, so per-layer cards, pinned probes, gate scalars,
and the layer aggregate are all apples-to-apples.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from drowse.errors import DrowseError, RequestError
from drowse.lens.key import LENS_SITES, LensKey

#: Positions before this index are excluded from the Jacobian average — early
#: positions act as attention sinks with atypical residual statistics.  Part of
#: the corpus-spec identity the sidecar records, so it lives with the artifact
#: rather than the estimator.
SKIP_FIRST_POSITIONS = 16
DEFAULT_SEQ_LEN = 128
#: Basis seeds per transport pass: one pass carries this many tangents (or
#: cotangents) through every block, so a document costs ``ceil(d_model /
#: dim_batch)`` passes.  The work is width-invariant; the width trades memory
#: for per-pass overhead (neuron-lens's transport runs 256).  Halves on OOM.
DEFAULT_DIM_BATCH = 256
#: Checkpoint cadence (prompts) for resumable fits.
DEFAULT_CHECKPOINT_EVERY = 25
#: A fit also checkpoints at the first document boundary this many seconds
#: after its last checkpoint (or its start), whatever the count: at 8B on MPS
#: a document takes about four minutes, so 25 documents would be 1.7 hours of
#: unsaved work.  A fixed rule, not a knob.
CHECKPOINT_SECONDS = 600.0

# The fit-configuration defaults above stay with the artifact: io/cli/session
# read them to stamp and validate sidecars without wanting the estimator, and
# they are plain ints, so keeping them here costs the split nothing.


class JacobianLensError(RuntimeError, DrowseError):
    """Raised when a Jacobian-lens fit or readout cannot proceed."""

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


class LensNotFittedError(JacobianLensError):
    """Raised when a lens artifact is required but absent for the model."""

    def user_message(self) -> tuple[int, str]:
        return (404, str(self) or self.__class__.__name__)


class JacobianLensCancelled(JacobianLensError):
    """Raised after a cooperative stop at a safe estimator boundary."""

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or "Jacobian-lens fit cancelled")


class MultiTokenWordError(RequestError):
    """Raised when a ``jlens/<word>`` atom has no single-token vocabulary id."""


class InvalidTokenIdError(RequestError):
    """Raised when a ``jlens/#<id>`` atom is malformed or outside the vocabulary."""


def resolve_token_atom(tokenizer: Any, atom: str, *, vocab_size: int) -> int:
    """Map the name after ``jlens/`` to the vocab id it addresses.

    ``#<int>`` is the exact-vocab-id form: the id is taken as written and
    bounds-checked against ``vocab_size`` (the unembedding's row count), never
    tokenized, so it reaches tokens with no clean single-word spelling.  Any
    other name is a word and resolves through :func:`resolve_word_token`.
    """
    if not atom.startswith("#"):
        return resolve_word_token(tokenizer, atom)
    digits = atom[1:]
    if not (digits.isascii() and digits.isdigit()):
        raise InvalidTokenIdError(
            f"token-id atom {atom!r} must be '#' followed by a non-negative integer"
        )
    token_id = int(digits)
    if token_id >= vocab_size:
        raise InvalidTokenIdError(
            f"token id {token_id} is outside this vocabulary (size {vocab_size})"
        )
    return token_id


def resolve_word_token(tokenizer: Any, word: str) -> int:
    """Map a word to the single vocab id its J-lens direction should use.

    Tries the leading-space piece first (``"▁word"``/``"Ġword"`` — the form
    the model actually emits in running text), then the bare piece. A
    candidate only counts when it round-trips: ``decode([id]).strip()`` must
    equal the word, so a normalizing tokenizer can't silently match a merge
    artifact. Raises :class:`MultiTokenWordError` (listing the pieces) when
    neither form is a single token.
    """
    pieces: list[str] = []
    for cand in (f" {word}", word):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1 and tokenizer.decode(ids).strip() == word:
            return int(ids[0])
        if not pieces and len(ids) > 1:
            pieces = [tokenizer.decode([i]) for i in ids]
    raise MultiTokenWordError(
        f"{word!r} is not a single token in this vocabulary"
        + (f" (pieces: {pieces})" if pieces else "")
        + " — the Jacobian lens can only address single-token words"
    )


class JacobianLens:
    """One lens: its kind and fp32 per-layer matrices at one or both sites.

    ``key`` names the kind (:class:`~drowse.lens.key.LensKey`; its site is
    ``out``, since the object can hold both).  ``jacobians`` are the ``out``
    site's matrices and ``mid`` the ``mid`` site's; either may be absent, and
    :meth:`matrices` / :meth:`layers` read a site explicitly.  Orientation
    follows the kind:

    - **Output kinds (J/R)** hold ``J_L = E[Σ ∂h_final/∂h_L]``, ``[d, d]``.
      :meth:`transport` maps a residual ``h`` to ``h @ J_Lᵀ`` (into the final
      basis), the front half of every vocabulary readout.
    - **Input kinds (K/S)** hold ``K_L = E[Σ ∂h_L/∂e]``, ``[d, d_emb]``, as
      rs-lenses' ``s-lens/{out,mid}.pt`` store them.  :meth:`pullback` maps a
      residual covector ``c`` to ``c @ K_L`` (into the embedding basis) and
      :meth:`footprint` an embedding-space vector ``ẽ`` to ``K_L ẽ`` (the
      residual direction a token leaves at layer ``L``).

    ``source_layers`` and the token-direction / atom-norm readers default to
    the ``out`` site, so the R family's readout code keeps reading ``out``.
    A read at a site the lens does not hold raises
    :class:`LensNotFittedError`; it never falls back to the other site.
    """

    def __init__(
        self,
        jacobians: Mapping[int, torch.Tensor] | None = None,
        *,
        n_prompts: int,
        d_model: int,
        key: LensKey = LensKey("J"),
        mid: Mapping[int, torch.Tensor] | None = None,
        d_emb: int | None = None,
    ) -> None:
        if key.site != "out":
            raise ValueError(
                f"a lens's key names its kind (site 'out'); pass {key.label} "
                "matrices as mid="
            )
        self.key = key
        self.n_prompts = int(n_prompts)
        self.d_model = int(d_model)
        self.d_emb = self.d_model if d_emb is None else int(d_emb)
        if key.family == "R" and self.d_emb != self.d_model:
            raise ValueError(f"an output-kind ({key.kind}) lens has no separate d_emb")
        self._sites: dict[str, dict[int, torch.Tensor]] = {}
        for site, given in (("out", jacobians), ("mid", mid)):
            if given is None:
                continue
            matrices = {int(l): m.to(torch.float32) for l, m in given.items()}
            for layer, matrix in matrices.items():
                if tuple(matrix.shape) != self.matrix_shape:
                    raise ValueError(
                        f"{key.kind}:{site} layer {layer} has shape "
                        f"{tuple(matrix.shape)}; a {key.kind} lens holds "
                        f"{list(self.matrix_shape)} matrices"
                    )
            self._sites[site] = matrices

    # -- identity and layout ------------------------------------------------

    @property
    def kind(self) -> str:
        """``J``, ``R``, ``K`` or ``S``."""
        return self.key.kind

    @property
    def sites(self) -> tuple[str, ...]:
        """The sites this lens holds, in ``LENS_SITES`` order."""
        return tuple(site for site in LENS_SITES if site in self._sites)

    def site_key(self, site: str) -> LensKey:
        """The per-site key (``R:mid``) bindings and logs use."""
        return LensKey(self.kind, site)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """``(d, d)`` for an output kind, ``(d, d_emb)`` for an input kind."""
        return (self.d_model, self.d_model if self.key.family == "R" else self.d_emb)

    def matrices(self, site: str = "out") -> dict[int, torch.Tensor]:
        """``{layer: matrix}`` at ``site`` (the live dict, not a copy)."""
        LensKey(self.kind, site)
        found = self._sites.get(site)
        if found is None:
            held = ", ".join(self.sites) or "none"
            raise LensNotFittedError(
                f"this {self.kind} lens has no {site!r} site (it holds: {held}); "
                f"fit or fetch a {self.kind} lens that carries {site}"
            )
        return found

    def layers(self, site: str = "out") -> list[int]:
        """The fitted layers at ``site``, sorted."""
        return sorted(self.matrices(site))

    def matrix(self, layer: int, site: str = "out") -> torch.Tensor:
        """One layer's matrix at ``site``."""
        matrices = self.matrices(site)
        if layer not in matrices:
            fitted = sorted(matrices)
            raise LensNotFittedError(
                f"layer {layer} not in fitted {self.kind}:{site} lens layers "
                f"{fitted[:3]}..{fitted[-3:]}"
            )
        return matrices[layer]

    @property
    def jacobians(self) -> dict[int, torch.Tensor]:
        """The ``out`` site's matrices."""
        return self.matrices("out")

    @property
    def source_layers(self) -> list[int]:
        """The ``out`` site's fitted layers."""
        return self.layers("out")

    def __repr__(self) -> str:
        spans = []
        for site in self.sites:
            layers = self.layers(site)
            span = f"[{layers[0]}..{layers[-1]}]" if layers else "[]"
            spans.append(f"{site}={span} ({len(layers)} layers)")
        return (
            f"JacobianLens({self.kind}, d_model={self.d_model}, "
            f"n_prompts={self.n_prompts}, {', '.join(spans) or 'no sites'})"
        )

    # -- reads ----------------------------------------------------------------

    def _require_family(self, family: str, read: str) -> None:
        if self.key.family != family:
            other = "pullback or footprint" if family == "R" else "transport"
            raise JacobianLensError(
                f"{read} reads an {family}-family lens; this is a {self.kind} lens "
                f"(read it by {other})"
            )

    def transport(self, hidden: torch.Tensor, layer: int, site: str = "out") -> torch.Tensor:
        """Output kinds: map a residual ``[..., d]`` into the final basis, ``h @ J_Lᵀ``."""
        self._require_family("R", "transport")
        J = self.matrix(layer, site).to(hidden.device)
        return hidden.to(torch.float32) @ J.T

    def pullback(self, covector: torch.Tensor, layer: int, site: str = "out") -> torch.Tensor:
        """Input kinds: a residual covector ``[..., d]`` into the embedding basis, ``c @ K_L``."""
        self._require_family("S", "pullback")
        K = self.matrix(layer, site).to(covector.device)
        return covector.to(torch.float32) @ K

    def footprint(self, embedding: torch.Tensor, layer: int, site: str = "out") -> torch.Tensor:
        """Input kinds: an embedding-space ``[..., d_emb]`` into the residual, ``K_L ẽ``."""
        self._require_family("S", "footprint")
        K = self.matrix(layer, site).to(embedding.device)
        return embedding.to(torch.float32) @ K.T

    def token_direction(
        self,
        token_id: int,
        unembed: torch.Tensor,
        *,
        layers: Sequence[int] | None = None,
        output_projection: torch.Tensor | None = None,
        site: str = "out",
    ) -> dict[int, torch.Tensor]:
        """Output kinds: per-layer direction for one vocab id, ``W_U[v] @ J_l``.

        Returns fp32 CPU tensors in the ``dict[int, Tensor]`` shape every
        drowse profile consumer (``fold_directions_to_subspace``,
        ``Profile``) expects. With an output projection P (OPT), use
        ``W_U[v] @ P @ J_l`` without materializing the full vocabulary map.
        """
        self._require_family("R", "a token direction")
        w = unembed[token_id].detach().to(torch.float32).cpu()
        if output_projection is not None:
            w = w @ output_projection.detach().to(device="cpu", dtype=torch.float32)
        matrices = self.matrices(site)
        requested = sorted(matrices) if layers is None else [int(l) for l in layers]
        missing = [l for l in requested if l not in matrices]
        if missing:
            fitted = sorted(matrices)
            raise LensNotFittedError(
                f"layers {missing} not in fitted {self.kind}:{site} lens layers "
                f"{fitted[:3]}..{fitted[-3:]}"
            )
        return {l: w @ matrices[l] for l in requested}

    def select_layers(self, layers: Mapping[str, Iterable[int]]) -> "JacobianLens":
        """Return a view-like lens holding exactly ``layers``, per site.

        ``layers`` maps each kept site to its layers (``{"out": [0, 1],
        "mid": [0, 1, 2]}``); sites it does not name are dropped.  Sites carry
        different layer sets by design (the R family's ``out`` stops below
        the last layer, its ``mid`` covers it), so each site is asked only
        for its own.  A site or layer the lens lacks raises
        :class:`LensNotFittedError`.  The tensors are shared with ``self``;
        callers that persist the result materialize their own fp32 shards
        through ``save_lens``.
        """
        kept: dict[str, dict[int, torch.Tensor]] = {}
        for site, wanted in layers.items():
            matrices = self.matrices(site)
            requested = sorted({int(l) for l in wanted})
            missing = [l for l in requested if l not in matrices]
            if missing:
                fitted = sorted(matrices)
                raise LensNotFittedError(
                    f"layers {missing} not in fitted {self.kind}:{site} lens layers "
                    f"{fitted[:3]}..{fitted[-3:]}"
                )
            kept[site] = {layer: matrices[layer] for layer in requested}
        if not any(kept.values()):
            raise ValueError("select_layers() needs at least one layer")
        return self._with_sites(kept)

    def _with_sites(self, sites: Mapping[str, Mapping[int, torch.Tensor]]) -> "JacobianLens":
        return JacobianLens(
            sites.get("out"),
            mid=sites.get("mid"),
            n_prompts=self.n_prompts,
            d_model=self.d_model,
            d_emb=self.d_emb,
            key=self.key,
        )

    @classmethod
    def union_layers(cls, lenses: Sequence["JacobianLens"]) -> "JacobianLens":
        """Combine same-corpus lenses that cover different layers, site by site.

        This is not a prompt-weighted average; every input must describe the
        same prompt set, ``d_model``, ``d_emb`` and kind.  Later inputs
        replace duplicate layers, which lets a missing-layer top-up overwrite
        a stale partial layer cleanly.
        """
        if not lenses:
            raise ValueError("union_layers() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.n_prompts != first.n_prompts
                or other.d_model != first.d_model
                or other.d_emb != first.d_emb
                or other.key != first.key
            ):
                raise ValueError("lenses disagree on n_prompts / d_model / d_emb / key")
        union: dict[str, dict[int, torch.Tensor]] = {}
        for lens in lenses:
            for site in lens.sites:
                union.setdefault(site, {}).update(lens.matrices(site))
        return first._with_sites(union)


@contextmanager
def _scoped_tf32(device: torch.device) -> Iterator[None]:
    """TF32 matmuls for one CUDA transport, restoring the caller's precision.

    The lens transport is a ``[L, d, d] @ [L, d]`` bmm whose output feeds a
    softmax readout, so TF32 costs nothing visible and buys a large CUDA
    speedup.  The precision setting is process-global, so it is scoped to the
    call: a model loaded or fitted beside the readout keeps full fp32.
    """
    if device.type != "cuda":
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def transport_rows(
    jacobian_rows: torch.Tensor, hidden_rows: torch.Tensor,
) -> torch.Tensor:
    """Batched transport ``J_l h_l`` for ``[R, d, d]`` matrices, ``[R, d]`` rows.

    The one front half of every readout surface (live step, gate, probe,
    replay, ``lens top``): rows land fp32 on the matrices' device.
    """
    hidden = hidden_rows.to(device=jacobian_rows.device, dtype=torch.float32)
    with _scoped_tf32(jacobian_rows.device):
        return torch.bmm(jacobian_rows, hidden.unsqueeze(-1)).squeeze(-1)


def topk_probabilities(
    logits: torch.Tensor, k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k readout probabilities without a full softmax tensor.

    ``topk(softmax(x))`` has the same indices as ``topk(x)``, so only the
    selected columns need normalizing — saving one vocab-sized allocation per
    readout row, which matters for large vocabularies and multi-layer J-lens
    sweeps.  The unit is the per-layer probability ``p_l(v)``, the same one
    ``readout_probabilities`` produces and every other lens surface reports.
    """
    logits_f = logits.float()
    vals, idxs = logits_f.topk(k, dim=-1)
    vals = (vals - logits_f.logsumexp(dim=-1, keepdim=True)).exp()
    return vals, idxs


def readout_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Calibrate per-layer lens logits into the shared probability unit.

    Kept as one explicit primitive so a live decode step can normalize its
    full ``[layers, vocab]`` matrix once, then share the result between pinned
    probes, per-layer cards, and the aggregate readout.
    """
    return logits.float().softmax(dim=-1)


def pack_readout_rows_to_host(*rows: torch.Tensor) -> torch.Tensor:
    """Pack selected readout rows for one accelerator-to-host transfer.

    The packed payload stays fp32 on the accelerator because MPS does not
    support float64.  fp32 still represents every practical vocabulary id
    exactly (up to 2**24), so integer token-id rows can share the transfer with
    probability/statistic rows without changing the public values.
    """
    return torch.cat([row.float() for row in rows], dim=0).detach().cpu()


def _check_rows(values: torch.Tensor, depths: "Sequence[float]", what: str) -> None:
    """A readout matrix is ``[layers, columns]`` with one depth per layer."""
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            f"{what} expects a [layers, columns] matrix, got shape "
            f"{tuple(values.shape)}"
        )
    if len(depths) != values.shape[0]:
        raise ValueError(
            f"{what}: {values.shape[0]} rows but {len(depths)} depths"
        )


def _depth_moments(
    p: torch.Tensor, d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Probability-weighted depth center of mass and spread per column.

    ``p`` is ``[L, K]`` per-layer probability, ``d`` the ``[L, 1]`` depths.
    """
    mass = p.sum(dim=0).clamp_min(1e-12)                        # [K]
    com = (p * d).sum(dim=0) / mass                             # [K]
    var = (p * (d - com.unsqueeze(0)) ** 2).sum(dim=0) / mass
    return com, var.clamp_min(0.0).sqrt()


def aggregate_readout_from_probabilities(
    probabilities: torch.Tensor,
    depths: "Sequence[float]",
    *,
    top_k: int = 8,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[int, float, float, float]]:
    """Aggregate already-calibrated ``[layers, vocab]`` probabilities.

    Strength must be computed over the whole vocabulary before selection, but
    depth CoM/spread are needed only for the selected tokens.  Gathering those
    columns first preserves the exact statistic while avoiding full-vocabulary
    depth tensors.
    """
    idxs, stats = aggregate_readout_tensors_from_probabilities(
        probabilities,
        depths,
        top_k=top_k,
        depth_tensor=depth_tensor,
    )
    host = pack_readout_rows_to_host(stats, idxs.reshape(1, -1))
    return [
        (
            int(host[3, j]),
            float(host[0, j]),
            float(host[1, j]),
            float(host[2, j]),
        )
        for j in range(int(idxs.numel()))
    ]


def aggregate_readout_tensors_from_probabilities(
    probabilities: torch.Tensor,
    depths: "Sequence[float]",
    *,
    top_k: int = 8,
    depth_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-resident aggregate selection and statistics.

    Returns ``(token_ids[K], stats[3,K])`` where the statistic rows are
    strength, depth center-of-mass, and depth spread. Keeping this tensor form
    separate from the public list conversion lets the live readout combine its
    per-layer and aggregate payloads into one GPU-to-host synchronization per
    token.
    """
    _check_rows(probabilities, depths, "aggregate readout")
    strength = probabilities.mean(dim=0)                       # [V]
    k = min(max(int(top_k), 0), int(strength.shape[-1]))
    vals, idxs = strength.topk(k)
    p = probabilities.index_select(-1, idxs)                   # [L, K]
    d = _depth_column(
        depths,
        device=probabilities.device,
        rows=int(probabilities.shape[0]),
        depth_tensor=depth_tensor,
    )                                                           # [L, 1]
    com, spread = _depth_moments(p, d)
    return idxs, torch.stack([vals, com, spread])


def aggregate_readout(
    logits: torch.Tensor,
    depths: "Sequence[float]",
    *,
    top_k: int = 8,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[int, float, float, float]]:
    """Layer-aggregate a per-layer lens readout into one ranked token list.

    ``logits`` is ``[L, vocab]`` — one full-vocabulary lens readout row per
    layer (``LogitReadout(model)(transport_rows(J, h))``); ``depths`` the matching
    normalized layer depths in ``[0, 1]`` (``layer / (n_layers − 1)``).
    Returns ``[(vocab_id, strength, com, spread), ...]`` sorted by
    descending strength.

    Raw lens logits are uncalibrated across layers, so each layer is first
    put through its own softmax.  From the per-layer probabilities
    ``p_l(v)``, two statistics per token:

    - ``strength = mean_l p_l(v)`` — the mean probability over the layer
      band, in ``[0, 1]``.  Uniform layer weights: the softmax already
      lets a confident layer dominate the ranking, so extra confidence
      weighting would double-count.
    - ``com``/``spread`` — the depth center of mass (+ std) weighted by
      the same per-layer probability ``p_l(v)``. The selected-layer
      readout is sharp, not diffuse (median per-layer max ≈ 0.8 on
      gemma-3-4b) — what changes over depth is *which* token leads, so a
      token's probability profile over depth IS its depth signal.
      Probability mass also discounts a genuinely diffuse (noise) layer
      automatically; the former within-layer salience ``p_l/max_v' p_l``
      handed such a layer's relative-top token a full vote regardless of
      absolute mass (in band the two weightings agree to ≲0.01 — one
      channel, ``p_l``, now backs every readout statistic).

    Top-k selection runs on the aggregated full-vocab strengths — a
    per-layer top-k union would miss a token that ranks mid-pack at every
    layer but top at none.
    """
    _check_rows(logits, depths, "aggregate readout")
    return aggregate_readout_from_probabilities(
        readout_probabilities(logits),
        depths,
        top_k=top_k,
        depth_tensor=depth_tensor,
    )


def token_readout_stats_from_probabilities(
    probabilities: torch.Tensor,
    depths: "Sequence[float]",
    token_ids: "Sequence[int]",
    *,
    token_ids_tensor: torch.Tensor | None = None,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[float, float, float, list[float]]]:
    """Per-token statistics from already-calibrated readout probabilities."""
    _check_rows(probabilities, depths, "token readout stats")
    if not token_ids:
        return []
    ids = _token_id_tensor(
        token_ids,
        device=probabilities.device,
        token_ids_tensor=token_ids_tensor,
    )
    p = probabilities.index_select(-1, ids)                    # [L, K]
    return _token_readout_stats_from_probability_columns(
        p, depths, depth_tensor=depth_tensor,
    )


def _token_probability_columns_from_logits(
    logits: torch.Tensor,
    token_ids: "Sequence[int]",
    *,
    token_ids_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact softmax probabilities for selected token ids only.

    The denominator remains the full-vocabulary logsumexp for each layer; only
    the returned probability columns are narrowed. This preserves gate/display
    thresholds while avoiding a full ``[layers, vocab]`` probability matrix for
    fixed-token probes.
    """
    ids = _token_id_tensor(
        token_ids,
        device=logits.device,
        token_ids_tensor=token_ids_tensor,
    )
    logits_f = logits.float()
    selected = logits_f.index_select(-1, ids)                  # [L, K]
    log_z = logits_f.logsumexp(dim=-1, keepdim=True)           # [L, 1]
    return (selected - log_z).exp()


def _depth_column(
    depths: "Sequence[float]",
    *,
    device: torch.device,
    rows: int,
    depth_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return depths as a device ``[L, 1]`` tensor, reusing a caller cache."""
    if depth_tensor is None:
        d = torch.tensor(
            [float(x) for x in depths],
            dtype=torch.float32,
            device=device,
        )
    else:
        d = depth_tensor.to(device=device, dtype=torch.float32)
        if d.ndim > 2:
            raise ValueError(
                f"depth_tensor must be [layers] or [layers, 1], got {tuple(d.shape)}"
            )
        d = d.reshape(-1)
    if int(d.numel()) != rows:
        raise ValueError(
            f"depth tensor has {int(d.numel())} rows, expected {rows}"
        )
    return d.reshape(rows, 1)


def _token_id_tensor(
    token_ids: "Sequence[int]",
    *,
    device: torch.device,
    token_ids_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return token ids as a device long tensor, reusing a caller cache."""
    if token_ids_tensor is None:
        return torch.tensor(
            [int(v) for v in token_ids],
            dtype=torch.long,
            device=device,
        )
    ids = token_ids_tensor.to(device=device, dtype=torch.long).reshape(-1)
    if int(ids.numel()) != len(token_ids):
        raise ValueError(
            f"token id tensor has {int(ids.numel())} ids, expected {len(token_ids)}"
        )
    return ids


def _token_rows_from_probability_columns(
    p: torch.Tensor,
    depths: "Sequence[float]",
    *,
    depth_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Device ``[3 + L, K]``: strength, depth center, spread, then ``p_l``,
    for selected token probability columns ``p`` ``[L, K]``."""
    _check_rows(p, depths, "token readout stats")
    strength = p.mean(dim=0)                                    # [K]
    d = _depth_column(
        depths,
        device=p.device,
        rows=int(p.shape[0]),
        depth_tensor=depth_tensor,
    )                                                           # [L, 1]
    com, spread = _depth_moments(p, d)
    return torch.cat([torch.stack([strength, com, spread]), p], dim=0)


def token_readout_rows(
    rows: torch.Tensor,
    depths: "Sequence[float]",
    token_ids: "Sequence[int]",
    *,
    from_logits: bool,
    token_ids_tensor: torch.Tensor | None = None,
    depth_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`token_readout_stats` (``from_logits``) or
    :func:`token_readout_stats_from_probabilities` before the host transfer:
    the device ``[3 + L, K]`` rows (strength, depth center, spread, then
    ``p_l``), so a caller can fold them into its one transfer and unpack
    the host copy with :func:`unpack_token_readout_rows`."""
    _check_rows(rows, depths, "token readout stats")
    if from_logits:
        p = _token_probability_columns_from_logits(
            rows, token_ids, token_ids_tensor=token_ids_tensor,
        )
    else:
        p = rows.index_select(-1, _token_id_tensor(
            token_ids, device=rows.device, token_ids_tensor=token_ids_tensor,
        ))
    return _token_rows_from_probability_columns(p, depths, depth_tensor=depth_tensor)


def unpack_token_readout_rows(
    host: torch.Tensor,
) -> list[tuple[float, float, float, list[float]]]:
    """``(strength, com, spread, per_layer)`` per column of a host copy of
    :func:`token_readout_rows`."""
    n_layers = int(host.shape[0]) - 3
    out: list[tuple[float, float, float, list[float]]] = []
    for j in range(int(host.shape[1])):
        per_layer = [float(host[3 + l, j]) for l in range(n_layers)]
        out.append(
            (
                float(host[0, j]), float(host[1, j]), float(host[2, j]),
                per_layer,
            )
        )
    return out


def _token_readout_stats_from_probability_columns(
    p: torch.Tensor,
    depths: "Sequence[float]",
    *,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[float, float, float, list[float]]]:
    """Shared stats body for selected token probability columns ``[L, K]``."""
    _check_rows(p, depths, "token readout stats")
    if p.shape[1] == 0:
        return []
    # one batched host transfer: 3 aggregate rows + the per-layer block
    host = _token_rows_from_probability_columns(
        p, depths, depth_tensor=depth_tensor,
    ).cpu()                                                     # [3+L, K]
    return unpack_token_readout_rows(host)


def token_readout_stats(
    logits: torch.Tensor,
    depths: "Sequence[float]",
    token_ids: "Sequence[int]",
    *,
    token_ids_tensor: torch.Tensor | None = None,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[float, float, float, list[float]]]:
    """Per-token readout statistics for pinned vocabulary ids.

    The single-token restriction of :func:`aggregate_readout`: the same
    per-layer softmax calibration, read at the requested ``token_ids``
    instead of selected top-k.  For each id:

    - ``strength = mean_l p_l(v)`` — mean fitted-layer probability, the aggregate
      readout's ranking stat and the ONE probe/gate/display channel
      (objective and apples-to-apples across tokens and layers, unlike a
      within-layer max normalization).
    - ``com`` / ``spread`` — depth center of mass (+ std), weighted by the
      same per-layer probability ``p_l(v)`` exactly like
      :func:`aggregate_readout` — the one channel backs every statistic
      (the band readout is sharp, so a token's probability profile over
      depth is its depth signal; a diffuse noise layer's vote is
      discounted by its own lack of mass).
    - ``per_layer`` — ``[p_l, ...]`` aligned with the logit rows.

    Returns one ``(strength, com, spread, per_layer)`` tuple per requested
    id, with a single batched host transfer.
    """
    _check_rows(logits, depths, "token readout stats")
    if not token_ids:
        return []
    return _token_readout_stats_from_probability_columns(
        _token_probability_columns_from_logits(
            logits, token_ids, token_ids_tensor=token_ids_tensor,
        ),
        depths,
        depth_tensor=depth_tensor,
    )
