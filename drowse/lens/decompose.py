"""Sparse decomposition of a direction against a lens dictionary.

A *dictionary* is a set of atoms in residual space that is never
materialized: a vocabulary-sized ``[V, d]`` matrix would be gigabytes on a
real model, so a dictionary exposes only the three operations the pursuit
needs — correlate a residual with every atom, form one atom, and the atom
norms.  The pursuit is the paper's gradient-pursuit decomposition: greedily
pick the atom that best correlates with the residual, re-solve the
coefficients over the selected atoms, repeat.

Every inner product is Mahalanobis: a dictionary is built over one layer's
:class:`~drowse.geometry.whitener.LayerWhitener` and lives in its whitened
coordinates, and :func:`sparse_decompose` maps the target there too
(``Σ^{-1/2}``: the target is a vector), so atom selection, the NNLS fit and
the share are all whitened.  Atoms enter by their type.  Neuron writes are
vectors and whiten by ``Σ^{-1/2}``.  Token atoms (``W_U J_l`` rows,
footprints) are *reads* and whiten by ``Σ^{+1/2}``
(:meth:`~drowse.geometry.whitener.LayerWhitener.whiten_dual`), so a token's
score against a target is the plain pairing ``u · t`` (its lens reading of
the target), and decomposing a ``jlens/``/``slens/`` atom (the raised vector
``Σu``) recovers its own token exactly.  A whitener that does not cover the
layer raises :class:`~drowse.errors.WhitenerError`; there is no Euclidean
path.

Three dictionaries exist: :class:`TokensOut` (``tokens-out``, ``W_U J_l``:
the tokens a direction is disposed to *say*), :class:`TokensIn`
(``tokens-in``, the S-family footprints ``K_l ẽ_v`` over the token mask: the
tokens whose *having been read* it resembles) and :class:`NeuronWrites`
(``neurons``, block ``l``'s neuron writes ``s · γ_out ⊙ W_down[:, n]``, Gemma's
post-feedforward gain included: the neurons whose writes compose it).  The
neuron dictionary is *signed*, because a neuron has two poles (``+w`` and
``−w``): it selects by ``|score|`` and solves its coefficients unconstrained,
the sign naming the pole; the token dictionaries keep the NNLS step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import torch

from drowse.errors import WhitenerError

if TYPE_CHECKING:
    from drowse.geometry.whitener import LayerWhitener
    from drowse.lens.footprint import TokenEmbeddings

#: The dictionaries ``decompose`` accepts, by wire name.
DICTIONARIES = ("tokens-out", "tokens-in", "neurons")


class LensDictionary(Protocol):
    """Atoms ``D[i] ∈ R^d`` addressed by index, never materialized, in one
    layer's whitened coordinates (:meth:`coords` maps a residual there)."""

    @property
    def kind(self) -> str:
        """The dictionary's name on the wire (``tokens-out``, …)."""
        ...

    @property
    def nonneg(self) -> bool:
        """Whether coefficients are constrained nonnegative (tokens) or signed."""
        ...

    @property
    def device(self) -> torch.device: ...

    def coords(self, vector: torch.Tensor) -> torch.Tensor:
        """A residual-space vector in the dictionary's whitened coordinates
        (fp32 ``[d]`` on :attr:`device`)."""
        ...

    def scores(self, residual: torch.Tensor) -> torch.Tensor:
        """``D @ residual`` over every atom (fp32 ``[V]``), ``residual`` in
        whitened coordinates."""
        ...

    def atom(self, index: int) -> torch.Tensor:
        """One atom ``D[index]`` (fp32 ``[d]``)."""
        ...

    def norms(self) -> torch.Tensor:
        """``‖D[i]‖`` for every atom (fp32 ``[V]``)."""
        ...


def _require_cover(whitener: "LayerWhitener", layer: int) -> None:
    if not whitener.covers_all([layer]):
        raise WhitenerError(
            f"decomposition at layer {layer} needs a Mahalanobis whitener "
            f"covering it (regenerate the neutral activation cache)"
        )


def token_atom_norms(
    jacobian: torch.Tensor, unembed: torch.Tensor, *, chunk: int = 8192,
) -> torch.Tensor:
    """Per-atom norms ``‖W_U[v] @ J‖`` for the whole vocabulary, computed in
    chunks so the ``[vocab, d]`` dictionary never exists in full."""
    norms = torch.empty(unembed.shape[0], device=unembed.device)
    for start in range(0, unembed.shape[0], chunk):
        block = unembed[start : start + chunk].float() @ jacobian
        norms[start : start + chunk] = block.norm(dim=-1)
    return norms


@dataclass
class TokensOut:
    """``W_U J_l``: the tokens a layer-``l`` residual is disposed to say.

    Held whitened as reads: ``jacobian`` becomes ``J_l Σ_l^{1/2}`` (the
    dual-whitened rows of ``whitener``), so atom ``v`` is ``Σ^{1/2} Jᵀ W_U[v]``.
    ``atom_norms`` may
    carry cached whitened norms (the runtime caches them per lens, layer and
    whitener); otherwise they are computed in chunks on first use.
    """

    jacobian: torch.Tensor
    unembed: torch.Tensor
    whitener: "LayerWhitener"
    layer: int
    atom_norms: torch.Tensor | None = None
    kind: str = field(default="tokens-out", init=False)
    nonneg: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        _require_cover(self.whitener, self.layer)
        self.jacobian = self.whitener.whiten_dual(
            self.layer, self.jacobian.to(dtype=torch.float32),
        ).to(device=self.unembed.device)

    @property
    def device(self) -> torch.device:
        return self.unembed.device

    def coords(self, vector: torch.Tensor) -> torch.Tensor:
        return self.whitener.whiten(self.layer, vector.detach().float()).to(self.device)

    def scores(self, residual: torch.Tensor) -> torch.Tensor:
        return (
            self.unembed @ (self.jacobian @ residual).to(self.unembed.dtype)
        ).float()

    def atom(self, index: int) -> torch.Tensor:
        return self.unembed[index].float() @ self.jacobian

    def norms(self) -> torch.Tensor:
        if self.atom_norms is None:
            self.atom_norms = token_atom_norms(self.jacobian, self.unembed)
        return self.atom_norms.to(device=self.device, dtype=torch.float32)


def footprint_atom_norms(
    matrix: torch.Tensor, embeddings: "TokenEmbeddings", *, chunk: int = 8192,
) -> torch.Tensor:
    """Per-atom norms ``‖K ẽ_v‖`` over the vocabulary, in chunks so the
    ``[vocab, d]`` dictionary never exists in full.  Masked tokens read 1.0
    (they score ``-inf`` and are never selected)."""
    norms = torch.ones(embeddings.vocab, dtype=torch.float32, device=embeddings.device)
    K = matrix.to(device=embeddings.device, dtype=torch.float32)
    for start in range(0, embeddings.vocab, chunk):
        ids = torch.arange(start, min(start + chunk, embeddings.vocab), device=embeddings.device)
        keep = embeddings.keep[ids]
        if not bool(keep.any()):
            continue
        block = embeddings.centred(ids[keep]) @ K.T
        norms[ids[keep]] = block.norm(dim=-1)
    return norms


@dataclass
class TokensIn:
    """``K_l ẽ_v``: the footprint a token leaves at layer ``l`` (S family, ``out``).

    Held whitened as reads: ``matrix`` becomes ``Σ_l^{1/2} K_l`` (the
    footprint is a covector), so atom ``v`` is ``Σ^{1/2} K_l ẽ_v``.  Atoms run over the token
    mask of ``embeddings``; masked tokens score ``-inf`` and are never
    selected.  ``atom_norms`` may carry cached whitened norms; otherwise they
    are computed in chunks on first use.
    """

    matrix: torch.Tensor
    embeddings: "TokenEmbeddings"
    whitener: "LayerWhitener"
    layer: int
    atom_norms: torch.Tensor | None = None
    kind: str = field(default="tokens-in", init=False)
    nonneg: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        _require_cover(self.whitener, self.layer)
        # Σ^{1/2} K = (Kᵀ Σ^{1/2})ᵀ: dual-whiten the columns of K as rows of Kᵀ.
        self.matrix = self.whitener.whiten_dual(
            self.layer, self.matrix.to(dtype=torch.float32).transpose(0, 1),
        ).transpose(0, 1).to(device=self.embeddings.device).contiguous()

    @property
    def device(self) -> torch.device:
        return self.embeddings.device

    def coords(self, vector: torch.Tensor) -> torch.Tensor:
        return self.whitener.whiten(self.layer, vector.detach().float()).to(self.device)

    def scores(self, residual: torch.Tensor) -> torch.Tensor:
        # (K ẽ_v) · r = ẽ_v · (Kᵀ r): one pullback, one pass over the vocabulary.
        u = residual.to(device=self.device, dtype=torch.float32) @ self.matrix
        return self.embeddings.scores(u.reshape(1, -1))[0]

    def atom(self, index: int) -> torch.Tensor:
        return self.embeddings.centred([index])[0] @ self.matrix.T

    def norms(self) -> torch.Tensor:
        if self.atom_norms is None:
            self.atom_norms = footprint_atom_norms(self.matrix, self.embeddings)
        return self.atom_norms.to(device=self.device, dtype=torch.float32)


@dataclass
class NeuronWrites:
    """Block ``l``'s neuron writes, signed: the neurons a direction is made of.

    ``writes`` is ``[d, d_ff]`` (column ``n`` the ``+``-pole write of neuron
    ``n``, :func:`~drowse.model.units.neuron_writes`), held whitened as rows
    ``W w_n`` (``W = Σ_l^{-1/2}``).  Coefficients are signed: a positive one
    is the ``+`` pole, a negative one the ``−`` pole.
    """

    writes: torch.Tensor
    whitener: "LayerWhitener"
    layer: int
    kind: str = field(default="neurons", init=False)
    nonneg: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _require_cover(self.whitener, self.layer)
        self.rows = self.whitener.whiten(
            self.layer, self.writes.to(dtype=torch.float32).T,
        ).contiguous()

    @property
    def device(self) -> torch.device:
        return self.rows.device

    def coords(self, vector: torch.Tensor) -> torch.Tensor:
        return self.whitener.whiten(self.layer, vector.detach().float()).to(self.device)

    def scores(self, residual: torch.Tensor) -> torch.Tensor:
        return self.rows @ residual.to(device=self.device, dtype=torch.float32)

    def atom(self, index: int) -> torch.Tensor:
        return self.rows[int(index)]

    def norms(self) -> torch.Tensor:
        return self.rows.norm(dim=-1)


@dataclass(frozen=True)
class SparseDecomposition:
    """One layer's sparse split of a direction against a dictionary.

    ``share`` is the fraction of the direction's variance the selected atoms
    carry; ``atoms`` is ``[(index, coeff), ...]`` by descending magnitude —
    vocabulary ids for a token dictionary.
    """

    layer: int
    share: float
    atoms: list[tuple[int, float]]
    dictionary: str

    def __repr__(self) -> str:
        return (
            f"SparseDecomposition({self.dictionary}, layer={self.layer}, "
            f"share={self.share:.3f}, k={len(self.atoms)})"
        )


def sparse_decompose(
    target: torch.Tensor,
    dictionary: LensDictionary,
    *,
    layer: int,
    k: int = 16,
    nnls_iters: int = 200,
) -> SparseDecomposition:
    """Greedy sparse pursuit of ``target`` against ``dictionary``, in the
    dictionary's whitened coordinates (``target`` maps there through
    :meth:`LensDictionary.coords`), so every inner product, the fit and the
    share are Mahalanobis.

    Atom scores are normalized by the atom norms (raw inner products would
    bias selection toward large-norm atoms), and only the ≤k selected atoms
    are formed.  After each selection the coefficients re-solve over the
    selected atoms — nonnegative least squares for a nonnegative dictionary
    (projected gradient with the exact Lipschitz step; the problem is tiny),
    ordinary least squares for a signed one.  The tiny ``k × k`` system is
    solved in fp64: read atoms (token rows) share their high-variance
    components, so in the dual metric they are nearly collinear and an fp32
    normal-equation solve mis-fits.  Selection stops early when no atom
    correlates with the residual in an admissible direction, or when the
    residual is already negligible (an exact atom target).
    """
    device = dictionary.device
    t = dictionary.coords(target).to(device=device, dtype=torch.float32)
    t_norm_sq = float(t.pow(2).sum())
    if t_norm_sq == 0.0:
        return SparseDecomposition(layer, 0.0, [], dictionary.kind)
    norms = dictionary.norms().clamp(min=1e-12)

    selected: list[int] = []
    rows: list[torch.Tensor] = []
    coeffs = torch.zeros(0, device=device)
    residual = t.clone()
    t64 = t.double()
    for _ in range(k):
        if float(residual.pow(2).sum()) <= 1e-12 * t_norm_sq:
            break
        scores = dictionary.scores(residual) / norms
        if not dictionary.nonneg:
            scores = scores.abs()
        if selected:
            scores[torch.tensor(selected, device=device)] = -torch.inf
        best = int(scores.argmax())
        if float(scores[best]) <= 0.0:
            break
        selected.append(best)
        rows.append(dictionary.atom(best))
        A = torch.stack(rows)  # [s, d]
        # fp64 on the host (MPS has no float64): the system is ≤ k×k.
        A64 = A.detach().to("cpu", torch.float64)
        gram = A64 @ A64.T
        b = A64 @ t64.cpu()
        if dictionary.nonneg:
            c = torch.cat([coeffs.cpu().double(), gram.new_zeros(1)])
            solved = _try_unconstrained_nonnegative(gram, b)
            if solved is None:
                lipschitz = float(torch.linalg.eigvalsh(gram)[-1].clamp(min=1e-12))
                for _ in range(nnls_iters):
                    c = torch.clamp(c - (gram @ c - b) / lipschitz, min=0.0)
            else:
                c = solved
        else:
            c = _solve_tiny(gram, b)
        coeffs = c.to(device=device, dtype=torch.float32)
        residual = t - A.T @ coeffs

    share = max(0.0, 1.0 - float(residual.pow(2).sum()) / t_norm_sq)
    pairs = sorted(
        (
            (idx, float(cf)) for idx, cf in zip(selected, coeffs)
            if float(cf) != 0.0 and (float(cf) > 0.0 or not dictionary.nonneg)
        ),
        key=lambda p: -abs(p[1]),
    )
    return SparseDecomposition(layer, share, pairs, dictionary.kind)


def _solve_tiny(gram: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Ridge-stabilized normal-equation solve on the CPU in fp64 (the system
    is ≤ k×k); the ridge is relative to the Gram's scale."""
    g_cpu = gram.detach().to("cpu", torch.float64)
    eye = torch.eye(g_cpu.shape[0], dtype=torch.float64)
    ridge = 1e-12 * float(g_cpu.diagonal().abs().max().clamp(min=1e-30))
    return torch.linalg.solve(g_cpu + ridge * eye, b.detach().to("cpu", torch.float64))


def _try_unconstrained_nonnegative(
    gram: torch.Tensor,
    b: torch.Tensor,
    *,
    tol: float = 1e-7,
) -> torch.Tensor | None:
    """Exact tiny least-squares solve when its coefficients are already >= 0.

    The greedy step solves ``min ||A^T c - t||²`` over the selected atoms.
    If the unconstrained normal-equation solution is nonnegative, it is also
    the NNLS optimum, so the projected-gradient loop is pure overhead.  If
    any coefficient is negative or the tiny system is singular, the caller
    keeps the projected-gradient fallback.
    """
    try:
        sol = _solve_tiny(gram, b)
    except RuntimeError:
        return None
    if not bool(torch.isfinite(sol).all()):
        return None
    if bool((sol < -tol * float(sol.abs().max().clamp(min=1e-30))).any()):
        return None
    return sol.clamp(min=0.0)
