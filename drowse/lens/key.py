"""Lens identity: which Jacobian a lens holds, and at which residual site.

A lens is one estimator run's per-layer matrices.  Its **kind** is the
direction of the Jacobian crossed with the backward rules it was taken under:

====  =========  ========  =====================================================
kind  side       rules     matrix
====  =========  ========  =====================================================
J     output     standard  ``J_L = E[Σ_{t'≥t} ∂h_final(t')/∂h_L(t)]`` (reverse)
R     output     RelP      J under the LN / identity / half rules
K     input      standard  ``K_L = E[Σ_{s≤t} ∂h_L(t)/∂e(s)]`` (forward)
S     input      RelP      K under the same rules
====  =========  ========  =====================================================

Its **site** is the residual read: ``out`` (a block's output) or ``mid`` (the
residual after the attention add — the MLP's input).  Output-side lenses read
an activation as what it is disposed to make the model *say*; input-side
lenses read it as what it has *read*.  One estimator (transport, in
:mod:`drowse.lens.transport`) fits every kind at both sites; every store,
binding and selection path is keyed on :class:`LensKey`.

Torch-free: the CLI, server, and source registry parse keys and local names
without importing the readout math.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re

LENS_KINDS = ("J", "R", "K", "S")
LENS_SITES = ("out", "mid")
#: Selection slots: one active lens per side, so an output-side (J/R) and an
#: input-side (K/S) lens can be resident together.
LENS_SIDES = ("output", "input")
#: Kinds the transport estimator fits: all four, at either or both sites.
FITTABLE_KINDS = LENS_KINDS
#: Documents a local fit averages over unless asked otherwise.  Measured on
#: SmolLM2-360M (FineWeb-Edu, MPS, both sites, 2026-10-31), the split-half
#: difference per layer at the ``out`` site:
#:
#: - 64 documents (even 32/32 halves): the R-lens 0.02 to 0.28, median 0.17
#:   (``mid`` 0.02 to 0.27, median 0.17), in 330 s; the S-lens 0.06 to 0.22,
#:   median 0.12 (``mid`` 0.01 to 0.16, median 0.12), in 342 s.
#: - 25 documents: the R-lens 0.04 to 0.45, median 0.27; the S-lens 0.12 to
#:   0.37, median 0.18, in 110 to 136 s.
#:
#: So 64 documents cut the medians to 0.62x (R) and 0.65x (S) of 25's, close
#: to the 1/sqrt(n) averaging-noise estimate of 0.63x, for 2.5 to 3x the
#: wall time.  The research lenses of record used 1,000.
DEFAULT_FIT_DOCUMENTS = 64

#: Documents ``lens counts`` streams by default (:mod:`drowse.lens.token_counts`):
#: at the lens corpus's 4,000 characters a document, about 16M tokens, where
#: the S mask keeps a token seen once (neuron-lens's rate of 5 per 88M) and
#: agrees with neuron-lens's 88M-token count on 97.5% of its kept tokens
#: (binomial thinning of its Qwen3-8B counts; 95.6% of what it keeps are
#: neuron-lens's).
DEFAULT_COUNT_DOCUMENTS = 16384

#: A local lens directory: the kind's letter, optionally ``-<tag>``
#: (``r``, ``j``, ``r-chat``).  The name carries the kind, so ``local:r`` is
#: always an R-lens and a fit can never land in another kind's directory.
LOCAL_LENS_NAME = re.compile(r"^[jrks](-[a-z0-9][a-z0-9._-]{0,61})?$")


@dataclass(frozen=True)
class LensKey:
    """One lens identity: a kind (J/R/K/S) at a site (out/mid)."""

    kind: str
    site: str = "out"

    def __post_init__(self) -> None:
        if self.kind not in LENS_KINDS:
            raise ValueError(
                f"lens kind must be one of {', '.join(LENS_KINDS)}, got {self.kind!r}"
            )
        if self.site not in LENS_SITES:
            raise ValueError(
                f"lens site must be one of {', '.join(LENS_SITES)}, got {self.site!r}"
            )

    @property
    def side(self) -> str:
        """``output`` for J/R (read through ``W_U``), ``input`` for K/S."""
        return "output" if self.kind in ("J", "R") else "input"

    @property
    def relp(self) -> bool:
        """Whether the Jacobian was taken under the RelP backward rules."""
        return self.kind in ("R", "S")

    @property
    def label(self) -> str:
        """The stable ``KIND:site`` spelling bindings and logs use."""
        return f"{self.kind}:{self.site}"

    @classmethod
    def parse(cls, label: str) -> "LensKey":
        kind, sep, site = label.partition(":")
        if not sep:
            raise ValueError(f"lens key must be KIND:site, got {label!r}")
        return cls(kind, site)


def site_layers(kind: str, site: str, n_layers: int) -> list[int]:
    """Every layer a ``kind`` lens holds at ``site`` on an ``n_layers`` model.

    The output kinds' ``out`` site stops below the last block, whose output is
    their target (its Jacobian is the identity); every other site covers
    every block.
    """
    key = LensKey(kind, site)
    top = n_layers - 1 if (key.side == "output" and site == "out") else n_layers
    return list(range(max(top, 0)))


def lens_selectable(sites: Iterable[str]) -> bool:
    """Whether a lens holding ``sites`` can be its family's active source.

    Both families' probes, gates, steering atoms and the R family's live
    readout read the ``out`` site, so a lens of either family without it
    cannot serve its family: publishing one leaves the selection (and the
    resident lens) where it was, rather than displace a readable source with
    one those surfaces refuse.  Select it explicitly to read it at ``mid``.
    """
    return "out" in set(sites)


def local_lens_name(kind: str, tag: str | None = None) -> str:
    """The local directory name for ``kind`` (``R`` → ``r``, ``r-<tag>``)."""
    LensKey(kind)
    name = kind.lower() if tag is None else f"{kind.lower()}-{tag}"
    if LOCAL_LENS_NAME.fullmatch(name) is None:
        raise ValueError(f"invalid local lens tag {tag!r}")
    return name


def local_lens_kind(name: str) -> str:
    """The kind a local lens name encodes (``r-chat`` → ``R``)."""
    if LOCAL_LENS_NAME.fullmatch(name) is None:
        raise ValueError(
            f"invalid local lens name {name!r} (must match {LOCAL_LENS_NAME.pattern})"
        )
    return name[0].upper()
