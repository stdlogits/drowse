"""Lens identity: which Jacobian a lens holds, and at which residual site.

A lens is one estimator run's per-layer matrices.  Its **kind** is the
direction of the Jacobian crossed with the backward rules it was taken under:

====  ======  ========  =====================================================
kind  family  rules     matrix
====  ======  ========  =====================================================
J     R       standard  ``J_L = E[Σ_{t'≥t} ∂h_final(t')/∂h_L(t)]`` (reverse)
R     R       RelP      J under the LN / identity / half rules
K     S       standard  ``K_L = E[Σ_{s≤t} ∂h_L(t)/∂e(s)]`` (forward)
S     S       RelP      K under the same rules
====  ======  ========  =====================================================

Its **site** is the residual read: ``out`` (a block's output) or ``mid`` (the
residual after the attention add — the MLP's input).  The R family (J, R:
output Jacobians, read through ``W_U``) reads an activation as what it is
disposed to make the model *say*; the S family (K, S: input Jacobians, read
against the input embedding) reads it as what it has *read*.  One estimator (transport, in
:mod:`drowse.lens.transport`) fits every kind at both sites; every store,
binding and selection path is keyed on :class:`LensKey`.

Torch-free: the CLI, server, and source registry parse keys and local names
without importing the readout math.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Literal

LENS_KINDS = ("J", "R", "K", "S")
LENS_SITES = ("out", "mid")
#: A lens family: ``R`` (J/R) or ``S`` (K/S).
LensFamily = Literal["R", "S"]
#: The families, R first.  Selection holds one active source per family, so
#: an R-family and an S-family lens can be resident together.
LENS_FAMILIES: tuple[LensFamily, ...] = ("R", "S")
#: A family's Jacobian direction: ``output`` (reverse, J/R) or ``input``
#: (forward, K/S).  It is the estimator's mode and the family's on-disk
#: spelling, frozen in two formats: lens format v7's
#: ``estimator_policy.direction`` and the source selection's slot keys
#: (``active.json``).  Nothing else spells a family this way.
FAMILY_DIRECTION: dict[str, str] = {"R": "output", "S": "input"}
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
#: FineWeb-Edu averaged 585 tokens a document (2,048 documents, SmolLM2's
#: tokenizer, sliced at 4,000 characters), so about 19M tokens, where the S mask
#: keeps a token seen twice (neuron-lens's rate of 5 per 88M).  Binomial
#: thinning of neuron-lens's Qwen3-8B counts to 16M tokens at that threshold
#: keeps 92.7% of its kept set, 99.3% of what it keeps being in it.
DEFAULT_COUNT_DOCUMENTS = 32768

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
    def family(self) -> LensFamily:
        """``R`` for J/R (read through ``W_U``), ``S`` for K/S."""
        return "R" if self.kind in ("J", "R") else "S"

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
    top = n_layers - 1 if (key.family == "R" and site == "out") else n_layers
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
