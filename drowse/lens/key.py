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
#: Documents a local fit averages over unless asked otherwise.  At 25
#: FineWeb-Edu documents on SmolLM2-360M the split-half difference was large:
#: the R-lens's ``out`` site ranged from 0.04 (last layers) to 0.45 (early
#: layers), median 0.27; the S-lens's from 0.12 to 0.37, median 0.18, at 110
#: to 136 s per kind on MPS.  64 documents split into even 32/32 halves and,
#: since the difference is averaging noise (1/sqrt n), should land near 0.6x
#: those values for about 2.5x the wall time.  The research lenses of record
#: used 1,000.
DEFAULT_FIT_DOCUMENTS = 64

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
