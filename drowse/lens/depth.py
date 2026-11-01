"""Which lens kind reads a place, by depth: S early, R late, both in a band.

The S lens (input Jacobians) is the faithful read of what a vector at a
shallow layer has *read*; the R lens (output Jacobians) of what a deep one
will make the model *say*.  Between them is a crossover band where both reads
show, weighted.  The band is a per-model prior (neuron-lens §6–7 put Qwen3-8B's S
reads through about L8 and R reads from about L4 on writes, L9 on reads); until
it is measured per model the default is 15–35% of depth.

Depth is ``layer / (n_layers − 1)``, the same normalized depth the lens
readout reports.  Below ``lo`` the weight is all S, above ``hi`` all R, and
linear in between, so ``w_S + w_R = 1`` at every layer.  Pure and torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BAND_LO = 0.15
DEFAULT_BAND_HI = 0.35


@dataclass(frozen=True)
class LensBand:
    """The S→R crossover band as fractions of depth, ``0 ≤ lo ≤ hi ≤ 1``.

    ``lo == hi`` is a hard switch at that depth (S at or below it).
    """

    lo: float = DEFAULT_BAND_LO
    hi: float = DEFAULT_BAND_HI

    def __post_init__(self) -> None:
        if not 0.0 <= self.lo <= self.hi <= 1.0:
            raise ValueError(
                f"a lens band needs 0 <= lo <= hi <= 1, got ({self.lo}, {self.hi})"
            )

    def s_weight(self, depth: float) -> float:
        """``w_S`` at normalized ``depth``; ``w_R = 1 − w_S``."""
        if depth <= self.lo:
            return 1.0
        if depth >= self.hi:
            return 0.0
        return (self.hi - depth) / (self.hi - self.lo)


def layer_depth(layer: int, n_layers: int) -> float:
    """``layer / (n_layers − 1)``, 0 for a one-layer model."""
    return 0.0 if n_layers <= 1 else int(layer) / (int(n_layers) - 1)


def s_weights(
    layers: "list[int] | tuple[int, ...]", n_layers: int, band: LensBand | None = None,
) -> list[float]:
    """``w_S`` at each of ``layers``, normalized to sum 1 over them (all
    zeros when none has S weight): the ``slens/`` probe's layer weights."""
    table = kind_weights(n_layers, band)
    raw = [table[int(layer)][0] for layer in layers]
    total = sum(raw)
    return [w / total for w in raw] if total > 0 else [0.0] * len(raw)


def kind_weights(
    n_layers: int, band: LensBand | None = None,
) -> list[tuple[float, float]]:
    """``(w_S, w_R)`` for every layer ``0..n_layers−1``, each pair summing to 1."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")
    band = LensBand() if band is None else band
    out: list[tuple[float, float]] = []
    for layer in range(n_layers):
        w_s = band.s_weight(layer_depth(layer, n_layers))
        out.append((w_s, 1.0 - w_s))
    return out


__all__ = [
    "DEFAULT_BAND_HI",
    "DEFAULT_BAND_LO",
    "LensBand",
    "kind_weights",
    "layer_depth",
    "s_weights",
]
