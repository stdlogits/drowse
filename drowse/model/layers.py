"""Layer selectors: the one parser, formatter and resolver for layer choices.

A selector is ``None`` or ``"all"`` (every available layer), ``"workspace"``
(the 40-90% depth band), ``"sample"`` (nine evenly spaced layers, a readout
convenience that fits and steering refuse), or an explicit index list.  The
text form of a list is comma-separated entries, each an index (``12``) or an
inclusive range (``4-8``), so ``"4-8,20"`` is ``[4, 5, 6, 7, 8, 20]``.  The
CLI, the server, the lens runtime and lens fits, manifold fits and the
steering grammar's ``{…}`` layer clause all parse and resolve through here.

Torch-free at import: the CLI parses selectors before any model loads.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from drowse.errors import RequestError

#: The named selector modes.
LAYER_MODES = ("all", "workspace", "sample")
#: How many layers ``"sample"`` picks.
SAMPLE_COUNT = 9

LayerSelector = Sequence[int] | str | None

# One list entry: an index or an inclusive ``a-b`` range.
_ENTRY_RE = re.compile(r"(\d+)\s*(?:-\s*(\d+))?")


class LayerSelectorError(RequestError):
    """A malformed layer selector.

    ``offset`` is the character offset into the selector text where the
    failing entry starts (``None`` when the whole text is at fault), so a
    caller embedding the selector in a longer string can point at it.
    """

    def __init__(self, msg: str, *, offset: int | None = None) -> None:
        self.offset = offset
        super().__init__(msg)


def _modes_text(allow_sample: bool) -> str:
    return ", ".join(m for m in LAYER_MODES if allow_sample or m != "sample")


def parse_layer_selector(
    text: str | None, *, allow_sample: bool = True,
) -> list[int] | str | None:
    """``"4-6,11"`` -> ``[4, 5, 6, 11]``; a mode passes through lower-cased.

    Entries keep their written order (a range expands in place).  ``None`` or
    blank text is ``None`` (the caller's default).  An empty entry, a
    negative index, a reversed range (``8-4``), a layer named twice, or
    ``sample`` when ``allow_sample`` is false raises
    :class:`LayerSelectorError` carrying the failing entry's offset.
    """
    if text is None or not text.strip():
        return None
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in LAYER_MODES:
        if lowered == "sample" and not allow_sample:
            raise LayerSelectorError(
                f"unknown layer mode {stripped!r} (want "
                f"{_modes_text(allow_sample)}, or an index list)",
                offset=text.index(stripped[0]),
            )
        return lowered
    layers: list[int] = []
    seen: set[int] = set()
    start = 0
    for part in text.split(","):
        body = part.strip()
        offset = start + (part.index(body[0]) if body else len(part))
        start += len(part) + 1
        if not body:
            raise LayerSelectorError(
                f"empty entry in layers {text!r}", offset=offset,
            )
        match = _ENTRY_RE.fullmatch(body)
        if match is None:
            if body.startswith("-") and _ENTRY_RE.fullmatch(body[1:].lstrip()):
                raise LayerSelectorError(
                    f"layer indices are non-negative; got {body!r}",
                    offset=offset,
                )
            raise LayerSelectorError(
                f"malformed layers {text!r} (want comma-separated indices or "
                f"a-b ranges, or one of {_modes_text(allow_sample)})",
                offset=offset,
            )
        lo = int(match.group(1))
        hi = lo if match.group(2) is None else int(match.group(2))
        if hi < lo:
            raise LayerSelectorError(
                f"reversed layer range {body!r} (write {hi}-{lo})",
                offset=offset,
            )
        for layer in range(lo, hi + 1):
            if layer in seen:
                raise LayerSelectorError(
                    f"layer {layer} is named twice in {text!r}", offset=offset,
                )
            seen.add(layer)
            layers.append(layer)
    return layers


def format_layer_indices(indices: Iterable[int]) -> str:
    """The canonical text of an index set: sorted, each maximal run of
    consecutive indices as ``a-b`` and a lone index as ``a`` (``"4-8,20"``).

    The inverse of :func:`parse_layer_selector` on index lists, up to order
    and duplicates.
    """
    ordered = sorted({int(i) for i in indices})
    runs: list[str] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        runs.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(runs)


def resolve_layer_selector(
    selector: LayerSelector,
    available: Sequence[int],
    *,
    n_layers: int,
    allow_sample: bool = True,
) -> list[int]:
    """Resolve ``selector`` against the ``available`` layers.

    ``available`` is what the selection draws from (a lens's fitted layers,
    or every source layer of a fit), ``n_layers`` the model depth that the
    workspace band is measured against.  An explicit list is returned as
    given, in order; callers that need a canonical set sort it themselves.
    """
    pool = [int(layer) for layer in available]
    if selector is None or selector == "all":
        return pool
    if isinstance(selector, str):
        mode = selector.lower()
        if mode == "workspace":
            from drowse.model.arch import workspace_layer_indices

            return workspace_layer_indices(pool, n_layers)
        if mode == "sample" and allow_sample:
            if len(pool) <= SAMPLE_COUNT:
                return pool
            step = (len(pool) - 1) / (SAMPLE_COUNT - 1)
            return sorted({pool[round(i * step)] for i in range(SAMPLE_COUNT)})
        raise RequestError(
            f"unknown layer mode {selector!r} (want "
            f"{_modes_text(allow_sample)}, or an index list)"
        )
    return [int(layer) for layer in selector]


def resolve_model_layers(selector: LayerSelector, n_layers: int) -> list[int]:
    """Resolve a fit's layer request against every layer of an ``n_layers`` model.

    Returns the canonical sorted, de-duplicated index set.  A string parses
    through :func:`parse_layer_selector`; ``sample`` is refused; an empty
    request or an index outside ``[0, n_layers)`` raises ``ValueError``.  A
    manifold fit and its weight-free preflight (which resolves against the
    config's layer count) both resolve here, so a proof covers exactly the
    roster the fit would evaluate.
    """
    if isinstance(selector, str):
        parsed = parse_layer_selector(selector, allow_sample=False)
        if parsed is None:
            raise RequestError("layers must name at least one layer")
        selector = parsed
    requested = sorted(set(resolve_layer_selector(
        selector, range(n_layers), n_layers=n_layers, allow_sample=False,
    )))
    if not requested:
        raise RequestError("layers must name at least one layer")
    if requested[0] < 0 or requested[-1] >= n_layers:
        raise RequestError(f"layers must lie in [0, {n_layers}); got {requested}")
    return requested


__all__ = [
    "LAYER_MODES",
    "LayerSelector",
    "LayerSelectorError",
    "SAMPLE_COUNT",
    "format_layer_indices",
    "parse_layer_selector",
    "resolve_layer_selector",
    "resolve_model_layers",
]
