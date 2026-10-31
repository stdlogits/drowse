"""Concept naming rules — the on-disk slug, the bipolar-pole join, reserved
namespaces, and unit references.

Pure string rules, no model or IO, at the bottom of the layer order so the
store, the steering grammar and the session share one definition
(:func:`canonical_concept_name`, :data:`RESERVED_NAMESPACES`,
:func:`parse_unit_ref`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from drowse.errors import UnitRefError

_SLUG_RE = re.compile(r"[^a-z0-9]+")
BIPOLAR_SEP = "."

#: Instrument namespaces: atoms under them name instrument units (J-/S-lens
#: tokens, SAE features, neurons, heads), not fitted concepts.  Their names
#: skip slug canonicalization and take no variant, and no manifold may be
#: authored under them.
RESERVED_NAMESPACES = frozenset({"jlens", "slens", "sae", "neuron", "head"})

#: The reserved namespaces whose names are model units (:class:`NeuronRef`,
#: :class:`HeadRef`).  A unit name carries its layer.
UNIT_NAMESPACES = frozenset({"neuron", "head"})


def _slug(s: str) -> str:
    """Normalize a single pole label to `[a-z0-9_]`.

    Collapses any non-alphanumeric run to `_`. Never produces the bipolar
    separator `.` — that is reserved for joining two slugged poles in
    `canonical_concept_name`.
    """
    return _SLUG_RE.sub("_", s.strip().lower()).strip("_")


def canonical_concept_name(concept: str, baseline: str | None = None) -> str:
    """Return the canonical on-disk name for a concept.

    Monopolar: `_slug(concept)`.
    Bipolar:   `f"{_slug(concept)}.{_slug(baseline)}"`.

    If `baseline` is None and `concept` already contains the bipolar
    separator `.`, the input is treated as a pre-composed bipolar name
    and each side is slugged independently. This makes `/steer happy.sad`
    and `/steer happy - sad` resolve to the same cache entry.
    """
    if baseline is None:
        if BIPOLAR_SEP in concept:
            pos, neg = concept.split(BIPOLAR_SEP, 1)
            return f"{_slug(pos)}{BIPOLAR_SEP}{_slug(neg)}"
        return _slug(concept)
    return f"{_slug(concept)}{BIPOLAR_SEP}{_slug(baseline)}"


# -- unit references ----------------------------------------------------------

#: ``L<layer><letter><index>``, case-insensitive.  The letter form (rather
#: than ``8.344``) is what the steering lexer can keep as one identifier.
#: ASCII digits only: ``\d`` would take any Unicode decimal digit and
#: canonicalize ``L٨n344`` to ``L8n344``.
_UNIT_RE = re.compile(r"[Ll]([0-9]+)([NnHh])([0-9]+)")


@dataclass(frozen=True, order=True)
class NeuronRef:
    """MLP neuron ``index`` of block ``layer``: coordinate ``index`` of the
    down projection's input.  ``str(ref)`` is the canonical ``L8n344``."""

    layer: int
    index: int

    namespace: ClassVar[str] = "neuron"
    letter: ClassVar[str] = "n"

    def __str__(self) -> str:
        return f"L{self.layer}{self.letter}{self.index}"

    @property
    def qualified(self) -> str:
        """The namespaced atom, ``neuron/L8n344``."""
        return f"{self.namespace}/{self}"


@dataclass(frozen=True, order=True)
class HeadRef:
    """Query head ``index`` of block ``layer``'s attention.  ``str(ref)`` is
    the canonical ``L20h17``."""

    layer: int
    index: int

    namespace: ClassVar[str] = "head"
    letter: ClassVar[str] = "h"

    def __str__(self) -> str:
        return f"L{self.layer}{self.letter}{self.index}"

    @property
    def qualified(self) -> str:
        """The namespaced atom, ``head/L20h17``."""
        return f"{self.namespace}/{self}"


UnitRef = NeuronRef | HeadRef

_UNIT_TYPES: dict[str, type[NeuronRef] | type[HeadRef]] = {
    NeuronRef.namespace: NeuronRef,
    HeadRef.namespace: HeadRef,
}


def parse_unit_ref(namespace: str, text: str) -> UnitRef:
    """Parse the name after ``neuron/`` or ``head/`` into its reference.

    Accepts ``L8n344`` / ``L20h17`` in any letter case (``l8N344``); leading
    zeros are dropped, so ``str`` of the result is the canonical form.  The
    letter must match the namespace (``n`` for neurons, ``h`` for heads).
    Raises :class:`~drowse.errors.UnitRefError` (400) otherwise.  Whether
    the layer and index exist is checked against the model, not here.
    """
    cls = _UNIT_TYPES.get(namespace)
    if cls is None:
        raise UnitRefError(
            f"'{namespace}/' is not a unit namespace "
            f"(units: {', '.join(sorted(UNIT_NAMESPACES))})"
        )
    match = _UNIT_RE.fullmatch(text)
    if match is None or match.group(2).lower() != cls.letter:
        example = "L8n344" if cls is NeuronRef else "L20h17"
        raise UnitRefError(
            f"'{namespace}/{text}' is not a {namespace} reference; write "
            f"L<layer>{cls.letter}<index> (e.g. '{namespace}/{example}')"
        )
    return cls(int(match.group(1)), int(match.group(3)))
