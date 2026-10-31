"""Unit references: ``neuron/L8n344`` and ``head/L20h17`` names."""

from __future__ import annotations

import pytest

from drowse.errors import RequestError, UnitRefError
from drowse.naming import (
    RESERVED_NAMESPACES,
    UNIT_NAMESPACES,
    HeadRef,
    NeuronRef,
    parse_unit_ref,
)


@pytest.mark.parametrize("text", ["L8n344", "l8N344", "L8N344", "l08n0344"])
def test_neuron_refs_parse_case_insensitively_to_canonical_text(text: str) -> None:
    ref = parse_unit_ref("neuron", text)
    assert ref == NeuronRef(8, 344)
    assert str(ref) == "L8n344"
    assert ref.qualified == "neuron/L8n344"


@pytest.mark.parametrize("text", ["L20h17", "l20H17", "L020h017"])
def test_head_refs_parse_case_insensitively_to_canonical_text(text: str) -> None:
    ref = parse_unit_ref("head", text)
    assert ref == HeadRef(20, 17)
    assert (str(ref), ref.qualified) == ("L20h17", "head/L20h17")


@pytest.mark.parametrize("namespace,text", [
    ("neuron", "L8h344"),      # a head's letter under neuron/
    ("head", "L20n17"),        # a neuron's letter under head/
    ("neuron", "8n344"),
    ("neuron", "L8n"),
    ("neuron", "Ln344"),
    ("neuron", "L8n344x"),
    ("neuron", "L-1n3"),
    ("neuron", " L8n344"),
    ("neuron", ""),
    ("jlens", "L8n344"),       # not a unit namespace
])
def test_malformed_unit_names_raise_a_request_error(namespace: str, text: str) -> None:
    with pytest.raises(UnitRefError) as info:
        parse_unit_ref(namespace, text)
    assert isinstance(info.value, RequestError)
    assert info.value.user_message()[0] == 400


def test_range_is_not_checked_here() -> None:
    """Whether the layer and index exist is a check against the model."""
    assert parse_unit_ref("neuron", "L999n123456") == NeuronRef(999, 123456)


def test_refs_are_frozen_ordered_and_distinct_by_kind() -> None:
    assert sorted([NeuronRef(2, 1), NeuronRef(1, 5)]) == [NeuronRef(1, 5), NeuronRef(2, 1)]
    assert NeuronRef(1, 2) != HeadRef(1, 2)
    assert len({NeuronRef(1, 2), NeuronRef(1, 2), HeadRef(1, 2)}) == 2
    with pytest.raises(AttributeError):
        NeuronRef(1, 2).layer = 3  # type: ignore[misc]


def test_unit_namespaces_are_reserved() -> None:
    assert UNIT_NAMESPACES == {"neuron", "head"}
    assert UNIT_NAMESPACES <= RESERVED_NAMESPACES
