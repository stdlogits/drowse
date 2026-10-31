"""Unit atoms in the grammar: ``neuron/…`` pushes, ``!neuron/…``/``!head/…`` ablations, unit probes."""

from __future__ import annotations

import re
from typing import Any

import pytest

from drowse.naming import HeadRef, NeuronRef
from drowse.steering.grammar import (
    AblationTerm,
    SteeringExprError,
    UnitAblationTerm,
    format_expr,
    parse_expr,
    referenced_selectors,
)
from drowse.steering.triggers import Trigger


def test_a_neuron_push_is_a_plain_term_under_its_canonical_key() -> None:
    steering = parse_expr("0.2 neuron/l8N344")
    assert dict(steering.alphas) == {"neuron/L8n344": 0.2}
    assert format_expr(steering) == "0.2 neuron/L8n344"
    negative = parse_expr("-0.3 neuron/L08n0344@response")
    assert dict(negative.alphas) == {"neuron/L8n344": (-0.3, Trigger.GENERATED_ONLY)}


@pytest.mark.parametrize("text,unit,coeff", [
    ("!neuron/L8n344", NeuronRef(8, 344), 1.0),
    ("!head/l20H17", HeadRef(20, 17), 1.0),
    ("0.5 !head/L20h17", HeadRef(20, 17), 0.5),
    ("-!neuron/L1n2", NeuronRef(1, 2), -1.0),
])
def test_unit_ablation_is_its_own_term(text: str, unit: Any, coeff: float) -> None:
    (key, term), = parse_expr(text).alphas.items()
    assert key == f"!{unit.qualified}"
    assert term == UnitAblationTerm(unit=unit, coeff=coeff, trigger=Trigger.BOTH)
    assert not isinstance(term, AblationTerm)


def test_repeated_unit_ablations_merge_and_conflicting_triggers_raise() -> None:
    merged = parse_expr("!neuron/L1n2 + 0.5 !neuron/l1n2")
    assert dict(merged.alphas) == {
        "!neuron/L1n2": UnitAblationTerm(NeuronRef(1, 2), 1.5, Trigger.BOTH),
    }
    with pytest.raises(SteeringExprError, match="conflicting triggers"):
        parse_expr("!neuron/L1n2 + !neuron/L1n2@response")


@pytest.mark.parametrize("text", [
    "0.2 neuron/L8n344",
    "!neuron/L8n344",
    "-!head/L20h17@after",
    "0.5 !head/L3h1@first:4",
    "0.3 honest + !head/L3h1 - 0.2 neuron/L2n5@after",
    "x@when:neuron/L8n344>2",
    "x@when:head/L20h17<=-0.5",
    "0.2 neuron/L8n344@response&when:head/L2h1>0",
    "!neuron/L8n344@when:neuron/L8n344>1",
])
def test_unit_expressions_round_trip(text: str) -> None:
    steering = parse_expr(text)
    assert parse_expr(format_expr(steering)) == steering


def test_unit_probes_are_canonical_probe_names() -> None:
    (entry,) = parse_expr("x@when:neuron/l8n344 > 2").alphas.values()
    assert isinstance(entry, tuple) and entry[1].gate is not None
    assert entry[1].gate.probe == "neuron/L8n344"
    (entry,) = parse_expr("x@when:head/L20H17<0").alphas.values()
    assert isinstance(entry, tuple) and entry[1].gate is not None
    assert entry[1].gate.probe == "head/L20h17"


def test_referenced_selectors_carry_canonical_unit_names() -> None:
    assert referenced_selectors("0.2 neuron/l8n344 + !head/L1H2") == [
        ("neuron", "L8n344", "raw"), ("head", "L1h2", "raw"),
    ]


@pytest.mark.parametrize("text,fragment,col", [
    ("0.5 head/L20h17", "a head has no single write direction; silence it with !head/L20h17", 4),
    ("head/L3h1@response", "silence it with !head/L3h1", 0),
    ("0.2 neuron/L8h344", "not a neuron reference", 11),
    ("!head/L3n1", "not a head reference", 6),
    ("0.2 neuron/8", "'neuron/' takes a unit", 11),
    ("neuron/#3", "'neuron/' takes a unit", 7),
    ("neuron/L8n344.x", "takes no '.'", 13),
    ("!neuron/L8n344~honest", "does not compose", 14),
    ("honest~neuron/L1n2", "does not compose", 7),
    ("honest|head/L1h2", "does not compose", 7),
    ("neuron/L1n2%x", "does not compose", 11),
    ("0.2 neuron/L8n344{8}", "take no layer clause", 17),
    ("!head/L3h1{3}", "take no layer clause", 10),
    ("x@when:neuron/L8x3>0", "not a neuron reference", 14),
    ("x@when:head/L8h3.y>0", "takes no '.'", 16),
    ("0.5 honest@when:neuron/L8n344~x > 0.5", "takes no channel suffix ('~')", 29),
    ("x@when:head/L3h1~foo > 0.5", "takes no channel suffix ('~')", 16),
    ("x@when:neuron/L8n344:fraction>0", "takes no channel suffix (':')", 20),
    ("x@when:neuron/L8n344@x>0", "takes no channel suffix ('@')", 20),
    ("x@when:head/L3h1[0]>0", "takes no channel suffix ('[')", 16),
])
def test_malformed_unit_expressions_fail_with_a_column(text: str, fragment: str, col: int) -> None:
    with pytest.raises(SteeringExprError, match=re.escape(fragment)) as info:
        parse_expr(text)
    assert info.value.col == col


def test_a_neuron_push_steers_through_the_units_family(tiny_session: Any) -> None:
    # The full semantics are pinned in tests/units/test_unit_steering.py.
    with tiny_session.steering("0.2 neuron/L1n3"):
        assert set(tiny_session._steering.hooks) == {1}
    assert "neuron/L1n3" in tiny_session.profiles


@pytest.mark.parametrize("text", ["!neuron/L1n3", "0.5 !head/L0h1@response"])
def test_a_unit_ablation_composes_to_a_unit_hook(tiny_session: Any, text: str) -> None:
    with tiny_session.steering(text):
        assert tiny_session._steering.unit_hooks
        assert not tiny_session._steering.hooks
