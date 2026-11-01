"""The parsed-IR golden: ``tests/fixtures/steering_ir.json``.

``browser_steering_expression.json`` pins only accept/reject and
``referenced_selectors``, because drowse.ai's TypeScript parser runs the same
table.  This golden pins what the parser *produces*, for Python alone until
drowse.ai's parser exists and matches it: for every accepted expression, the
ordered ``Steering.alphas`` terms (key, kind, coefficient or along/onto,
operands, position, layer scope, canonical trigger with its fields, the
term's own text) and the canonical :func:`format_expr` string; for every
rejected one, the error class (the wire ``code``).

Bare atoms resolve against installed manifolds, so the fixture carries its
own registry (``registry``: namespace, name, fit mode, node labels), installed
into a fresh home before any case parses.  A case may name ``profileNames``
(the session-local override set).

Floats compare through ``json.dumps``, so ``-0.0`` versus ``0.0`` and the
exact binary value (``0.30000000000000004``) are part of the pin.

Regenerate after a deliberate grammar change (Python is the source of truth;
the inputs stay, the outputs are recomputed)::

    PYTHONPATH=. python -m tests.steering.test_steering_ir
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import pytest

from drowse.steering.grammar import (
    AblationTerm,
    ManifoldTerm,
    ProjectedTerm,
    UnitAblationTerm,
    format_expr,
    parse_expr,
    term_parts,
)
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger

FIXTURE = Path(__file__).parent.parent / "fixtures" / "steering_ir.json"
SCHEMA_VERSION = 1


def _trigger_record(trig: Trigger, text: str | None) -> dict[str, Any] | None:
    """``None`` for both phases; else the canonical text and every field."""
    if trig == Trigger.BOTH:
        return None
    gate = trig.gate
    return {
        "text": text,
        "prompt": trig.prompt,
        "generated": trig.generated,
        "thinking": trig.thinking,
        "response": trig.response,
        "firstN": trig.first_n,
        "afterN": trig.after_n,
        "gate": None if gate is None else {
            "probe": gate.probe, "op": gate.op, "threshold": gate.threshold,
        },
    }


def _layers_record(layers: Any) -> Any:
    return list(layers) if isinstance(layers, tuple) else layers


def ir_record(steering: Steering) -> dict[str, Any]:
    """The golden's record of one parsed expression."""
    terms: list[dict[str, Any]] = []
    for (key, val), part in zip(steering.alphas.items(), term_parts(steering)):
        rec: dict[str, Any] = {"key": key, "text": part.text}
        trig: Trigger
        if isinstance(val, ManifoldTerm):
            trig = val.trigger
            position = val.position
            rec.update(
                kind="manifold", manifold=val.manifold,
                position=position if isinstance(position, str) else list(position),
                along=val.along, onto=val.onto, layers=_layers_record(val.layers),
            )
        elif isinstance(val, ProjectedTerm):
            trig = val.trigger
            rec.update(
                kind="projection", base=val.base, operator=val.operator,
                onto=val.onto, coeff=val.coeff, layers=_layers_record(val.layers),
            )
        elif isinstance(val, UnitAblationTerm):
            trig = val.trigger
            rec.update(kind="unitAblation", unit=val.unit.qualified, coeff=val.coeff)
        elif isinstance(val, AblationTerm):
            trig = val.trigger
            rec.update(
                kind="ablation", target=val.target, coeff=val.coeff,
                layers=_layers_record(val.layers),
            )
        else:
            coeff, trig = (
                (float(val[0]), val[1]) if isinstance(val, tuple)
                else (float(val), steering.trigger)
            )
            rec.update(
                kind="vector", target=part.target, coeff=coeff,
                layers=_layers_record(part.layers),
            )
        rec["trigger"] = _trigger_record(trig, part.trigger)
        terms.append(rec)
    return {"canonical": format_expr(steering), "terms": terms}


def install_registry(registry: list[dict[str, Any]]) -> None:
    """Author the fixture's manifolds (folders only, nothing fitted)."""
    from drowse.store import selectors
    from drowse.store.manifold_authoring import create_discover_manifold_folder

    for entry in registry:
        create_discover_manifold_folder(
            entry["namespace"], entry["name"], "steering IR golden",
            fit_mode=entry["fitMode"],
            node_corpora={label: [f"{label} statement."] for label in entry["labels"]},
            hyperparams={"max_dim": 1},
        )
    selectors.invalidate()


def _parse(case: dict[str, Any]) -> Steering:
    names = case.get("profileNames")
    return parse_expr(
        case["expression"], profile_names=None if names is None else set(names),
    )


def _error_name(case: dict[str, Any]) -> str:
    try:
        _parse(case)
    except Exception as exc:  # noqa: BLE001 - the class name is the record
        return type(exc).__name__
    raise AssertionError(f"{case['expression']!r} parsed; the golden says it is rejected")


GOLDEN = json.loads(FIXTURE.read_text())


@pytest.fixture
def ir_home(isolated_home: Path) -> Iterator[Path]:
    install_registry(GOLDEN["registry"])
    yield isolated_home


def test_fixture_schema() -> None:
    assert GOLDEN["schemaVersion"] == SCHEMA_VERSION
    expressions = [c["expression"] for c in GOLDEN["accepted"]]
    expressions += [c["expression"] for c in GOLDEN["rejected"]]
    assert len(expressions) == len(set(expressions)), "duplicate expression"


@pytest.mark.parametrize(
    "case", GOLDEN["accepted"], ids=[c["expression"] for c in GOLDEN["accepted"]],
)
def test_parsed_ir_matches_golden(ir_home: Path, case: dict[str, Any]) -> None:
    steering = _parse(case)
    expected = {"canonical": case["canonical"], "terms": case["terms"]}
    assert json.dumps(ir_record(steering)) == json.dumps(expected)
    # The canonical form is a fixed point: it parses to a value that renders
    # to itself (``%g`` is lossy beyond six digits, so the value may differ).
    again = _parse({**case, "expression": case["canonical"]})
    assert format_expr(again) == case["canonical"]


@pytest.mark.parametrize(
    "case", GOLDEN["rejected"], ids=[c["expression"] for c in GOLDEN["rejected"]],
)
def test_rejected_expression_matches_golden(ir_home: Path, case: dict[str, Any]) -> None:
    assert _error_name(case) == case["error"]


def regenerate(path: Path = FIXTURE) -> None:
    """Recompute every output in ``path`` from its inputs, in a scratch home."""
    from drowse.store import selectors

    data = json.loads(path.read_text())
    with tempfile.TemporaryDirectory() as home:
        previous = os.environ.get("DROWSE_HOME")
        os.environ["DROWSE_HOME"] = home
        selectors.invalidate()
        try:
            install_registry(data["registry"])
            accepted = []
            for case in data["accepted"]:
                inputs = {
                    k: case[k] for k in ("expression", "profileNames", "note")
                    if k in case
                }
                accepted.append({**inputs, **ir_record(_parse(case))})
            rejected = []
            for case in data["rejected"]:
                inputs = {k: case[k] for k in ("expression", "profileNames", "note") if k in case}
                rejected.append({**inputs, "error": _error_name(case)})
        finally:
            if previous is None:
                os.environ.pop("DROWSE_HOME", None)
            else:
                os.environ["DROWSE_HOME"] = previous
            selectors.invalidate()
    data.update(schemaVersion=SCHEMA_VERSION, accepted=accepted, rejected=rejected)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    regenerate()
