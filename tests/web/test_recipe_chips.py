"""The recipe chips write the server's canonical expression, and rebuild from its parts.

``tests/fixtures/recipe_chips.json`` is shared with ``webui/scripts/recipe-chips.test.mjs``: the
node suite checks that each rack's chips write the fixture's expression and that the chips rebuilt
from its ``parts`` write it again; this side checks that the expression is what ``format_expr``
writes for it (every ablation spelled the server's way: ``!x`` at full strength, ``0.5 !x``
otherwise; numbers as ``%g``) and that ``parts`` is what ``POST .../steering/validate`` answers.
There is one grammar: the TypeScript side never parses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from drowse.server.session_routes import _part_json
from drowse.steering.grammar import format_expr, parse_expr, term_parts

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "recipe_chips.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_chips_write_the_canonical_expression(case: dict[str, Any]) -> None:
    expression = case["expression"]
    assert format_expr(parse_expr(expression)) == expression


@pytest.mark.parametrize(
    "case", FIXTURE["cases"] + FIXTURE["readOnly"], ids=lambda case: case["expression"][:48],
)
def test_parts_are_what_validation_answers(case: dict[str, Any]) -> None:
    parts = [_part_json(part) for part in term_parts(parse_expr(case["expression"]))]
    assert parts == case["parts"]
    joined = " + ".join(part["text"] for part in parts).replace("+ -", "- ")
    assert joined == format_expr(parse_expr(case["expression"]))
