"""The one layer-selector parser and its two resolvers (``drowse.model.layers``)."""
from __future__ import annotations

import pytest

from drowse.model.layers import (
    LayerSelectorError,
    format_layer_indices,
    parse_layer_selector,
    resolve_layer_selector,
    resolve_model_layers,
)


@pytest.mark.parametrize(("text", "expected"), [
    (None, None),
    ("  ", None),
    ("3,7, 11", [3, 7, 11]),
    ("Workspace", "workspace"),
    ("all", "all"),
    ("sample", "sample"),
    ("4-8,20", [4, 5, 6, 7, 8, 20]),
    ("20, 4 - 6", [20, 4, 5, 6]),
    ("3-3", [3]),
])
def test_parse_layer_selector(text: str | None, expected: object) -> None:
    assert parse_layer_selector(text) == expected


@pytest.mark.parametrize(("text", "message", "offset"), [
    ("3,x", "malformed layers", 2),
    (",,", "empty entry", 0),
    ("1, ,2", "empty entry", 3),
    ("8-4", "reversed layer range '8-4'", 0),
    ("4-8, 6", "layer 6 is named twice", 5),
    ("2,2", "layer 2 is named twice", 2),
    ("-1", "non-negative", 0),
    ("1,-3", "non-negative", 2),
    ("1-", "malformed layers", 0),
])
def test_parse_layer_selector_rejects_malformed_text(
    text: str, message: str, offset: int,
) -> None:
    with pytest.raises(LayerSelectorError, match=message) as exc:
        parse_layer_selector(text)
    assert exc.value.offset == offset
    assert exc.value.user_message()[0] == 400


def test_parse_layer_selector_can_refuse_sample() -> None:
    assert parse_layer_selector("sample") == "sample"
    with pytest.raises(LayerSelectorError, match="unknown layer mode") as exc:
        parse_layer_selector("  sample", allow_sample=False)
    assert exc.value.offset == 2


@pytest.mark.parametrize(("indices", "text"), [
    ([20, 4, 5, 6, 7, 8], "4-8,20"),
    ([8, 10, 12], "8,10,12"),
    ([4, 5], "4-5"),
    ([3], "3"),
    ([1, 1, 2], "1-2"),
])
def test_format_layer_indices_is_the_canonical_text(
    indices: list[int], text: str,
) -> None:
    assert format_layer_indices(indices) == text
    assert parse_layer_selector(text) == sorted(set(indices))


def test_resolve_layer_selector_draws_from_the_available_pool() -> None:
    fitted = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    assert resolve_layer_selector(None, fitted, n_layers=41) == fitted
    assert resolve_layer_selector("workspace", fitted, n_layers=41) == [16, 20, 24, 28, 32, 36]
    assert len(resolve_layer_selector("sample", fitted, n_layers=41)) == 9
    assert resolve_layer_selector([8, 4], fitted, n_layers=41) == [8, 4]
    with pytest.raises(ValueError, match="unknown layer mode"):
        resolve_layer_selector("sample", fitted, n_layers=41, allow_sample=False)


@pytest.mark.parametrize(("selector", "expected"), [
    (None, list(range(10))),
    ("all", list(range(10))),
    ("workspace", [4, 5, 6, 7, 8]),
    ("WORKSPACE", [4, 5, 6, 7, 8]),
    ("7, 3", [3, 7]),
    ("7,1-3", [1, 2, 3, 7]),
    ([9, 2, 2], [2, 9]),
])
def test_resolve_model_layers_is_the_sorted_unique_fit_roster(
    selector: object, expected: list[int],
) -> None:
    assert resolve_model_layers(selector, 10) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(("selector", "message"), [
    ("", "at least one layer"),
    ([], "at least one layer"),
    ("sample", "unknown layer mode"),
    ("3,x", "malformed layers"),
    ("7, 3,3", "named twice"),
    ("8-12", r"\[0, 10\)"),
    ([10], r"\[0, 10\)"),
    ([-1, 2], r"\[0, 10\)"),
])
def test_resolve_model_layers_rejects(selector: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_model_layers(selector, 10)  # type: ignore[arg-type]


def test_fit_and_preflight_share_the_resolver() -> None:
    """The fit and its weight-free preflight must resolve the same roster."""
    from drowse.fit import pipeline, preflight

    assert pipeline.resolve_model_layers is resolve_model_layers
    assert preflight.resolve_model_layers is resolve_model_layers
