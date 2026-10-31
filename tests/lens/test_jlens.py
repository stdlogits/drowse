"""CPU tests for the Jacobian lens object and readout math: select and
union, transport, token directions, readout calibration, word→token.

The estimator's tests live in ``test_fit.py`` and ``test_lens_transport.py``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from drowse.lens.jlens import (
    JacobianLens,
    JacobianLensError,
    MultiTokenWordError,
    aggregate_readout,
    aggregate_readout_from_probabilities,
    aggregate_readout_tensors_from_probabilities,
    transport_rows,
    readout_probabilities,
    resolve_word_token,
    topk_probabilities,
    token_readout_stats,
    token_readout_stats_from_probabilities,
)
from drowse.model.readout import get_final_norm, get_unembedding
from tests.support.jlens_toys import TOY_D as _D
from tests.support.jlens_toys import TOY_VOCAB as _VOCAB
from tests.support.jlens_toys import CharTokenizer as _CharTokenizer
from tests.support.jlens_toys import frozen_toy as _frozen_model


def test_select_and_union_layers() -> None:
    j0 = torch.randn(_D, _D)
    j1 = torch.randn(_D, _D)
    j2 = torch.randn(_D, _D)
    lens = JacobianLens({0: j0, 1: j1}, n_prompts=3, d_model=_D)

    selected = lens.select_layers({"out": [1]})
    assert selected.source_layers == [1]
    assert torch.equal(selected.jacobians[1], lens.jacobians[1])

    union = JacobianLens.union_layers([
        selected,
        JacobianLens({2: j2}, n_prompts=3, d_model=_D),
    ])
    assert union.source_layers == [1, 2]


def test_identity_transport_readout_matches_model_logits() -> None:
    """With J_l = I at the last source layer... the readout must equal running
    the remaining blocks' *absence* — i.e. norm+unembed of the residual itself
    (the logit-lens identity check)."""
    model = _frozen_model(n_layers=2)
    ids = _CharTokenizer()("a prompt that is long enough.")["input_ids"]
    finals: list[torch.Tensor] = []
    handle = model.model.layers[-1].register_forward_hook(
        lambda _m, _a, output: finals.append(output[0] if isinstance(output, tuple) else output),
    )
    try:
        with torch.no_grad():
            expected = model(input_ids=ids).logits
    finally:
        handle.remove()
    h = finals[-1]

    from drowse.model.readout import LogitReadout

    lens = JacobianLens({1: torch.eye(_D)}, n_prompts=1, d_model=_D)
    rows = h.reshape(-1, _D)
    out = LogitReadout(cast(Any, model))(transport_rows(
        lens.jacobians[1].expand(rows.shape[0], _D, _D), rows,
    ))
    assert torch.allclose(out.float(), expected.squeeze(0).float(), atol=1e-5)


def test_transport_unknown_layer_raises() -> None:
    lens = JacobianLens({0: torch.eye(_D)}, n_prompts=1, d_model=_D)
    with pytest.raises(JacobianLensError):
        lens.transport(torch.zeros(_D), 3)


def test_accessors_on_toy_model() -> None:
    model = _frozen_model()
    w = get_unembedding(cast(Any, model))
    assert w.shape == (_VOCAB, _D)
    assert get_final_norm(cast(Any, model)) is model.model.norm


def test_token_direction_shape_and_math() -> None:
    J = torch.randn(2, _D, _D)
    lens = JacobianLens({0: J[0], 1: J[1]}, n_prompts=1, d_model=_D)
    unembed = torch.randn(_VOCAB, _D)
    dirs = lens.token_direction(5, unembed)
    assert set(dirs) == {0, 1}
    assert torch.allclose(dirs[1], unembed[5] @ J[1])


def test_token_direction_can_restrict_layers() -> None:
    J = torch.randn(2, _D, _D)
    lens = JacobianLens({0: J[0], 1: J[1]}, n_prompts=1, d_model=_D)
    unembed = torch.randn(_VOCAB, _D)
    dirs = lens.token_direction(5, unembed, layers=[1])
    assert set(dirs) == {1}
    assert torch.allclose(dirs[1], unembed[5] @ J[1])


def test_topk_probabilities_matches_full_softmax() -> None:
    """The selected-column shortcut agrees with a full softmax, and reports
    the same unit ``readout_probabilities`` does — the one lens unit."""
    logits = torch.randn(4, _VOCAB)
    vals, idxs = topk_probabilities(logits, 7)
    expected = readout_probabilities(logits)
    exp_vals, exp_idxs = expected.topk(7, dim=-1)
    assert torch.equal(idxs, exp_idxs)
    assert torch.allclose(vals, exp_vals)


def test_aggregate_readout_strength_is_mean_probability() -> None:
    logits = torch.randn(3, _VOCAB)
    depths = [0.4, 0.6, 0.8]
    rows = aggregate_readout(logits, depths, top_k=_VOCAB)
    probs = logits.softmax(dim=-1).mean(dim=0)
    got = {tok: s for tok, s, _, _ in rows}
    assert len(got) == _VOCAB
    for tok, s in got.items():
        assert s == pytest.approx(float(probs[tok]), abs=1e-6)
    # sorted by descending strength
    strengths = [s for _, s, _, _ in rows]
    assert strengths == sorted(strengths, reverse=True)


def test_aggregate_readout_reuses_calibrated_probabilities_exactly() -> None:
    logits = torch.randn(5, _VOCAB)
    depths = [0.41, 0.52, 0.63, 0.74, 0.85]
    expected = aggregate_readout(logits, depths, top_k=4)
    got = aggregate_readout_from_probabilities(
        readout_probabilities(logits), depths, top_k=4,
    )
    assert got == expected


def test_aggregate_readout_tensor_surface_matches_list_surface() -> None:
    logits = torch.randn(5, _VOCAB)
    probabilities = readout_probabilities(logits)
    depths = [0.41, 0.52, 0.63, 0.74, 0.85]
    expected = aggregate_readout_from_probabilities(
        probabilities, depths, top_k=4,
    )

    ids, stats = aggregate_readout_tensors_from_probabilities(
        probabilities, depths, top_k=4,
    )
    got = [
        (
            int(ids[j]),
            float(stats[0, j]),
            float(stats[1, j]),
            float(stats[2, j]),
        )
        for j in range(int(ids.numel()))
    ]

    assert got == expected


def test_aggregate_readout_avoids_float64_for_mps_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_to = torch.Tensor.to

    def reject_float64(
        self: torch.Tensor, *args: Any, **kwargs: Any,
    ) -> torch.Tensor:
        dtype = kwargs.get("dtype")
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if dtype is torch.float64:
            raise AssertionError("device-side float64 is unsupported on MPS")
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", reject_float64)
    probabilities = readout_probabilities(torch.randn(5, _VOCAB))
    rows = aggregate_readout_from_probabilities(
        probabilities, [0.41, 0.52, 0.63, 0.74, 0.85], top_k=4,
    )
    assert len(rows) == 4


def test_token_readout_stats_uses_exact_columns_without_full_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.jlens as jlens_module

    logits = torch.randn(
        4, _VOCAB, generator=torch.Generator().manual_seed(616),
    )
    depths = [0.41, 0.52, 0.63, 0.74]
    token_ids = [1, 4, 1]
    expected = token_readout_stats_from_probabilities(
        readout_probabilities(logits), depths, token_ids,
    )

    def _fail_full_probabilities(_logits: torch.Tensor) -> torch.Tensor:
        raise AssertionError("fixed-token stats should not full-softmax")

    monkeypatch.setattr(
        jlens_module, "readout_probabilities", _fail_full_probabilities,
    )

    got = token_readout_stats(logits, depths, token_ids)

    assert len(got) == len(expected)
    for got_row, expected_row in zip(got, expected):
        assert got_row[:3] == pytest.approx(expected_row[:3])
        assert got_row[3] == pytest.approx(expected_row[3])


def test_aggregate_readout_com_tracks_where_a_token_leads() -> None:
    # Token 0 dominates the earliest layer, token 1 the latest. Sharp
    # logits concentrate each layer's probability mass on its leader, so
    # each token's depth CoM sits at the layer(s) it leads.
    depths = [0.2, 0.5, 0.8]
    logits = torch.full((3, _VOCAB), -10.0)
    logits[0, 0] = 10.0
    logits[1, 3] = 10.0
    logits[2, 1] = 10.0
    rows = {tok: (com, spread) for tok, _, com, spread in
            aggregate_readout(logits, depths, top_k=_VOCAB)}
    com0, _ = rows[0]
    com1, _ = rows[1]
    assert com0 < 0.3          # leads only the early layer
    assert com1 > 0.7          # leads only the late layer
    assert com0 < rows[3][0] < com1  # mid-layer leader sits between


def test_aggregate_readout_single_layer_degenerates() -> None:
    logits = torch.randn(1, _VOCAB)
    rows = aggregate_readout(logits, [0.55], top_k=4)
    assert len(rows) == 4
    for _, strength, com, spread in rows:
        assert 0.0 <= strength <= 1.0
        assert com == pytest.approx(0.55, abs=1e-6)
        assert spread == pytest.approx(0.0, abs=1e-6)


def test_aggregate_readout_com_is_probability_mass_weighted() -> None:
    # The design point (2026-07-09): the band readout is sharp, and what
    # changes over depth is WHICH token leads — so depth CoM weights by
    # the per-layer probability itself, the same channel behind strength.
    # A fully-diffuse layer (uniform readout — nothing is being "said")
    # must not pin any token's depth: its vote is discounted by its own
    # lack of mass. The former within-layer salience handed the uniform
    # layer a FULL vote for every token (sal = 1.0 at the tied max),
    # reading token 0 early (com ≈ 0.33 here) even though its readout
    # probability concentrates late. (At the toy vocab of 13 the uniform
    # layer still carries p = 1/13 per token; at a real ~260k vocab the
    # discount is ~4e-6 and the pull toward the sharp layer is total.)
    depths = [0.3, 0.9]
    logits = torch.zeros(2, _VOCAB)   # layer 0 uniform (fully diffuse)
    logits[1, 1] = 12.0               # sharp late layer led by token 1
    logits[1, 0] = 9.0                # token 0 present but not leading
    p = logits.softmax(dim=-1)
    d = torch.tensor(depths)
    rows = {tok: com for tok, _, com, _ in
            aggregate_readout(logits, depths, top_k=_VOCAB)}
    for tok in (0, 1):
        expected = float((p[:, tok] * d).sum() / p[:, tok].sum())
        assert rows[tok] == pytest.approx(expected, abs=1e-6)
    # token 0's depth is pulled past the uniform layer toward its mass
    assert rows[0] > 0.5
    assert rows[1] > 0.8


def test_aggregate_readout_validates_shapes() -> None:
    with pytest.raises(ValueError):
        aggregate_readout(torch.randn(_VOCAB), [0.5], top_k=3)
    with pytest.raises(ValueError):
        aggregate_readout(torch.randn(2, _VOCAB), [0.5], top_k=3)
    with pytest.raises(ValueError):
        aggregate_readout(torch.randn(0, _VOCAB), [], top_k=3)


class _WordTokenizer:
    """encode/decode stub: vocabulary maps ' fake'->1, 'fake'->2, multi otherwise."""

    def __init__(self, single: dict[str, int], decode_map: dict[int, str]) -> None:
        self.single = single
        self.decode_map = decode_map

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.single:
            return [self.single[text]]
        return [3, 4]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.decode_map.get(i, "?") for i in ids)


def test_resolve_word_token_prefers_leading_space() -> None:
    tok = _WordTokenizer({" fake": 1, "fake": 2}, {1: " fake", 2: "fake"})
    assert resolve_word_token(tok, "fake") == 1


def test_resolve_word_token_falls_back_to_bare() -> None:
    tok = _WordTokenizer({"fake": 2}, {2: "fake"})
    assert resolve_word_token(tok, "fake") == 2


def test_resolve_word_token_multi_token_raises() -> None:
    tok = _WordTokenizer({}, {3: "fa", 4: "ke"})
    with pytest.raises(MultiTokenWordError, match="pieces"):
        resolve_word_token(tok, "fake")


def test_resolve_word_token_decode_mismatch_raises() -> None:
    # single-token hit whose decode is a different word: must NOT be accepted
    tok = _WordTokenizer({" fake": 1, "fake": 2}, {1: " fakes", 2: "faked"})
    with pytest.raises(MultiTokenWordError):
        resolve_word_token(tok, "fake")


# --------------------------------------------------------------------------
# module split: readout stays import-light, the estimator loads on demand
# --------------------------------------------------------------------------

def test_readout_module_does_not_import_the_estimator() -> None:
    """``drowse.lens.jlens`` is imported per decode step by the lens
    instrument, so it must not drag in the backward-pass estimator."""
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable, "-c",
            "import drowse.lens.jlens, sys;"
            "print('drowse.lens.fit' in sys.modules)",
        ],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"
