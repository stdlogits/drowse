"""``fit_lens``: transport over a corpus, against the autograd oracle.

The toy lens model (a genuine tiny llama, ``tests/support/jlens_toys``) is
fitted directly, without the lifecycle: every kind at both sites equals the
oracle's equal-weight mean over the same documents; the split-half check
matches its definition on an odd document count; resume continues the sums
(and the halves) exactly; and the fit's guards (short documents, the OOM
ladder, an accelerator's zero rows, cancellation, unsupported blocks) hold.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import drowse.lens.fit as fit_module
from drowse.errors import RequestError, UnsupportedBlockError, is_out_of_memory_error
from drowse.lens.fit import LensSums, fit_lens, fit_site_layers
from drowse.lens.jlens import JacobianLensCancelled, JacobianLensError
from drowse.lens.key import FITTABLE_KINDS, site_layers
from drowse.lens.transport import document_estimate
from drowse.model.blocks import model_spec
from tests.support.blocks import tiny_dense_model
from tests.support.jlens_toys import TOY_D, CharTokenizer, frozen_toy
from tests.support.lens import LENS_PROMPTS
from tests.support.lens_oracle import oracle_lens

_TOK = CharTokenizer()
#: Five documents of ragged length (an odd count for the split-half check).
_DOCS = [*LENS_PROMPTS, "lmkjihgfedcbamlkjihgfedcbalm"]
_ROWS = [_TOK.encode(doc) for doc in _DOCS]


def _layers(kind: str, n: int = 3) -> dict[str, list[int]]:
    return {site: site_layers(kind, site, n) for site in ("out", "mid")}


def _assert_close(lens: Any, ref: dict[str, dict[int, torch.Tensor]], *, rtol: float = 1e-5) -> None:
    for site, matrices in ref.items():
        assert lens.layers(site) == sorted(matrices)
        for layer, want in matrices.items():
            got = lens.matrix(layer, site)
            err = float((got.double() - want.double()).norm() / want.double().norm())
            assert err < rtol, (site, layer, err)


def test_every_kind_is_fittable_and_matches_the_oracle() -> None:
    assert FITTABLE_KINDS == ("J", "R", "K", "S")
    model = frozen_toy()
    for kind in FITTABLE_KINDS:
        fitted = fit_lens(model, _TOK, _DOCS, kind=kind, dim_batch=4)
        assert fitted.lens.key.kind == kind and fitted.lens.n_prompts == len(_DOCS)
        assert fitted.lens.sites == ("out", "mid")
        ref = oracle_lens(model, _ROWS, kind=kind, layers=_layers(kind), skip_first=16)
        _assert_close(fitted.lens, ref)


def test_documents_weigh_equally_at_ragged_lengths() -> None:
    model = frozen_toy()
    joint = fit_lens(model, _TOK, _DOCS[:2], kind="R", sites=["out"]).lens
    solo = [fit_lens(model, _TOK, [doc], kind="R", sites=["out"]).lens for doc in _DOCS[:2]]
    for layer in joint.source_layers:
        mean = (solo[0].jacobians[layer] + solo[1].jacobians[layer]) / 2
        assert torch.allclose(joint.jacobians[layer], mean, atol=1e-6)


def test_split_half_is_the_even_odd_difference_on_an_odd_count() -> None:
    model = frozen_toy()
    spec = model_spec(model)
    fitted = fit_lens(model, _TOK, _DOCS, kind="S", layers={"out": [0, 2], "mid": [1]})
    estimates = [
        document_estimate(
            spec, torch.tensor(row), kind="S", layers={"out": [0, 2], "mid": [1]},
            skip_first=16,
        )
        for row in _ROWS
    ]
    assert set(fitted.split_half) == {"out", "mid"}
    for site, layers in (("out", [0, 2]), ("mid", [1])):
        assert set(fitted.split_half[site]) == set(layers)
        for layer in layers:
            per_doc = [est[site][layer].double() for est in estimates]
            even = torch.stack(per_doc[0::2]).mean(0)  # documents 0, 2, 4
            odd = torch.stack(per_doc[1::2]).mean(0)  # documents 1, 3
            mean = torch.stack(per_doc).mean(0)
            want = float((even - odd).norm() / mean.norm())
            assert math.isclose(fitted.split_half[site][layer], want, rel_tol=1e-4)


def test_split_half_needs_both_halves_and_can_be_off() -> None:
    model = frozen_toy()
    assert fit_lens(model, _TOK, _DOCS[:1], kind="R").split_half == {}
    assert fit_lens(model, _TOK, _DOCS, kind="R", split_half=False).split_half == {}


def test_resume_continues_the_sums_and_the_halves_exactly() -> None:
    model = frozen_toy()
    full = fit_lens(model, _TOK, _DOCS, kind="R", dim_batch=4)
    saved: list[LensSums] = []

    def keep(sums: LensSums) -> None:
        saved.append(LensSums(
            sums.kind, sums.d_model, sums.n_prompts,
            {s: {l: m.clone() for l, m in ms.items()} for s, ms in sums.total.items()},
            None if sums.odd is None else {
                s: {l: m.clone() for l, m in ms.items()} for s, ms in sums.odd.items()
            },
        ))

    fit_lens(model, _TOK, _DOCS[:3], kind="R", dim_batch=4, on_checkpoint=keep, checkpoint_every=3)
    assert [sums.n_prompts for sums in saved] == [3]
    resumed = fit_lens(
        model, _TOK, _DOCS[3:], kind="R", dim_batch=4, initial=saved[0], progress_base=3,
    )
    assert resumed.lens.n_prompts == len(_DOCS)
    for site in ("out", "mid"):
        for layer in full.lens.layers(site):
            assert torch.allclose(
                resumed.lens.matrix(layer, site), full.lens.matrix(layer, site), atol=1e-6,
            )
            assert math.isclose(
                resumed.split_half[site][layer], full.split_half[site][layer], rel_tol=1e-5,
            )


def test_a_base_without_halves_resumes_without_a_split_half() -> None:
    model = frozen_toy()
    head = fit_lens(model, _TOK, _DOCS[:2], kind="J", sites=["out"]).lens
    base = LensSums.from_means(head)  # a durable artifact keeps no odd half
    assert base.odd is None and base.n_prompts == 2
    tail = fit_lens(model, _TOK, _DOCS[2:], kind="J", sites=["out"], initial=base)
    assert tail.split_half == {} and tail.lens.n_prompts == len(_DOCS)


def test_checkpoints_fire_at_the_cadence_and_hold_raw_sums() -> None:
    model = frozen_toy()
    seen: list[tuple[int, torch.Tensor]] = []
    fit_lens(
        model, _TOK, _DOCS, kind="K", sites=["mid"], checkpoint_every=2,
        on_checkpoint=lambda sums: seen.append((sums.n_prompts, sums.total["mid"][1].clone())),
    )
    assert [n for n, _ in seen] == [2, 4]
    two = fit_lens(model, _TOK, _DOCS[:2], kind="K", sites=["mid"]).lens
    assert torch.allclose(seen[0][1] / 2, two.matrix(1, "mid"), atol=1e-6)


def test_short_prompts_are_dropped_and_all_short_raises() -> None:
    model = frozen_toy()
    fitted = fit_lens(model, _TOK, ["tiny", *_DOCS[:2]], kind="R", sites=["out"])
    assert fitted.lens.n_prompts == 2
    with pytest.raises(JacobianLensError, match="no usable prompts"):
        fit_lens(model, _TOK, ["tiny", "short"], kind="R")


def test_pretokenized_rows_skip_the_tokenizer() -> None:
    class Refusing:
        def __call__(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("the tokenizer must not run")

    model = frozen_toy()
    by_ids = fit_lens(model, Refusing(), _DOCS[:2], kind="S", input_id_rows=_ROWS[:2]).lens
    by_text = fit_lens(model, _TOK, _DOCS[:2], kind="S").lens
    assert torch.equal(by_ids.matrix(0, "mid"), by_text.matrix(0, "mid"))
    with pytest.raises(ValueError, match="aligned"):
        fit_lens(model, Refusing(), _DOCS[:2], kind="S", input_id_rows=_ROWS[:1])


def test_out_of_memory_halves_the_pass_and_keeps_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = frozen_toy()
    reference = fit_lens(model, _TOK, _DOCS[:2], kind="R", dim_batch=4).lens
    real_pass = fit_module._pass
    widths: list[int] = []

    def flaky(*args: Any, **kwargs: Any) -> Any:
        widths.append(args[4])
        if len(widths) == 1:
            raise RuntimeError("MPS backend out of memory (simulated)")
        return real_pass(*args, **kwargs)

    monkeypatch.setattr(fit_module, "_pass", flaky)
    messages: list[str] = []
    fitted = fit_lens(model, _TOK, _DOCS[:2], kind="R", dim_batch=4, on_progress=messages.append)
    assert widths[:3] == [4, 2, 2]
    assert "dim_batch=2" in messages[-1]
    for site in ("out", "mid"):
        for layer in reference.layers(site):
            assert torch.allclose(fitted.lens.matrix(layer, site), reference.matrix(layer, site), atol=1e-6)


def test_other_errors_and_a_width_of_one_still_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    model = frozen_toy()

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(fit_module, "_pass", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        fit_lens(model, _TOK, _DOCS[:1], kind="R")

    def oom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(fit_module, "_pass", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        fit_lens(model, _TOK, _DOCS[:1], kind="R", dim_batch=1)


def test_zero_rows_from_an_accelerator_read_as_out_of_memory() -> None:
    good = torch.ones(2, 3, TOY_D)
    fit_module._refuse_zero_rows(good)
    lost = good.clone()
    lost[1, 2] = 0
    with pytest.raises(JacobianLensError) as info:
        fit_module._refuse_zero_rows(lost)
    assert is_out_of_memory_error(info.value)


class _CountingEvent:
    """``is_set`` turns true on its ``fire_at``-th call."""

    def __init__(self, fire_at: int) -> None:
        self.calls = 0
        self.fire_at = fire_at

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls >= self.fire_at


def test_a_stop_between_documents_checkpoints_the_completed_prefix() -> None:
    model = frozen_toy()
    saved: list[int] = []
    # Calls: before doc 0, after doc 0, before doc 1 (fires).
    with pytest.raises(JacobianLensCancelled, match="after 1 prompts"):
        fit_lens(
            model, _TOK, _DOCS, kind="R", cancel_event=_CountingEvent(3),
            on_checkpoint=lambda sums: saved.append(sums.n_prompts),
        )
    assert saved == [1]


def test_a_stop_inside_a_document_abandons_it_without_a_checkpoint() -> None:
    model = frozen_toy()
    saved: list[int] = []
    # dim_batch=2 over d=6: three passes; calls: start, after pass 1, after pass 2 (fires).
    with pytest.raises(JacobianLensCancelled, match="inside a document"):
        fit_lens(
            model, _TOK, _DOCS, kind="S", dim_batch=2, cancel_event=_CountingEvent(3),
            on_checkpoint=lambda sums: saved.append(sums.n_prompts),
        )
    assert saved == []


@pytest.mark.parametrize(("model_type", "overrides", "fragment"), [
    ("qwen3_moe", {"moe_intermediate_size": 16, "num_experts": 4, "num_experts_per_tok": 2},
     "mixture-of-experts"),
    ("glm", {}, "fused gate-up"),
    ("qwen3_5_text", dict(
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
    ), "linear-attention"),
])
def test_unsupported_blocks_refuse_before_any_work(
    model_type: str, overrides: dict[str, Any], fragment: str,
) -> None:
    class Refusing:
        def __call__(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("no work before the refusal")

    model = tiny_dense_model(model_type, **overrides)
    with pytest.raises(UnsupportedBlockError, match=fragment) as info:
        fit_lens(model, Refusing(), _DOCS, kind="R")
    assert info.value.user_message()[0] == 422


def test_site_layers_are_validated_per_site() -> None:
    assert fit_site_layers("R", 4) == {"out": [0, 1, 2], "mid": [0, 1, 2, 3]}
    assert fit_site_layers("S", 4, sites=["mid"]) == {"mid": [0, 1, 2, 3]}
    assert fit_site_layers("J", 4, layers={"mid": [3, 1]}) == {"mid": [1, 3]}
    with pytest.raises(RequestError, match="output kinds' target"):
        fit_site_layers("J", 4, layers={"out": [3]})
    with pytest.raises(RequestError, match="names no layers"):
        fit_site_layers("K", 4, layers={"out": []})
    with pytest.raises(RequestError, match="sites"):
        fit_site_layers("K", 4, sites=["out", "out"])
    with pytest.raises(JacobianLensError, match="fittable"):
        fit_site_layers("Q", 4)


def test_a_resume_base_must_match_the_request() -> None:
    model = frozen_toy()
    base = fit_lens(model, _TOK, _DOCS[:1], kind="R", sites=["out"]).lens
    with pytest.raises(ValueError, match="sites and layers"):
        fit_lens(model, _TOK, _DOCS[1:], kind="R", initial=LensSums.from_means(base))
    other = fit_lens(model, _TOK, _DOCS[:1], kind="J", sites=["out"]).lens
    with pytest.raises(ValueError, match="J lens"):
        fit_lens(model, _TOK, _DOCS[1:], kind="R", sites=["out"], initial=LensSums.from_means(other))


def test_fit_leaves_the_model_untouched() -> None:
    model = frozen_toy()
    before = {name: p.clone() for name, p in model.named_parameters()}
    fit_lens(model, _TOK, _DOCS[:1], kind="S")
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
    assert all("forward" not in module.__dict__ for module in model.modules())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
