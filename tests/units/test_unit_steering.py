"""Unit steering: ``!neuron/…`` / ``!head/…`` mean-ablation and ``α neuron/…`` pushes.

An ablation must equal a hand-patched forward (the unit's activation replaced
by its neutral mean, blended by the coefficient) to 1e-5; a push must equal
the one kernel applied to the neuron's write direction registered as an
ordinary profile.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from drowse.model.units import UnitIndexError, neuron_write
from drowse.steering.triggers import Trigger
from tests.support.sessions import forward_logits, register_vectors
from tests.support.units import patched_forward_logits

TEXT = "a probe sentence for the units"


def _ids(session: Any) -> torch.Tensor:
    return session.tokenizer(TEXT, return_tensors="pt")["input_ids"]


def _blocks(session: Any) -> Any:
    return session.units.runtime.spec().blocks


def _stats(session: Any) -> Any:
    return session.units.runtime.stats()


def _neuron_patch(mean: float, n: int, c: float = 1.0) -> Any:
    def patch(x: torch.Tensor) -> torch.Tensor:
        x[..., n] = (1.0 - c) * x[..., n] + c * mean
        return x
    return patch


def _head_patch(mean: torch.Tensor, h: int, hd: int, c: float = 1.0) -> Any:
    def patch(x: torch.Tensor) -> torch.Tensor:
        span = slice(h * hd, (h + 1) * hd)
        x[..., span] = (1.0 - c) * x[..., span] + c * mean
        return x
    return patch


# ---------------------------------------------------------------- ablation


def test_a_full_neuron_ablation_equals_a_hand_patched_forward(tiny_session: Any) -> None:
    block = _blocks(tiny_session)[1]
    mean = float(_stats(tiny_session).layer(1).neuron_mean[3])
    expected = patched_forward_logits(
        tiny_session.model, _ids(tiny_session), {block.mlp.down: _neuron_patch(mean, 3)},
    )
    plain = forward_logits(tiny_session, TEXT)
    with tiny_session.steering("!neuron/L1n3"):
        ablated = forward_logits(tiny_session, TEXT)
    torch.testing.assert_close(ablated.float(), expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(ablated.float(), plain.float(), atol=1e-6)
    torch.testing.assert_close(forward_logits(tiny_session, TEXT), plain)


@pytest.mark.parametrize("c", [0.3, 2.0, -0.5])
def test_a_partial_neuron_ablation_blends_toward_the_mean(tiny_session: Any, c: float) -> None:
    block = _blocks(tiny_session)[2]
    mean = float(_stats(tiny_session).layer(2).neuron_mean[7])
    expected = patched_forward_logits(
        tiny_session.model, _ids(tiny_session), {block.mlp.down: _neuron_patch(mean, 7, c)},
    )
    with tiny_session.steering(f"{c:g} !neuron/L2n7" if c >= 0 else f"-{-c:g} !neuron/L2n7"):
        got = forward_logits(tiny_session, TEXT)
    torch.testing.assert_close(got.float(), expected, rtol=1e-5, atol=1e-5)


def test_a_head_ablation_replaces_its_z_slice(tiny_session: Any) -> None:
    block = _blocks(tiny_session)[0]
    hd = block.attn.head_dim
    mean = _stats(tiny_session).layer(0).head_mean[2]
    expected = patched_forward_logits(
        tiny_session.model, _ids(tiny_session), {block.attn.o: _head_patch(mean, 2, hd)},
    )
    with tiny_session.steering("!head/L0h2"):
        got = forward_logits(tiny_session, TEXT)
    torch.testing.assert_close(got.float(), expected, rtol=1e-5, atol=1e-5)


def test_units_on_one_module_share_one_hook_and_compose(tiny_session: Any) -> None:
    blocks = _blocks(tiny_session)
    stats = _stats(tiny_session)
    m3 = float(stats.layer(1).neuron_mean[3])
    m5 = float(stats.layer(1).neuron_mean[5])
    hd = blocks[3].attn.head_dim
    hm = stats.layer(3).head_mean[1]

    def both(x: torch.Tensor) -> torch.Tensor:
        return _neuron_patch(m5, 5, 0.5)(_neuron_patch(m3, 3)(x))

    expected = patched_forward_logits(
        tiny_session.model, _ids(tiny_session),
        {blocks[1].mlp.down: both, blocks[3].attn.o: _head_patch(hm, 1, hd)},
    )
    with tiny_session.steering("!neuron/L1n3 + 0.5 !neuron/L1n5 + !head/L3h1"):
        manager = tiny_session._steering
        assert len(manager.unit_hooks) == 2  # one per touched module
        assert not manager.hooks
        got = forward_logits(tiny_session, TEXT)
    torch.testing.assert_close(got.float(), expected, rtol=1e-5, atol=1e-5)
    assert not tiny_session._steering.unit_hooks


def test_unit_ablation_honors_its_trigger(tiny_session: Any) -> None:
    block = _blocks(tiny_session)[1]
    mean = float(_stats(tiny_session).layer(1).neuron_mean[3])
    ablated = patched_forward_logits(
        tiny_session.model, _ids(tiny_session), {block.mlp.down: _neuron_patch(mean, 3)},
    )
    plain = forward_logits(tiny_session, TEXT).float()
    ctx = tiny_session._steering.ctx
    with tiny_session.steering("!neuron/L1n3@response"):
        (hook,) = tiny_session._steering.unit_hooks.values()
        assert not hook.is_static
        ctx.is_prefill = True
        torch.testing.assert_close(forward_logits(tiny_session, TEXT).float(), plain)
        ctx.is_prefill = False
        torch.testing.assert_close(
            forward_logits(tiny_session, TEXT).float(), ablated, rtol=1e-5, atol=1e-5,
        )
    with tiny_session.steering("!neuron/L1n3@first:2"):
        ctx.is_prefill, ctx.gen_step = False, 1
        torch.testing.assert_close(
            forward_logits(tiny_session, TEXT).float(), ablated, rtol=1e-5, atol=1e-5,
        )
        ctx.gen_step = 2
        torch.testing.assert_close(forward_logits(tiny_session, TEXT).float(), plain)
    ctx.reset()


def test_a_gated_unit_ablation_follows_its_gate(tiny_session: Any) -> None:
    block = _blocks(tiny_session)[1]
    mean = float(_stats(tiny_session).layer(1).neuron_mean[3])
    ablated = patched_forward_logits(
        tiny_session.model, _ids(tiny_session), {block.mlp.down: _neuron_patch(mean, 3)},
    )
    ctx = tiny_session._steering.ctx
    with tiny_session.steering("!neuron/L1n3@when:head/L0h0>1"):
        ctx.probe_scores = {"head/L0h0": 2.0}
        torch.testing.assert_close(
            forward_logits(tiny_session, TEXT).float(), ablated, rtol=1e-5, atol=1e-5,
        )
        ctx.probe_scores = {"head/L0h0": 0.5}
        torch.testing.assert_close(
            forward_logits(tiny_session, TEXT).float(),
            forward_logits(tiny_session, TEXT).float(),
        )
    ctx.reset()


def test_unit_hooks_route_a_compiled_session_to_the_eager_module(tiny_session: Any) -> None:
    manager = tiny_session._steering
    assert manager.all_fast_path()
    with tiny_session.steering("!neuron/L1n3"):
        assert not manager.all_fast_path()
        assert manager.static_steerable()
        assert manager.plan is not None and manager.plan.static_offsets() is None
    with tiny_session.steering("!neuron/L1n3@response"):
        assert not manager.static_steerable()
    assert manager.all_fast_path()


def test_an_ablated_generation_runs_its_ops_and_leaves_no_hook(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.generation.sampling import SamplingConfig
    from drowse.steering.hooks import UnitOp

    calls: list[tuple[int, ...]] = []
    apply = UnitOp.apply

    def counting(self: UnitOp, x: torch.Tensor) -> None:
        calls.append(tuple(x.shape))
        apply(self, x)

    monkeypatch.setattr(UnitOp, "apply", counting)
    result = tiny_session.generate(
        "hello", stateless=True, sampling=SamplingConfig(max_tokens=4),
        steering="!neuron/L1n3 + !neuron/L2n9 + !head/L1h0",
    ).first
    assert result.applied_steering == "!neuron/L1n3 + !neuron/L2n9 + !head/L1h0"
    # Three modules (two down projections, one output projection), every
    # forward: the prefill over the whole prompt, then one position per step.
    forwards = len(result.tokens)
    assert len(calls) == 3 * forwards
    assert calls[0][1] > 1 and all(shape[1] == 1 for shape in calls[3:])
    assert not tiny_session._steering.unit_hooks
    for block in _blocks(tiny_session):
        assert not block.mlp.down._forward_pre_hooks
        assert not block.attn.o._forward_pre_hooks


# -------------------------------------------------------------------- push


def test_a_neuron_push_is_the_kernel_on_its_write_direction(tiny_session: Any) -> None:
    block = _blocks(tiny_session)[2]
    write = neuron_write(block, 11)
    tiny_session.registry.put_profile("local/w", {2: write.clone()})
    for alpha in ("0.7", "-0.7"):
        with tiny_session.steering(f"{alpha} neuron/L2n11"):
            assert set(tiny_session._steering.hooks) == {2}
            pushed = forward_logits(tiny_session, TEXT)
        with tiny_session.steering(f"{alpha} local/w"):
            reference = forward_logits(tiny_session, TEXT)
        torch.testing.assert_close(pushed, reference, rtol=1e-6, atol=1e-6)
    registered = tiny_session.profiles["neuron/L2n11"]
    assert set(registered) == {2}
    torch.testing.assert_close(registered[2].cpu(), write)


def test_the_two_poles_push_opposite_ways(tiny_session: Any) -> None:
    plain = forward_logits(tiny_session, TEXT).float()
    with tiny_session.steering("0.6 neuron/L1n4"):
        up = forward_logits(tiny_session, TEXT).float() - plain
    with tiny_session.steering("-0.6 neuron/L1n4"):
        down = forward_logits(tiny_session, TEXT).float() - plain
    assert torch.nn.functional.cosine_similarity(up, down, dim=0) < -0.5


def test_a_push_and_an_ablation_compose(tiny_session: Any) -> None:
    register_vectors(tiny_session, "v")
    with tiny_session.steering("0.3 v + !neuron/L1n3 - 0.2 neuron/L2n5@response"):
        manager = tiny_session._steering
        assert manager.unit_hooks and manager.hooks
        assert {g.trigger for h in manager.hooks.values() for g in h.groups} >= {
            Trigger.BOTH, Trigger.GENERATED_ONLY,
        }


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(("expr", "fragment"), [
    ("!neuron/L1n64", "neuron 64 is outside block 1 (64 neurons, 0..63)"),
    ("!head/L0h4", "head 4 is outside block 0 (4 heads, 0..3)"),
    ("!neuron/L4n0", "layer 4 is outside the model (4 blocks, 0..3)"),
    ("0.3 neuron/L1n99", "neuron 99 is outside block 1"),
])
def test_an_unknown_unit_is_a_400_before_the_scope(
    tiny_session: Any, expr: str, fragment: str,
) -> None:
    with pytest.raises(UnitIndexError) as info:
        tiny_session.steering(expr)
    status, text = info.value.user_message()
    assert status == 400 and fragment in text
    assert not tiny_session._steering.unit_hooks


def test_units_on_an_undescribed_model_are_a_422(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.units.runtime as runtime_mod
    from drowse.errors import UnsupportedBlockError

    def refuse(_model: Any) -> Any:
        raise UnsupportedBlockError("fused gate-up projection ('gate_up_proj')", layer=0)

    monkeypatch.setattr(runtime_mod, "model_spec", refuse)
    session = make_tiny_session()
    for call in (
        lambda: session.steering("!neuron/L1n3"),
        lambda: session.steering("0.3 neuron/L1n3"),
        lambda: session.add_probe("head/L1h0"),
    ):
        with pytest.raises(UnsupportedBlockError) as info:
            call()
        assert info.value.user_message()[0] == 422


def test_units_without_neutral_statistics_are_a_422(make_tiny_session: Any) -> None:
    from drowse.units.runtime import UnitsUnavailableError

    session = make_tiny_session()
    neutral = session._neutral
    neutral.adopt(dict(neutral.means), neutral.resident_whitener)  # drops the stats
    with pytest.raises(UnitsUnavailableError) as info:
        with session.steering("!neuron/L1n3"):
            pass
    status, text = info.value.user_message()
    assert status == 422 and "no unit statistics" in text
    with pytest.raises(UnitsUnavailableError):
        session.add_probe("neuron/L1n3")
