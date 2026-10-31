"""Unit directions against hand computation on the tiny dense models."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from drowse.model.arch import get_layers
from drowse.model.blocks import model_spec
from drowse.model.units import (
    UnitIndexError,
    head_read_rows,
    head_write_matrix,
    neuron_read,
    neuron_read_rows,
    neuron_reads,
    neuron_write,
    neuron_writes,
)
from tests.support.blocks import tiny_dense_model

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float()


@pytest.mark.parametrize("family", ["llama", "gemma2", "gemma4_text"])
def test_neuron_write_by_hand(family: str) -> None:
    model = tiny_dense_model(family)
    layer: Any = get_layers(model)[1]
    spec = model_spec(model).blocks[1]
    n = 5
    expected = _f32(layer.mlp.down_proj.weight[:, n])
    if family == "gemma2":  # 1 + w post-feedforward norm, no scalar
        expected = expected * (1.0 + _f32(layer.post_feedforward_layernorm.weight))
    if family == "gemma4_text":  # plain post-feedforward norm and the layer scalar
        expected = (
            float(layer.layer_scalar) * _f32(layer.post_feedforward_layernorm.weight) * expected
        )
    write = neuron_write(spec, n)
    assert write.dtype == torch.float32 and write.device.type == "cpu"
    torch.testing.assert_close(write, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(neuron_writes(spec)[:, n], write)


def test_llama_neuron_write_is_the_residual_change_per_unit_activation() -> None:
    """On a block without output norm or scalar, the write is exact, not linearized."""
    model = tiny_dense_model("llama")
    layer: Any = get_layers(model)[0]
    spec = model_spec(model).blocks[0]
    ids = torch.tensor([[1, 5, 9, 4]])
    outs: list[torch.Tensor] = []
    bump = 0.0

    def nudge(_m: Any, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        x = args[0].clone()
        x[..., 7] += bump
        return (x,)

    handle = layer.mlp.down_proj.register_forward_pre_hook(nudge)
    capture = layer.register_forward_hook(
        lambda _m, _a, out: outs.append((out[0] if isinstance(out, tuple) else out)[0, -1].clone()),
    )
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
            bump = 0.25
            model(ids, use_cache=False)
    finally:
        handle.remove()
        capture.remove()
    torch.testing.assert_close(
        (outs[1] - outs[0]).float(), 0.25 * neuron_write(spec, 7), rtol=1e-5, atol=1e-6,
    )


@pytest.mark.parametrize("family", ["qwen3", "gemma3_text"])
def test_neuron_read_by_hand(family: str) -> None:
    model = tiny_dense_model(family)
    layer: Any = get_layers(model)[2]
    spec = model_spec(model).blocks[2]
    norm = layer.pre_feedforward_layernorm if family.startswith("gemma") else layer.post_attention_layernorm
    gamma = _f32(norm.weight) + (1.0 if family == "gemma3_text" else 0.0)
    n = 11
    a = F.normalize(_f32(layer.mlp.gate_proj.weight[n]) * gamma, dim=0)
    b = F.normalize(_f32(layer.mlp.up_proj.weight[n]) * gamma, dim=0)
    torch.testing.assert_close(neuron_read(spec, n, "+"), a + b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(neuron_read(spec, n, "-"), a - b, rtol=1e-5, atol=1e-6)
    rows_a, rows_b = neuron_read_rows(spec)
    torch.testing.assert_close(rows_a[n], a, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rows_b[n], b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(neuron_reads(spec, "-")[n], a - b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["qwen2", "gemma2", "gemma4_text"])
def test_head_write_and_reads_by_hand(family: str) -> None:
    model = tiny_dense_model(family)
    layer_index = 1
    layer: Any = get_layers(model)[layer_index]
    spec = model_spec(model).blocks[layer_index]
    attn = layer.self_attn
    hd = spec.attn.head_dim
    h = 3
    kv = h // (spec.attn.n_heads // spec.attn.n_kv_heads)
    write = _f32(attn.o_proj.weight[:, h * hd:(h + 1) * hd])
    if family == "gemma2":
        write = (1.0 + _f32(layer.post_attention_layernorm.weight))[:, None] * write
    if family == "gemma4_text":  # the scalar applies at the block output, not at mid
        write = _f32(layer.post_attention_layernorm.weight)[:, None] * write
    torch.testing.assert_close(head_write_matrix(spec, h), write, rtol=1e-5, atol=1e-6)

    gamma = _f32(layer.input_layernorm.weight) + (1.0 if family == "gemma2" else 0.0)
    reads = head_read_rows(spec, h)
    torch.testing.assert_close(reads.query, _f32(attn.q_proj.weight[h * hd:(h + 1) * hd]) * gamma)
    key = _f32(attn.k_proj.weight[kv * hd:(kv + 1) * hd]) * gamma
    torch.testing.assert_close(reads.key, key)
    if attn.v_proj is None:  # gemma4's full layers read keys as values
        torch.testing.assert_close(reads.value, key)
    else:
        torch.testing.assert_close(reads.value, _f32(attn.v_proj.weight[kv * hd:(kv + 1) * hd]) * gamma)
    assert reads.query.shape == (hd, spec.d_model)


def test_head_write_reproduces_the_attention_update() -> None:
    """Without a sandwich norm, the attention update is the sum of the heads' writes."""
    model = tiny_dense_model("qwen3")
    layer: Any = get_layers(model)[0]
    spec = model_spec(model).blocks[0]
    ids = torch.tensor([[1, 5, 9, 4, 3]])
    seen: dict[str, torch.Tensor] = {}
    pre = layer.self_attn.o_proj.register_forward_pre_hook(
        lambda _m, args: seen.__setitem__("z", args[0][0, -1].clone()),
    )
    post = layer.self_attn.register_forward_hook(
        lambda _m, _a, out: seen.__setitem__("o", out[0][0, -1].clone()),
    )
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        pre.remove()
        post.remove()
    hd = spec.attn.head_dim
    total = sum(
        head_write_matrix(spec, h) @ seen["z"][h * hd:(h + 1) * hd].float()
        for h in range(spec.attn.n_heads)
    )
    torch.testing.assert_close(total, seen["o"].float(), rtol=1e-5, atol=1e-6)


def test_indices_and_poles_are_checked() -> None:
    spec = model_spec(tiny_dense_model("llama")).blocks[2]
    with pytest.raises(UnitIndexError, match="neuron 48 is outside block 2"):
        neuron_write(spec, 48)
    with pytest.raises(UnitIndexError, match="neuron -1"):
        neuron_read(spec, -1, "+")
    with pytest.raises(UnitIndexError, match="head 4 is outside block 2"):
        head_write_matrix(spec, 4)
    with pytest.raises(UnitIndexError):
        head_read_rows(spec, 4)
    with pytest.raises(ValueError, match="pole"):
        neuron_read(spec, 0, "x")  # type: ignore[arg-type]
    assert UnitIndexError("x").user_message()[0] == 400
