"""BlockSpec discovery: every covered dense family described exactly, the rest refused."""

from __future__ import annotations

import copy
import dataclasses
import gc
import math
import weakref
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from drowse.errors import UnsupportedBlockError
from drowse.model.arch import get_layers
from drowse.model.blocks import (
    activation,
    activation_derivative,
    activation_factor,
    block_specs,
    model_spec,
    verify_model_spec,
)
from tests.support.blocks import (
    DENSE_FAMILIES,
    default_init_model,
    spec_forward,
    tiny_dense_model,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_IDS = torch.tensor([1, 7, 11, 13, 19, 23, 29, 31])


@pytest.fixture(scope="module", params=sorted(DENSE_FAMILIES))
def dense(request: pytest.FixtureRequest) -> tuple[str, Any]:
    return request.param, tiny_dense_model(request.param)


def _model_states(model: Any) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    """``(mids, outs, logits)`` from the model's own forward."""
    spec = model_spec(model)
    mids: dict[int, torch.Tensor] = {}
    outs: dict[int, torch.Tensor] = {}
    handles = []
    for block in spec.blocks:

        def take_mid(_m: Any, args: Any, i: int = block.layer) -> None:
            mids[i] = args[0][0].clone()

        def take_out(_m: Any, _a: Any, out: Any, i: int = block.layer) -> None:
            outs[i] = (out[0] if isinstance(out, tuple) else out)[0].clone()

        handles.append(block.mlp_in_norm.module.register_forward_pre_hook(take_mid))
        handles.append(block.module.register_forward_hook(take_out))
    try:
        with torch.no_grad():
            logits = model(_IDS[None], use_cache=False).logits[0]
    finally:
        for handle in handles:
            handle.remove()
    return mids, outs, logits


def test_spec_forward_reproduces_the_model(dense: tuple[str, Any]) -> None:
    """The description alone recomputes every mid, every block output and the logits."""
    _family, model = dense
    spec = model_spec(model)
    mids, outs, logits = _model_states(model)
    ref_mids, ref_outs, ref_logits = spec_forward(spec, _IDS)
    for block in spec.blocks:
        torch.testing.assert_close(ref_mids[block.layer], mids[block.layer], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ref_outs[block.layer], outs[block.layer], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ref_logits, logits, atol=1e-4, rtol=1e-4)


def test_spec_forward_needs_every_term(dense: tuple[str, Any]) -> None:
    """Negative control: dropping one block's MLP input gain moves the output."""
    _family, model = dense
    spec = model_spec(model)
    _mids, outs, _logits = _model_states(model)
    norm = spec.blocks[0].mlp_in_norm
    weight = getattr(norm.module, "weight")
    with torch.no_grad():
        weight.mul_(1.5)
    try:
        _ref_mids, ref_outs, _ = spec_forward(spec, _IDS)
    finally:
        with torch.no_grad():
            weight.div_(1.5)
    assert float((ref_outs[0] - outs[0]).abs().max()) > 1e-2


def test_verify_model_spec_accepts_every_covered_family(dense: tuple[str, Any]) -> None:
    _family, model = dense
    spec = verify_model_spec(model)
    assert len(spec.blocks) == len(get_layers(model))


_EXPECTED: dict[str, dict[str, Any]] = {
    "llama": dict(sandwich=False, offset=0.0, act="silu", qk_norm=False, qkv_bias=True,
                  windows=[None, None, None], tables=[None] * 3, scalar=False),
    "mistral": dict(sandwich=False, offset=0.0, act="silu", qk_norm=False, qkv_bias=False,
                    windows=[3, 3, 3], tables=[None] * 3, scalar=False),
    "qwen2": dict(sandwich=False, offset=0.0, act="silu", qk_norm=False, qkv_bias=True,
                  windows=[None, 3, 3], tables=[None] * 3, scalar=False),
    "qwen3": dict(sandwich=False, offset=0.0, act="silu", qk_norm=True, qkv_bias=False,
                  windows=[None, None, None], tables=[None] * 3, scalar=False),
    "gemma2": dict(sandwich=True, offset=1.0, act="gelu_tanh", qk_norm=False, qkv_bias=False,
                   windows=[3, None, 3], tables=[None] * 3, scalar=False),
    "gemma3_text": dict(sandwich=True, offset=1.0, act="gelu_tanh", qk_norm=True, qkv_bias=False,
                        windows=[3, None, 3],
                        tables=["sliding_attention", "full_attention", "sliding_attention"],
                        scalar=False),
    "gemma4_text": dict(sandwich=True, offset=0.0, act="gelu_tanh", qk_norm=True, qkv_bias=False,
                        windows=[3, None, None],
                        tables=["sliding_attention", "full_attention", "full_attention"],
                        scalar=True),
    "gemma4_unified_text": dict(sandwich=True, offset=0.0, act="gelu_tanh", qk_norm=True,
                                qkv_bias=False, windows=[3, None, None],
                                tables=["sliding_attention", "full_attention", "full_attention"],
                                scalar=True),
}


def test_block_fields_name_each_family_terms(dense: tuple[str, Any]) -> None:
    family, model = dense
    want = _EXPECTED[family]
    spec = model_spec(model)
    blocks = block_specs(model)
    assert [b.layer for b in blocks] == [0, 1, 2]
    assert [b.module for b in blocks] == list(get_layers(model))
    for block in blocks:
        assert (block.attn_out_norm is not None) is want["sandwich"]
        assert (block.mlp_out_norm is not None) is want["sandwich"]
        assert block.attn_in_norm.offset == want["offset"]
        assert block.mlp.act == want["act"]
        assert (block.attn.q_norm is not None) is want["qk_norm"]
        assert (block.attn.k_norm is not None) is want["qk_norm"]
        assert block.attn.qkv_bias is want["qkv_bias"]
        # llama's attention_bias covers o_proj too; qwen2 biases only q/k/v.
        assert block.attn.o_bias is (family == "llama")
        assert not block.mlp.bias
        assert (block.scalar is not None) is want["scalar"]
        assert block.d_model == 32 and block.d_ff == 48
        assert block.attn.n_heads == 4
    assert [b.attn.sliding_window for b in blocks] == want["windows"]
    assert [b.attn.rope_table for b in blocks] == want["tables"]
    gemma = family.startswith("gemma")
    assert spec.embed_scale == pytest.approx(math.sqrt(32) if gemma else 1.0)
    assert spec.embed is model.get_input_embeddings()
    assert (spec.n_layers, spec.d_model, spec.d_ff, spec.vocab) == (3, 32, 48, 64)
    assert spec.final_norm.offset == want["offset"]


def test_gemma_specific_terms() -> None:
    """Softcaps, the per-layer scalar, keys-as-values and the weightless value norm."""
    gemma2 = model_spec(tiny_dense_model("gemma2"))
    assert gemma2.final_logit_softcap == 30.0
    assert all(b.attn.logit_softcap == 50.0 for b in gemma2.blocks)
    assert gemma2.blocks[0].attn.scaling == pytest.approx(8 ** -0.5)

    gemma4 = model_spec(tiny_dense_model("gemma4_text"))
    assert gemma4.final_logit_softcap == 20.0
    sliding, full = gemma4.blocks[0], gemma4.blocks[1]
    assert sliding.attn.v is not None and full.attn.v is None
    assert (sliding.attn.head_dim, full.attn.head_dim) == (8, 16)
    assert (sliding.attn.n_kv_heads, full.attn.n_kv_heads) == (2, 1)
    assert full.attn.group_size == 4 and full.attn.kv_head(3) == 0
    assert full.attn.v_norm is not None and not full.attn.v_norm.weighted
    assert full.attn.scaling == 1.0
    module: Any = sliding.module
    assert sliding.scale() == pytest.approx(float(module.layer_scalar))

    llama = model_spec(tiny_dense_model("llama"))
    assert llama.final_logit_softcap is None
    assert llama.blocks[0].scale() == 1.0
    assert llama.blocks[0].attn.scaling == pytest.approx(8 ** -0.5)


def test_norm_spec_apply_and_gain_match_the_module(dense: tuple[str, Any]) -> None:
    _family, model = dense
    block = model_spec(model).blocks[1]
    x = torch.randn(5, 32, dtype=torch.float64)
    for norm in filter(None, (block.attn_in_norm, block.attn_out_norm, block.mlp_in_norm, block.mlp_out_norm)):
        with torch.no_grad():
            expected = norm.module(x)
        torch.testing.assert_close(norm.apply(x).double(), expected, atol=1e-5, rtol=1e-5)
        assert norm.gain().dtype == torch.float32 and norm.gain().shape == (32,)


def test_multimodal_composite_describes_its_language_model() -> None:
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    cfg = Gemma3Config(
        text_config=dict(
            model_type="gemma3_text", hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, vocab_size=64, max_position_embeddings=64,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, image_size=28, patch_size=14,
        ),
        mm_tokens_per_image=4,
    )
    model = Gemma3ForConditionalGeneration(cfg).eval()
    spec = model_spec(model)
    assert [b.module for b in spec.blocks] == list(model.model.language_model.layers)
    assert spec.embed is model.model.language_model.embed_tokens
    assert spec.rotary is model.model.language_model.rotary_emb
    assert spec.final_norm.module is model.model.language_model.norm


# -- refusals ---------------------------------------------------------------------

_MOE = {"moe_intermediate_size": 16, "num_experts": 4, "num_experts_per_tok": 2}

_REFUSED: list[tuple[str, str, dict[str, Any], int | None, str]] = [
    ("glm-fused-gate-up", "glm", {}, 0, "fused gate-up projection"),
    ("phi3-fused-qkv", "phi3", {}, 0, "fused qkv projection"),
    ("qwen3-moe", "qwen3_moe", _MOE, 0, "mixture-of-experts MLP"),
    ("qwen2-moe", "qwen2_moe", {**_MOE, "shared_expert_intermediate_size": 16}, 0,
     "mixture-of-experts MLP"),
    ("gpt-oss-sinks", "gpt_oss", {"num_local_experts": 4, "num_experts_per_tok": 2}, 0,
     "attention tensor 'sinks'"),
    ("qwen3.5-hybrid", "qwen3_5_text", dict(
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
    ), 0, "linear-attention layer ('linear_attn')"),
    ("gemma4-per-layer-input", "gemma4_text", {"hidden_size_per_layer_input": 16}, 0,
     "'per_layer_input_gate'"),
    ("gemma4-kv-sharing", "gemma4_text", {"num_kv_shared_layers": 1}, 2,
     "keys and values are shared"),
    ("gemma4-moe", "gemma4_text", {
        "enable_moe_block": True, "num_experts": 4, "top_k_experts": 2,
        "moe_intermediate_size": 16,
    }, 0, "mixture-of-experts block"),
    ("ministral3-query-scaling", "ministral3", {}, None, "llama_4_scaling_beta"),
    ("granite-multiplier", "granite", {"residual_multiplier": 0.5}, None,
     "'residual_multiplier'"),
    ("cohere-layernorm", "cohere", {"logit_scale": 1.0}, 0, "not an RMSNorm"),
    ("gpt2-no-rope", "gpt2", {}, None, "no rotary embedding"),
    ("smollm3-nope", "smollm3", {}, 0, "applies no RoPE"),
    ("olmo2-whole-q-norm", "olmo2", {}, 0, "q_norm"),
]


@pytest.mark.parametrize(
    "model_type,overrides,layer,fragment",
    [case[1:] for case in _REFUSED],
    ids=[case[0] for case in _REFUSED],
)
def test_unsupported_blocks_are_refused_naming_layer_and_reason(
    model_type: str, overrides: dict[str, Any], layer: int | None, fragment: str,
) -> None:
    model = tiny_dense_model(model_type, **overrides)
    with pytest.raises(UnsupportedBlockError) as info:
        model_spec(model)
    assert info.value.layer == layer
    assert fragment in str(info.value)
    if layer is not None:
        assert str(info.value).startswith(f"layer {layer}: ")
    assert info.value.user_message()[0] == 422


def test_verify_refuses_a_forward_term_the_structure_hides() -> None:
    """helium is llama-shaped but rotates RoPE pairs interleaved."""
    model = tiny_dense_model("helium")
    spec = model_spec(model)  # structurally a dense block
    with pytest.raises(UnsupportedBlockError, match="attention forward differs") as info:
        verify_model_spec(model, spec)
    assert info.value.layer == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("family", sorted(DENSE_FAMILIES))
def test_verify_accepts_default_init_bf16_models(family: str, seed: int) -> None:
    """bf16 rounding is not a term: a fresh bf16 model of each family verifies.

    Held to its bf16 forward, an fp32 description missed by 5-20% of the
    update on these seeds (mistral, qwen2, qwen3) and on real checkpoints
    (SmolLM2-360M, Qwen3-0.6B); the fp32 replay separates the two.
    """
    model = default_init_model(family, seed=seed, dtype=torch.bfloat16)
    spec = verify_model_spec(model)
    assert len(spec.blocks) == len(get_layers(model))


def test_verify_still_refuses_a_hidden_term_in_bf16() -> None:
    """The replay does not loosen the check: helium in bf16 is still refused."""
    model = tiny_dense_model("helium").to(torch.bfloat16)
    with pytest.raises(UnsupportedBlockError, match="attention forward differs"):
        verify_model_spec(model)


@pytest.mark.parametrize("cap", [50.0, 2.0, 0.5])
def test_verify_refuses_a_softcap_the_forward_does_not_apply(cap: float) -> None:
    """Gemma 3 keeps ``attn_logit_softcapping`` but never passes it to attention.

    A cap of 50 never binds on the probe's scores; the boosted replay
    shows it anyway.
    """
    model = tiny_dense_model("gemma3_text", attn_logit_softcapping=cap)
    assert model_spec(model).blocks[0].attn.logit_softcap == cap
    with pytest.raises(UnsupportedBlockError, match="applies no logit softcap") as info:
        verify_model_spec(model)
    assert info.value.layer == 0


def test_verify_refuses_a_cap_the_description_lacks() -> None:
    """gemma2 caps its scores at 50; a description without the cap is refused."""
    model = tiny_dense_model("gemma2")
    spec = model_spec(model)
    first = spec.blocks[0]
    uncapped = dataclasses.replace(
        spec, blocks=(
            dataclasses.replace(first, attn=dataclasses.replace(first.attn, logit_softcap=None)),
            *spec.blocks[1:],
        ),
    )
    with pytest.raises(UnsupportedBlockError, match="attention forward differs") as info:
        verify_model_spec(model, uncapped)
    assert info.value.layer == 0
    verify_model_spec(model, spec)
    module: Any = first.attn.module
    assert module.scaling == first.attn.scaling  # the probe restores the module


def test_model_spec_is_memoized_per_live_model() -> None:
    """Discovery runs device forwards once per model, not on every read."""
    model = tiny_dense_model("llama")
    norm = cast(nn.Module, get_layers(model)[0].input_layernorm)
    probes: list[int] = []
    handle = norm.register_forward_hook(lambda *_: probes.append(1))
    try:
        spec = model_spec(model)
        assert probes == [1]
        assert model_spec(model) is spec and block_specs(model) == list(spec.blocks)
        assert probes == [1]
    finally:
        handle.remove()

    # A replaced submodule is a different structure: discovered again.
    layer: Any = get_layers(model)[2]
    layer.input_layernorm = copy.deepcopy(layer.input_layernorm)
    fresh = model_spec(model)
    assert fresh is not spec and fresh.blocks[2].attn_in_norm.module is layer.input_layernorm

    # The memo holds the model weakly.
    ref = weakref.ref(model)
    del model, layer, norm, spec, fresh
    gc.collect()
    assert ref() is None


def test_a_duck_typed_mlp_with_an_extra_term_is_refused() -> None:
    model = tiny_dense_model("llama")
    mlp = get_layers(model)[1].mlp
    original = type(mlp)

    class ScaledMLP(original):  # type: ignore[valid-type, misc]
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return 2.0 * super().forward(x)

    mlp.__class__ = ScaledMLP
    with pytest.raises(UnsupportedBlockError, match="does not compute") as info:
        model_spec(model)
    assert info.value.layer == 1


def test_an_unknown_activation_is_refused() -> None:
    model = tiny_dense_model("llama")
    layer: Any = get_layers(model)[0]
    layer.mlp.act_fn = nn.GELU()  # exact (erf) GELU
    with pytest.raises(UnsupportedBlockError, match="neither SiLU nor tanh-GELU"):
        model_spec(model)


@pytest.mark.parametrize(("model_type", "offset"), [("gemma3_text", 1.0), ("gemma4_text", 0.0)])
def test_large_bf16_norm_gains_keep_their_convention(model_type: str, offset: float) -> None:
    """A bf16 module rounds gains of hundreds by whole units (gemma-3-270m's MLP
    output norms reach ~1,650), so the convention is judged relative to the gain."""
    model = default_init_model(model_type, seed=0, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(0)
    gain = 200.0 + 1450.0 * torch.rand(32, generator=gen)
    gain[:4] = 0.5 + 4.0 * torch.rand(4, generator=gen)  # a few small gains, as in the 270m
    norm = get_layers(model)[1].post_feedforward_layernorm
    norm.weight.copy_(gain - offset)
    spec = model_spec(model)
    assert spec.blocks[1].mlp_out_norm is not None
    assert spec.blocks[1].mlp_out_norm.offset == offset


def test_a_layernorm_is_refused() -> None:
    model = tiny_dense_model("llama")
    get_layers(model)[2].input_layernorm = nn.LayerNorm(32).double()
    with pytest.raises(UnsupportedBlockError, match="attention input norm") as info:
        model_spec(model)
    assert info.value.layer == 2


# -- activations ------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["silu", "gelu_tanh"])
def test_activation_factor_and_derivative(kind: Any) -> None:
    g = torch.linspace(-5, 5, 101, dtype=torch.float64, requires_grad=True)
    y = activation(kind, g)
    (grad,) = torch.autograd.grad(y.sum(), g)
    torch.testing.assert_close(activation_derivative(kind, g.detach()), grad)
    torch.testing.assert_close(activation_factor(kind, g.detach()) * g.detach(), y.detach())
    module = nn.SiLU() if kind == "silu" else nn.GELU(approximate="tanh")
    torch.testing.assert_close(y.detach(), module(g.detach()))
