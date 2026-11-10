"""Real-weight checks; select one model at a time with DROWSE_TEST_MODEL."""

import gc
import math
from typing import Any

import pytest
import torch

from drowse.fit.capture import _capture_all_hidden_states
from drowse.generation.decode import GenerationConfig, GenerationState, generate_steered
from drowse.lens.jlens import JacobianLens, transport_rows
from drowse.model.arch import get_layers
from drowse.model.load import load_model
from drowse.model.readout import LogitReadout
from tests.support.gpu import gpu_model_id, load_or_skip_inaccessible

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="GPU unavailable",
)]


@pytest.fixture(scope="module")
def loaded():
    model_id = gpu_model_id()
    model, tokenizer = load_or_skip_inaccessible(lambda: load_model(model_id, device="auto", quantize=None), model_id)
    yield model, tokenizer
    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@pytest.mark.parametrize("prompt", ["The capital of France is", "Once upon a time there was a cat"])
def test_native_identity_readout_matches_real_output_head(loaded: tuple[Any, Any], prompt: str):
    model, tokenizer = loaded
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    layers = get_layers(model)
    last = len(layers) - 1
    captured = _capture_all_hidden_states(model, layers, ids, layer_indices=[last], pool_index=ids.shape[1] - 1)
    dim = captured[last].numel()
    lens = JacobianLens({last: torch.eye(dim, device=device)}, n_prompts=1, d_model=dim)
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits[0, -1].float()
        row = captured[last].reshape(1, dim)
        J = lens.jacobians[last].to(device).unsqueeze(0)
        actual = LogitReadout(model)(transport_rows(J, row))[0].float()
    # The fused LM head and the lens readout run different kernels; on MPS
    # half-precision the two differ by a few ulps on a handful of logits.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)


def test_native_decode_matches_direct_cached_reference(loaded: tuple[Any, Any]):
    model, tokenizer = loaded
    device = next(model.parameters()).device
    ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to(device)
    references = []
    forced = []
    cache = None
    current = ids
    with torch.inference_mode():
        for _ in range(4):
            output = model(input_ids=current, past_key_values=cache, use_cache=True)
            logits = output.logits[0, -1].float()
            values, indices = logits.topk(min(1024, logits.numel()))
            probabilities = values.softmax(-1)
            references.append((indices, probabilities))
            forced.append(int(indices[0]))
            current = indices[:1].reshape(1, 1)
            cache = output.past_key_values
    del cache
    frames = []
    generated = generate_steered(
        model, tokenizer, ids, GenerationConfig(max_new_tokens=4, temperature=1., top_k=1024, top_p=1.),
        GenerationState(), forced_prefix=forced, seed=18, logprobs=5,
        on_token=lambda *frame: frames.append(frame),
    )
    assert generated == forced
    for frame in frames:
        _, _, token_id, logprob, alternatives, perplexity, step = frame
        indices, probabilities = references[step]
        assert token_id == forced[step]
        assert logprob == pytest.approx(float(probabilities[0].log()), abs=2e-5)
        expected_entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum()
        assert perplexity == pytest.approx(math.exp(float(expected_entropy)), rel=2e-5)
        assert [alt.id for alt in alternatives] == indices[:5].tolist()
        assert [alt.logprob for alt in alternatives] == pytest.approx(probabilities[:5].log().tolist(), abs=2e-5)


@pytest.mark.parametrize("ablate", [False, True])
def test_native_steering_matches_manual_intervention(loaded: tuple[Any, Any], ablate: bool):
    from drowse.steering.hooks import _SUBSPACE_GAIN, SteeringManager, SteeringPlan
    from drowse.geometry.subspace import LayerSubspace
    from drowse.geometry.inject import SynthesizedSubspace

    model, tokenizer = loaded
    parameter = next(model.parameters())
    layers = get_layers(model)
    layer = len(layers) // 2
    ids = tokenizer("A small cat sat", return_tensors="pt")["input_ids"].to(parameter.device)
    dim = model.config.hidden_size
    basis = torch.zeros(2, dim)
    basis[0, 3], basis[1, 7] = 1., 1.
    mean = torch.zeros(dim)
    target = torch.tensor([.25, 0. if ablate else -.125])
    kappa = torch.tensor([0., .5 if ablate else 0.])
    synth = SynthesizedSubspace({layer: LayerSubspace.affine(mean, basis, basis.clone())}, {layer: target}, {layer: 1.}, {layer: kappa})
    plan = SteeringPlan()
    plan.add_subspace("check", synth)
    manager = SteeringManager()
    manager.install(plan, layers, parameter.device, parameter.dtype)
    try:
        with torch.inference_mode():
            actual = model(ids, use_cache=False).logits
    finally:
        manager.clear_all()

    def intervene(_module: Any, _args: Any, output: Any):
        h = output[0] if isinstance(output, tuple) else output
        replacement = h.float().clone()
        replacement[..., 3] += .25 * _SUBSPACE_GAIN
        replacement[..., 7] = (
            replacement[..., 7] * .5 if ablate else replacement[..., 7] - .125 * _SUBSPACE_GAIN
        )
        replacement = replacement.to(h.dtype)
        return (replacement, *output[1:]) if isinstance(output, tuple) else replacement

    handle = layers[layer].register_forward_hook(intervene)
    try:
        with torch.inference_mode():
            expected = model(ids, use_cache=False).logits
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_verify_model_spec_accepts_the_loaded_model(loaded: tuple[Any, Any]):
    """A covered family verifies at its load precision (bf16 on CUDA and MPS).

    Held to its own bf16 forward, the fp32 description of SmolLM2-360M was
    refused (19% of layer 4's attention update is rounding against massive
    activations); the fp32 replay holds structure, not precision.
    """
    from drowse.errors import UnsupportedBlockError
    from drowse.model.blocks import model_spec, verify_model_spec

    model, _tokenizer = loaded
    try:
        spec = model_spec(model)
    except UnsupportedBlockError as exc:
        pytest.skip(f"{gpu_model_id()} is not a covered dense family: {exc}")
    assert verify_model_spec(model, spec, length=64) is spec
