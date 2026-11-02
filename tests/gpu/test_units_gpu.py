"""Units on a real model: neutral statistics, probes, gates and ablation.

Runs on the conftest ``gpu_session`` (``DROWSE_TEST_MODEL``, default
SmolLM2-360M-Instruct): the neutral pass measures every unit, a probe reads a
neuron and a head through a generation, and an ablation matches a
hand-patched forward at the model's own precision.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from drowse.generation.sampling import SamplingConfig
from tests.support.units import patched_forward_logits

pytestmark = pytest.mark.gpu


@pytest.fixture
def units_session(gpu_session: Any) -> Iterator[Any]:
    yield gpu_session
    for name in list(gpu_session.units.names):
        gpu_session.remove_probe(name)
    gpu_session.units.set_live(True)


def _units(session: Any) -> tuple[str, str]:
    spec = session.units.runtime.spec()
    layer = spec.n_layers // 3
    return f"neuron/L{layer}n{spec.d_ff // 3}", f"head/L{2 * spec.n_layers // 3}h1"


def test_the_neutral_pass_measures_every_unit(units_session: Any) -> None:
    stats = units_session.units.runtime.stats()
    spec = units_session.units.runtime.spec()
    assert sorted(stats.layers) == list(range(spec.n_layers))
    assert stats.n_documents > 24
    for layer in stats.layers.values():
        for tensor in layer.tensors().values():
            assert bool(torch.isfinite(tensor).all())
        # A dead pole is rare; a dead neuron (both poles) rarer still.
        assert float((layer.neuron_scale.amax(0) > 0).float().mean()) > 0.99
        assert bool((layer.head_scale > 0).all())


def test_unit_probes_read_a_generation(units_session: Any) -> None:
    neuron, head = _units(units_session)
    units_session.add_probe(neuron)
    units_session.add_probe(head)
    result = units_session.generate(
        "Name three colors.", stateless=True,
        sampling=SamplingConfig(max_tokens=12, temperature=0.0, seed=0),
    ).first
    readings = result.measurements["instruments"]["units"]["readings"]
    assert set(readings) == {neuron, head}
    for reading in readings.values():
        assert math.isfinite(reading["value"]) and abs(reading["value"]) < 1e4
    assert readings[head]["value"] >= 0
    assert not units_session.units.hooks_attached


def test_a_unit_ablation_matches_the_hand_patched_forward(units_session: Any) -> None:
    neuron, head = _units(units_session)
    runtime = units_session.units.runtime
    stats = runtime.stats()
    n_layer, n_index = (int(p) for p in neuron.split("/L")[1].split("n"))
    h_layer, h_index = (int(p) for p in head.split("/L")[1].split("h"))
    n_block, h_block = runtime.block(n_layer), runtime.block(h_layer)
    hd = h_block.attn.head_dim
    n_mean = float(stats.layer(n_layer).neuron_mean[n_index])
    h_mean = stats.layer(h_layer).head_mean[h_index]

    def ablate_neuron(x: torch.Tensor) -> torch.Tensor:
        x[..., n_index] = n_mean
        return x

    def ablate_head(x: torch.Tensor) -> torch.Tensor:
        x[..., h_index * hd:(h_index + 1) * hd] = h_mean.to(x)
        return x

    ids = units_session.tokenizer(
        "The quick brown fox jumps over the lazy dog.", return_tensors="pt",
    )["input_ids"].to(units_session.device)
    model = units_session.model
    expected = patched_forward_logits(
        model, ids, {n_block.mlp.down: ablate_neuron, h_block.attn.o: ablate_head},
    )
    plain = patched_forward_logits(model, ids, {})
    with units_session.steering(f"!{neuron} + !{head}"):
        with torch.inference_mode():
            got = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    scale = float(expected.abs().max())
    assert float((got - expected).abs().max()) <= 2e-2 * scale
    assert float((got - plain).abs().max()) > 0


def test_a_unit_gate_fires_on_a_real_model(units_session: Any) -> None:
    neuron, _head = _units(units_session)
    sampling = SamplingConfig(max_tokens=10, temperature=0.0, seed=0)

    def tokens(steering: str) -> list[int]:
        return list(units_session.generate(
            "Describe the sea.", stateless=True, steering=steering, sampling=sampling,
        ).first.tokens)

    always = tokens(f"!{neuron}@response")
    assert tokens(f"!{neuron}@when:{neuron}>-1e9") == always
    assert not units_session.units.hooks_attached
