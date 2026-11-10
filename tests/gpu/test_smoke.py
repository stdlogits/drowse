"""Smoke tests for drowse.

Requires a GPU (CUDA or Apple Silicon MPS) and downloads the public
SmolLM2-360M-Instruct test model on first run. Override with
``DROWSE_TEST_MODEL``. Run with: pytest tests/gpu/test_smoke.py -v
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.support.gpu import gpu_model_id, load_or_skip_inaccessible

# Skip entire module if no GPU backend is available.
_HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _HAS_GPU,
        reason="No GPU backend available (neither CUDA nor MPS)",
    ),
]

MODEL_ID = gpu_model_id()
# MPS runs ~3-5x slower than CUDA for this model; relax absolute timing budgets.


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from drowse.model.load import load_model
    # device="auto" picks cuda > mps > cpu; the skipif above guarantees a GPU.
    return load_or_skip_inaccessible(
        lambda: load_model(MODEL_ID, quantize=None, device="auto"), MODEL_ID,
    )


@pytest.fixture(scope="module")
def layers(model_and_tokenizer: Any) -> Any:
    from drowse.model.arch import get_layers
    model, _ = model_and_tokenizer
    return get_layers(model)


@pytest.fixture(scope="module")
def num_layers(layers: Any) -> int:
    return len(layers)


def _encode_and_capture_all(
    model: Any, tokenizer: Any, prompt: str, response: str, layers: Any,
    device: Any,
) -> dict[int, torch.Tensor]:
    """Last-content-token hidden state per layer for one turn pair, fp32."""
    from drowse.fit.capture import _encode_and_capture_all_batch

    batch = _encode_and_capture_all_batch(
        model, tokenizer, [prompt], [response], layers, device,
    )
    return {idx: rows[0] for idx, rows in batch.items()}


def _extract_profile(model: Any, tokenizer: Any, concept: str, layers: Any) -> Any:
    """A per-layer direction profile for a single concept vs the empty pole.

    The 4.0 equivalent of a difference-of-means vector: capture the two pole
    centroids, take their per-layer difference ``δ = pos − neg``, and fold that
    direction into a one-pole affine ray via the production
    ``fold_directions_to_subspace``, then read its baked-direction view
    (``folded_directions``).  Same centroid-difference direction the old
    DiM extractor produced, through the surviving fold primitive.
    """
    from drowse.geometry.fold import fold_directions_to_subspace, folded_directions

    device = next(model.parameters()).device
    prompt = "Describe your current state."
    pos = _encode_and_capture_all(model, tokenizer, prompt, concept, layers, device)
    neg = _encode_and_capture_all(model, tokenizer, prompt, "", layers, device)
    directions = {
        L: (pos[L].to(torch.float32) - neg[L].to(torch.float32))
        for L in pos if L in neg
    }
    from tests.support.whitener import isotropic_whitener
    neutral_means = {L: neg[L].to(torch.float32) for L in directions}
    dim = int(next(iter(directions.values())).numel())
    mfld = fold_directions_to_subspace(
        concept, directions, neutral_means,
        whitener=isotropic_whitener(directions, dim), label="p",
    )
    return folded_directions(mfld)


@pytest.fixture(scope="module")
def layer_means(model_and_tokenizer: Any, layers: Any) -> Any:
    # The probe-centering mean is the per-layer mean of the neutral activation
    # stack (X.mean(0)) — same corpus, same pooling — so there is no separate
    # compute_layer_means pass.
    from drowse.fit.capture import compute_neutral_activations
    model, tokenizer = model_and_tokenizer
    acts = compute_neutral_activations(model, tokenizer, layers)
    return {idx: X.mean(dim=0) for idx, X in acts.items()}


@pytest.fixture(scope="module")
def happy_profile(model_and_tokenizer: Any, layers: Any) -> Any:
    model, tokenizer = model_and_tokenizer
    return _extract_profile(model, tokenizer, "happy", layers)


def _steer_plan(*pairs: Any) -> Any:
    """Route baked ``(profile, alpha)`` pairs through the unified backend.

    Every vector lowers to a rank-1 push fragment (unit dir + baked-magnitude
    coord), all composed into one merged affine subspace via
    ``synthesize_subspace`` on one ``SteeringPlan`` — the dispatch the session
    does.
    Anchored at zero (enough for the smoke "steering changes the output"
    assertions; the GPU gate owns strength calibration).
    """
    from drowse.geometry.inject import synthesize_subspace
    from drowse.steering.hooks import SteeringPlan

    push: list[Any] = []
    neutral_means: dict[int, torch.Tensor] = {}
    for profile, alpha in pairs:
        basis_dirs: dict[int, torch.Tensor] = {}
        coord_dirs: dict[int, torch.Tensor] = {}
        for L, vec in profile.items():
            v = vec.to(torch.float32).reshape(-1)
            n = float(v.norm())
            if n < 1e-12:
                continue
            basis_dirs[L] = (v / n).reshape(1, -1)
            # Follow the profile's device (the loaded steering vectors live on
            # the model device); a CPU-default coord would mismatch the
            # on-device basis inside synthesize_subspace (``c_i @ B_i``).
            coord_dirs[L] = torch.tensor([n], device=v.device)
            neutral_means.setdefault(L, torch.zeros_like(v))
        push.append((basis_dirs, coord_dirs, alpha))
    from tests.support.whitener import isotropic_whitener
    dim = int(next(iter(neutral_means.values())).numel())
    synth = synthesize_subspace(
        push, [], neutral_means=neutral_means,
        whitener=isotropic_whitener(neutral_means, dim),
    )
    plan = SteeringPlan()
    plan.add_subspace("steer", synth)
    return plan


class TestVectorExtraction:
    def test_returns_valid_profile(self, happy_profile: Any, model_and_tokenizer: Any) -> None:
        model, _ = model_and_tokenizer
        cfg = getattr(model.config, "text_config", None) or model.config
        hidden_dim = cfg.hidden_size
        assert isinstance(happy_profile, dict)
        assert len(happy_profile) > 0
        for layer_idx, vec in happy_profile.items():
            assert isinstance(layer_idx, int)
            assert vec.shape == (hidden_dim,)
            norm = vec.norm().item()
            assert norm > 0 and not math.isinf(norm) and not math.isnan(norm)

class TestSteering:
    def test_steered_output_differs(self, model_and_tokenizer: Any, layers: Any, happy_profile: Any) -> None:
        from drowse.steering.hooks import SteeringManager
        from drowse.generation.decode import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        prompt = "Tell me about your day."
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False,
        ).to(device)

        config = GenerationConfig(max_new_tokens=20, temperature=0.0)

        # Unsteered
        state0 = GenerationState()
        ids0 = generate_steered(model, tokenizer, input_ids.clone(), config, state0)

        # Steered
        mgr = SteeringManager()
        mgr.install(_steer_plan((happy_profile, 5.0)), layers, device, dtype)

        state1 = GenerationState()
        ids1 = generate_steered(model, tokenizer, input_ids.clone(), config, state1)

        mgr.clear_all()

        assert ids0 != ids1, "Steered output should differ from unsteered"

class TestSteeringThroughput:
    @pytest.mark.perf
    def test_throughput_regression(self, model_and_tokenizer: Any, layers: Any, happy_profile: Any, layer_means: Any) -> None:
        """Steered generation should be at least 85% of vanilla throughput."""
        from drowse.steering.hooks import SteeringManager
        from drowse.generation.decode import GenerationConfig, GenerationState, generate_steered

        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a short story."}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False,
        ).to(device)
        config = GenerationConfig(max_new_tokens=100, temperature=0.7)

        # Finish every setup forward before either timed run.  Extracting the
        # two remaining profiles between vanilla and steered timing heats MPS
        # and made this ratio depend on test order rather than hook overhead.
        curious_profile = _extract_profile(model, tokenizer, "curious", layers)
        concise_profile = _extract_profile(model, tokenizer, "concise", layers)

        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()

        # Three vectors lower to one merged affine subspace.
        mgr = SteeringManager()
        plan = _steer_plan(
            (happy_profile, 0.8), (curious_profile, 0.5), (concise_profile, 0.3),
        )

        def timed(steered: bool) -> float:
            if steered:
                mgr.install(plan, layers, device, dtype)
            else:
                mgr.clear_all()
            state = GenerationState()
            synchronize()
            started = time.perf_counter()
            ids = generate_steered(
                model, tokenizer, input_ids.clone(), config, state, seed=0,
            )
            synchronize()
            return len(ids) / max(time.perf_counter() - started, 0.1)

        # MPS throughput drifts materially with command-cache warmup and device
        # temperature. One vanilla-then-steered pair can report anything from
        # 64% to >100% for this tiny model even though the hook path is
        # unchanged. Warm both graph shapes, then balance ordering and compare
        # medians. The 85% product gate itself remains unchanged.
        warm_config = GenerationConfig(max_new_tokens=2, temperature=0.7)
        original_config = config
        config = warm_config
        timed(False)
        timed(True)
        config = original_config
        vanilla_samples: list[float] = []
        steered_samples: list[float] = []
        for steered in (False, True, True, False, False, True):
            (steered_samples if steered else vanilla_samples).append(timed(steered))

        vanilla_tps = sorted(vanilla_samples)[1]
        steered_tps = sorted(steered_samples)[1]

        mgr.clear_all()

        ratio = steered_tps / vanilla_tps
        print(  # noqa: T201 — the gate reports its numbers
            f"steered: vanilla {vanilla_tps:.1f} tok/s, "
            f"steered {steered_tps:.1f} tok/s ({ratio:.0%})"
        )
        assert ratio >= 0.85, (
            f"Steered throughput ({steered_tps:.1f} tok/s) is only "
            f"{ratio:.0%} of vanilla ({vanilla_tps:.1f} tok/s), expected >= 85%; "
            f"vanilla samples={vanilla_samples!r}, steered samples={steered_samples!r}"
        )


class TestAblationPerformance:
    """Combined ablation + additive throughput must stay >= 80% of vanilla.

    Slightly looser than the steered-only 85% bar because the hot path
    does one extra matmul per active ablation layer per step. Intentional.
    """

    @pytest.mark.perf
    def test_throughput_with_ablation(
        self, model_and_tokenizer: Any, layers: Any, layer_means: Any,
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from drowse.generation.sampling import SamplingConfig
        from drowse.session.session import DrowseSession

        monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
        model, tokenizer = model_and_tokenizer
        session = DrowseSession(model, tokenizer, probes=["register"])

        try:
            prompt = "Write a 200-word essay on the history of bicycles."
            sampling = SamplingConfig(
                max_tokens=128, temperature=0.0, seed=0,
            )

            def synchronize() -> None:
                if session.device.type == "cuda":
                    torch.cuda.synchronize(session.device)
                elif session.device.type == "mps":
                    torch.mps.synchronize()

            # Pay one-time tokenizer, scene-template, and backend warmup before
            # either timed path. Both measurements then run the same fixed
            # decode budget; comparing a 510-token EOS completion with a
            # 1,024-token length completion mostly measures the latter's longer
            # KV-cache horizon, not steering overhead.
            session.generate(
                prompt,
                sampling=SamplingConfig(max_tokens=1, temperature=0.0, seed=0),
                stateless=True,
            )

            # Vanilla baseline.
            synchronize()
            t0 = time.perf_counter()
            # Keep the two measurements on identical prompt state.  Without
            # stateless mode the second call includes the first call's entire
            # essay in its prefill, so the old test mostly measured a much
            # longer conversation rather than steering overhead.
            r_vanilla = session.generate(
                prompt, sampling=sampling, stateless=True,
            ).first
            synchronize()
            dt_vanilla = max(time.perf_counter() - t0, 0.1)
            tok_s_vanilla = max(len(r_vanilla.tokens) / dt_vanilla, 1e-6)

            # Pick two probes from the auto-loaded set: one for additive, one for ablation.
            probes = list(session.probes)
            assert len(probes) >= 2, "need at least two probes for this test"
            additive_name, ablation_name = probes[0], probes[1]

            expr = f"0.3 {additive_name} + !{ablation_name}"
            with session.steering(expr):
                # Hook composition is a cold setup cost, not per-token hot-path
                # throughput. Time the same fixed decode only after the scope is
                # installed.
                synchronize()
                t0 = time.perf_counter()
                r_combined = session.generate(
                    prompt, sampling=sampling, stateless=True,
                ).first
                synchronize()
                dt_combined = max(time.perf_counter() - t0, 0.1)
            tok_s_combined = max(len(r_combined.tokens) / dt_combined, 1e-6)

            ratio = tok_s_combined / tok_s_vanilla
            print(  # noqa: T201 — the gate reports its numbers
                f"ablation: vanilla {tok_s_vanilla:.1f} tok/s, "
                f"ablation + additive {tok_s_combined:.1f} tok/s ({ratio:.0%})"
            )
            assert ratio >= 0.80, (
                f"combined ablation + additive too slow: "
                f"{ratio:.2%} of vanilla (vanilla={tok_s_vanilla:.1f} tok/s, "
                f"combined={tok_s_combined:.1f} tok/s; "
                f"tokens={len(r_vanilla.tokens)}/{len(r_combined.tokens)}, "
                f"seconds={dt_vanilla:.2f}/{dt_combined:.2f})"
            )
        finally:
            session.close()

    @pytest.mark.perf
    def test_throughput_with_unit_ablation(
        self, model_and_tokenizer: Any, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Eight neurons and two heads mean-ablated (``UnitOp`` pre-hooks on
        down and output projections) stay >= 80% of vanilla tok/s."""
        from drowse.generation.sampling import SamplingConfig
        from drowse.session.session import DrowseSession

        monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
        model, tokenizer = model_and_tokenizer
        session = DrowseSession(model, tokenizer, probes=[])
        try:
            spec = session.units.runtime.spec()
            n_layers, d_ff = spec.n_layers, spec.d_ff
            n_heads = spec.blocks[0].n_heads
            neurons = [
                f"!neuron/L{layer}n{(97 * layer) % d_ff}"
                for layer in range(2, n_layers, max(1, (n_layers - 2) // 8))
            ][:8]
            heads = [
                f"!head/L{n_layers // 3}h{1 % n_heads}",
                f"!head/L{2 * n_layers // 3}h{5 % n_heads}",
            ]
            assert len(neurons) == 8
            expr = " + ".join(neurons + heads)
            prompt = "Write a 200-word essay on the history of bicycles."
            sampling = SamplingConfig(max_tokens=128, temperature=0.0, seed=0)

            def synchronize() -> None:
                if session.device.type == "cuda":
                    torch.cuda.synchronize(session.device)
                elif session.device.type == "mps":
                    torch.mps.synchronize()

            def timed(steering: str | None) -> float:
                synchronize()
                started = time.perf_counter()
                result = session.generate(
                    prompt, sampling=sampling, stateless=True, steering=steering,
                ).first
                synchronize()
                return len(result.tokens) / max(time.perf_counter() - started, 0.1)

            # The first steer builds the neutral baseline and its unit
            # statistics; warm both paths before timing, then balance the
            # order and compare medians (MPS drifts with warmup and heat).
            warm = SamplingConfig(max_tokens=2, temperature=0.0, seed=0)
            session.generate(prompt, sampling=warm, stateless=True)
            session.generate(prompt, sampling=warm, stateless=True, steering=expr)
            vanilla: list[float] = []
            ablated: list[float] = []
            for steered in (False, True, True, False, False, True):
                (ablated if steered else vanilla).append(
                    timed(expr if steered else None),
                )
            vanilla_tps = sorted(vanilla)[1]
            ablated_tps = sorted(ablated)[1]
            ratio = ablated_tps / vanilla_tps
            print(  # noqa: T201 — the gate reports its numbers
                f"unit ablation: vanilla {vanilla_tps:.1f} tok/s, "
                f"ablated {ablated_tps:.1f} tok/s ({ratio:.0%})"
            )
            assert ratio >= 0.80, (
                f"unit ablation too slow: {ratio:.0%} of vanilla "
                f"(vanilla={vanilla_tps:.1f} tok/s, ablated={ablated_tps:.1f} tok/s; "
                f"vanilla samples={vanilla!r}, ablated samples={ablated!r})"
            )
        finally:
            session.close()


#: Lens-fit text for the instrumented gate: two documents are enough to give
#: the R lens its full per-layer shape (one ``d × d`` transport per source
#: layer), which is all the live readout's cost depends on.
_GATE_LENS_DOCS = [
    "The history of the printing press begins in fifteenth-century Mainz, "
    "where Johannes Gutenberg combined movable metal type with a screw press "
    "adapted from wine-making, transforming the economics of written text.",
    "Photosynthesis converts light energy into chemical energy in two stages: "
    "the light-dependent reactions split water and generate ATP and NADPH, "
    "while the Calvin cycle fixes atmospheric carbon dioxide into sugars.",
]


class TestDashboardPerformance:
    """The served dashboard's decode, as a fraction of vanilla tok/s.

    The other throughput gates time the steering hooks alone.  This one times
    a reply the way ``drowse serve`` streams it to the dashboard: the default
    probe roster scored on every token (with the inspector's subspace
    coordinates), the R lens's live readout over every fitted layer, eight
    alternatives per token, a loom row per token, and each token's WebSocket
    frame built and serialized.  On SmolLM2-360M (MPS) it measures about
    0.63 of vanilla (the lens readout is about 3 ms a token, the probes about
    2, the envelope, row and frame about 1.5); the bar leaves room for MPS
    drift.
    """

    @pytest.mark.perf
    def test_throughput_with_dashboard_instruments(
        self, model_and_tokenizer: Any, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        from drowse.generation.sampling import SamplingConfig
        from drowse.generation.token_callback import TokenConsumer, TokenConsumerOptions
        from drowse.server.ws_events import build_token_event
        from drowse.session.session import DrowseSession

        monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
        model, tokenizer = model_and_tokenizer
        session = DrowseSession(model, tokenizer, probes=None)
        try:
            assert len(session.probes) >= 10, sorted(session.probes)
            session.lens.runtime.lens = session.lens.fit(
                _GATE_LENS_DOCS, corpus_spec="gpu-dashboard-gate",
                dim_batch=64, seq_len=64,
            )
            live_layers = session.lens.set_live(True).layers or ()
            assert len(live_layers) >= len(session.layers) - 1
            prompt = "Write a 200-word essay on the history of bicycles."
            frames: list[int] = []

            def synchronize() -> None:
                if session.device.type == "cuda":
                    torch.cuda.synchronize(session.device)
                elif session.device.type == "mps":
                    torch.mps.synchronize()

            def timed(instrumented: bool, max_tokens: int = 128) -> float:
                if not instrumented:
                    sampling = SamplingConfig(max_tokens=max_tokens, temperature=0.0, seed=0)
                    synchronize()
                    started = time.perf_counter()
                    result = session.generate(prompt, sampling=sampling, stateless=True).first
                    synchronize()
                    return len(result.tokens) / max(time.perf_counter() - started, 0.1)
                # The dashboard's submit frame: alternatives on, subspace
                # coordinates on while probes sit in the rack, a loom row per
                # token, the native WebSocket consumer's options.
                sampling = SamplingConfig(
                    max_tokens=max_tokens, temperature=0.0, seed=0,
                    return_top_k=8, persist_subspace_coords=True,
                )
                holder: list[str | None] = [None]

                def on_event(event: Any) -> None:
                    frame = build_token_event(session, holder, event)
                    frames.append(len(json.dumps(frame, ensure_ascii=False)))

                consumer = TokenConsumer(on_event, TokenConsumerOptions(
                    live_scores=True, lens_readout=True, sae_readout=True,
                    perplexity=True,
                ))
                session.tree.reset()
                synchronize()
                started = time.perf_counter()
                result = session.generate(prompt, sampling=sampling, on_token=consumer).first
                synchronize()
                return len(result.tokens) / max(time.perf_counter() - started, 0.1)

            # Warm both paths (the first instrumented reply uploads the probe
            # factors and the live lens stack), then balance the order and
            # compare medians: MPS drifts with warmup and heat.
            timed(False, 4)
            timed(True, 4)
            frames.clear()
            vanilla: list[float] = []
            served: list[float] = []
            for instrumented in (False, True, True, False, False, True):
                (served if instrumented else vanilla).append(timed(instrumented))
            assert frames and all(size > 0 for size in frames)
            vanilla_tps = sorted(vanilla)[1]
            served_tps = sorted(served)[1]
            ratio = served_tps / vanilla_tps
            print(  # noqa: T201 — the gate reports its numbers
                f"dashboard: vanilla {vanilla_tps:.1f} tok/s, "
                f"instrumented {served_tps:.1f} tok/s ({ratio:.0%}); "
                f"{len(session.probes)} probes, {len(live_layers)} lens layers, "
                f"{sum(frames) // len(frames)} B/frame"
            )
            assert ratio >= _DASHBOARD_MIN_RATIO, (
                f"dashboard decode too slow: {ratio:.0%} of vanilla "
                f"(vanilla={vanilla_tps:.1f} tok/s, instrumented={served_tps:.1f} tok/s; "
                f"vanilla samples={vanilla!r}, instrumented samples={served!r})"
            )
        finally:
            session.lens.runtime.clear()
            session.close()


#: The instrumented gate's bar (see :class:`TestDashboardPerformance`).
_DASHBOARD_MIN_RATIO = 0.50
