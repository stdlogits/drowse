"""The neutral cache's mid means: the centring mean of an S read at ``mid``.

The neutral forward also averages each block's ``mid`` residual (after its
attention, the MLP's input) at the activations' own pooled positions; the
cache stores one ``[D]`` mean per layer beside the activations, a cache
without them is stale and recomputes, and a model the block description
refuses records them as unavailable (a residual read at ``mid`` then
refuses, never reads uncentred).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from drowse.errors import NeutralMeansError, UnsupportedBlockError

pytestmark = pytest.mark.usefixtures("isolated_home")


def _hand_mid_means(session: Any) -> dict[int, torch.Tensor]:
    """The mid residual at each neutral row's pooled position, averaged, from
    a plain forward per row (a pre-hook on each block's MLP input norm)."""
    from drowse.fit import capture
    from drowse.model.blocks import model_spec

    pairs = capture._neutral_pairs()
    rendered = capture._prepare_capture_batch(
        session.tokenizer, [p for p, _ in pairs], [r for _, r in pairs],
    )
    spec = model_spec(session.model)
    sums = {block.layer: torch.zeros(spec.d_model, dtype=torch.float64) for block in spec.blocks}
    seen: dict[int, torch.Tensor] = {}

    def keep(layer: int) -> Any:
        def hook(_m: torch.nn.Module, args: tuple[Any, ...]) -> None:
            seen[layer] = args[0][0].detach().double()
        return hook

    handles = [
        block.mlp_in_norm.module.register_forward_pre_hook(keep(block.layer))
        for block in spec.blocks
    ]
    try:
        with torch.no_grad():
            for ids, end in rendered:
                session.model(input_ids=ids, use_cache=False)
                for layer in sums:
                    sums[layer] += seen[layer][int(end)]
    finally:
        for handle in handles:
            handle.remove()
    return {layer: (total / len(rendered)).float() for layer, total in sums.items()}


def test_the_neutral_pass_stores_each_blocks_mid_mean(tiny_session: Any) -> None:
    from drowse.base.paths import model_dir
    from drowse.fit.neutral import load_or_compute_neutral_cache
    from drowse.store.neutral import NEUTRAL_CAPTURE_VERSION, load_neutral_cache

    cache = load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id, force=True,
    )
    assert cache.mid.means is not None and cache.mid.reason is None
    expected = _hand_mid_means(tiny_session)
    assert sorted(cache.mid.means) == sorted(expected) == sorted(cache.activations)
    for layer, mean in cache.mid.means.items():
        assert mean.dtype == torch.float32 and mean.shape == (cache.activations[layer].shape[1],)
        torch.testing.assert_close(mean, expected[layer], rtol=1e-4, atol=1e-5)
        # The mid mean is not the block-output mean (the MLP sits between).
        assert not torch.allclose(mean, cache.activations[layer].mean(0), atol=1e-4)
    sidecar = json.loads(
        (model_dir(tiny_session.model_id) / "neutral_activations.json").read_text()
    )
    assert sidecar["mid"] == {"available": True}
    assert sidecar["capture_version"] == NEUTRAL_CAPTURE_VERSION == 2
    reloaded = load_neutral_cache(tiny_session.model_id)
    assert reloaded.mid.means is not None
    for layer, mean in cache.mid.means.items():
        assert torch.equal(reloaded.mid.means[layer], mean)


def test_a_cache_without_mid_means_recomputes(tiny_session: Any) -> None:
    """A cache written before the mid means is stale derived data: the load
    refuses it and the next neutral load recomputes it, never fails and never
    skips centring at mid."""
    from drowse.base.paths import model_dir
    from drowse.fit.neutral import load_or_compute_neutral_cache
    from drowse.store.neutral import load_neutral_cache

    pointer = model_dir(tiny_session.model_id) / "neutral_activations.json"
    sidecar = json.loads(pointer.read_text())
    sidecar["capture_version"] = 1
    del sidecar["mid"]
    pointer.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError):
        load_neutral_cache(tiny_session.model_id)
    cache = load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id,
    )
    assert cache.mid.means is not None
    assert load_neutral_cache(tiny_session.model_id).sidecar["mid"] == {"available": True}


def test_a_malformed_mid_block_or_tensor_is_rejected(tiny_session: Any) -> None:
    from drowse.base.paths import model_dir
    from drowse.store.neutral import load_neutral_cache

    pointer = model_dir(tiny_session.model_id) / "neutral_activations.json"
    sidecar = json.loads(pointer.read_text())
    pointer.write_text(json.dumps({**sidecar, "mid": {"available": True, "extra": 1}}))
    with pytest.raises(ValueError, match="mid"):
        load_neutral_cache(tiny_session.model_id)
    # A cache that says mid means are unavailable while its shards hold them
    # does not match its sidecar.
    pointer.write_text(json.dumps({**sidecar, "mid": {"available": False, "reason": "x"}}))
    with pytest.raises(ValueError, match="keys"):
        load_neutral_cache(tiny_session.model_id)


def test_an_undescribed_model_has_no_mid_means_and_mid_reads_refuse(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.model.blocks as blocks_mod
    from drowse.fit.neutral import load_or_compute_neutral_cache
    from drowse.session.neutral import NeutralBaseline

    def refuse(_model: Any) -> Any:
        raise UnsupportedBlockError("mixture-of-experts block ('experts')", layer=0)

    monkeypatch.setattr(blocks_mod, "model_spec", refuse)
    cache = load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id, force=True,
    )
    assert cache.mid.means is None and "experts" in str(cache.mid.reason)
    assert cache.sidecar["mid"] == {"available": False, "reason": cache.mid.reason}
    from contextlib import nullcontext

    baseline = NeutralBaseline(
        model=tiny_session.model, tokenizer=tiny_session.tokenizer,
        layers=tiny_session.layers, model_id=tiny_session.model_id,
        exclusive=lambda *_a, **_kw: nullcontext(), steered=lambda: False,
    )
    assert sorted(baseline.site_means("out")) == list(range(len(tiny_session.layers)))
    with pytest.raises(NeutralMeansError, match="experts") as info:
        baseline.site_means("mid")
    assert info.value.user_message()[0] == 422


def test_a_failed_mid_pass_on_a_described_model_recomputes_on_the_next_load(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an out-of-memory error at capture batch size 1 published
    ``mid: {available: false}`` and every later load reused it, so every S
    read of a mid residual refused (422) for a model that has mid means.
    Only the block description's refusal lasts; a failure recomputes."""
    import drowse.fit.capture as capture_mod
    from drowse.fit.neutral import MidMeans, load_or_compute_neutral_cache

    def oom(self: MidMeans) -> Any:
        raise RuntimeError("out of memory at capture batch size 1 (simulated)")

    with monkeypatch.context() as patch:
        patch.setattr(MidMeans, "finish", oom)
        failed = load_or_compute_neutral_cache(
            tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
            model_id=tiny_session.model_id, force=True,
        )
    assert failed.mid.means is None and "simulated" in str(failed.mid.reason)
    assert failed.sidecar["mid"]["available"] is False

    passes: list[int] = []
    compute = capture_mod.compute_neutral_activations

    def counting(*args: Any, **kwargs: Any) -> Any:
        passes.append(1)
        return compute(*args, **kwargs)

    monkeypatch.setattr(capture_mod, "compute_neutral_activations", counting)
    cache = load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id,
    )
    assert passes == [1]
    assert cache.mid.means is not None and cache.sidecar["mid"] == {"available": True}
    # The recomputed cache is reused as is.
    load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id,
    )
    assert passes == [1]


def test_a_refused_model_reuses_its_unavailable_record(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model the description refuses keeps ``mid`` (and units) unavailable
    across loads without recomputing: that state is the model's."""
    import drowse.fit.capture as capture_mod
    import drowse.model.blocks as blocks_mod
    from drowse.fit.neutral import load_or_compute_neutral_cache

    def refuse(_model: Any) -> Any:
        raise UnsupportedBlockError("mixture-of-experts block ('experts')", layer=0)

    monkeypatch.setattr(blocks_mod, "model_spec", refuse)
    load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id, force=True,
    )
    passes: list[int] = []
    compute = capture_mod.compute_neutral_activations

    def counting(*args: Any, **kwargs: Any) -> Any:
        passes.append(1)
        return compute(*args, **kwargs)

    monkeypatch.setattr(capture_mod, "compute_neutral_activations", counting)
    cache = load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id,
    )
    assert passes == []
    assert cache.mid.means is None and "experts" in str(cache.mid.reason)
    assert cache.units.tensors is None


def test_site_means_are_the_monitor_mean_at_out_and_the_mid_means_at_mid(
    tiny_session: Any,
) -> None:
    from drowse.store.neutral import load_neutral_cache

    cache = load_neutral_cache(tiny_session.model_id)
    lens = tiny_session.lens
    assert cache.mid.means is not None
    for layer, mean in tiny_session.layer_means.items():
        assert lens.neutral_mean("out", layer) is mean
        torch.testing.assert_close(mean, cache.activations[layer].mean(0))
        assert torch.equal(lens.neutral_mean("mid", layer), cache.mid.means[layer])
    with pytest.raises(NeutralMeansError, match="no layer 99"):
        lens.neutral_mean("out", 99)


def test_a_baseline_that_cannot_build_refuses_residual_reads() -> None:
    from contextlib import nullcontext

    from drowse.session.neutral import NeutralBaseline

    class Broken(NeutralBaseline):
        def ensure(self, *, on_progress: Any = None) -> None:
            self._load_error = RuntimeError("disk full")

    baseline = Broken(
        model=object(), tokenizer=object(), layers=[], model_id="m",
        exclusive=lambda *_a, **_kw: nullcontext(), steered=lambda: False,
    )
    with pytest.raises(NeutralMeansError, match="disk full"):
        baseline.site_means("out")
