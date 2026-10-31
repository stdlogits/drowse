"""Units-family helpers: an inert instrument for rigs without a model, a
hand-built neutral unit statistics set, and a reference forward that
replaces chosen unit activations by hand (the oracle the unit hooks are
checked against)."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from drowse.units.instrument import UnitsInstrument
from drowse.units.runtime import UnitsRuntime
from drowse.units.stats import LayerUnitStats, UnitStats

__all__ = [
    "inert_units_instrument",
    "patched_forward_logits",
    "random_unit_stats",
]


def _refuse() -> UnitStats:
    raise AssertionError("this rig has no unit statistics")


def inert_units_instrument(*, replay: Any = None) -> UnitsInstrument:
    """A real :class:`UnitsInstrument` over a model-less runtime.

    With no probes attached and no unit gates it never touches its runtime,
    so read-plane rigs can carry the fourth family without a model.
    """
    from tests.support.replay import unwired_replay

    lock = threading.RLock()

    @contextmanager
    def exclusive(_message: str, **_kwargs: Any) -> Iterator[None]:
        with lock:
            yield

    runtime = UnitsRuntime(
        model=torch.nn.Module(), device=torch.device("cpu"),
        exclusive=exclusive, stats=_refuse,
    )
    return UnitsInstrument(
        runtime,
        model_id="stub/model",
        n_layers=1,
        dtype=torch.float32,
        exclusive=exclusive,
        replay=unwired_replay if replay is None else replay,
        put_direction=lambda _name, _profile: None,
        has_direction=lambda _name: False,
    )


def random_unit_stats(spec: Any, *, seed: int = 0) -> UnitStats:
    """Positive random statistics shaped like ``spec``'s blocks."""
    gen = torch.Generator().manual_seed(seed)
    layers: dict[int, LayerUnitStats] = {}
    for block in spec.blocks:
        attn = block.attn
        layers[block.layer] = LayerUnitStats(
            neuron_mean=torch.randn(block.d_ff, generator=gen),
            neuron_scale=torch.rand(2, block.d_ff, generator=gen) + 0.5,
            head_mean=torch.randn(attn.n_heads, attn.head_dim, generator=gen),
            head_scale=torch.rand(attn.n_heads, generator=gen) + 0.5,
        )
    return UnitStats(layers=layers, n_documents=24)


def patched_forward_logits(
    model: Any,
    input_ids: torch.Tensor,
    patches: Mapping[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Last-position logits of one forward in which each module in
    ``patches`` reads ``patch(input)`` instead of its input (a pre-hook
    returning a fresh tensor, never touching drowse's hooks)."""
    def pre_hook(patch: Callable[[torch.Tensor], torch.Tensor]) -> Any:
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
            return (patch(args[0].clone()), *args[1:])
        return hook

    handles = []
    try:
        for module, patch in patches.items():
            handles.append(module.register_forward_pre_hook(pre_hook(patch)))
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
        return out.logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()
