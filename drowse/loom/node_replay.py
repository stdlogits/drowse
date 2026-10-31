"""Node replay: one forward over a loom node's tokens, tapped inside every block.

:class:`NodeReplayer` rebuilds a node's forward the way every token readout
does (:meth:`drowse.loom.replay.TokenReplayer.forward`: the node's prompt
render plus its raw decode record, under the node recipe's steering unless
the caller asks for the unsteered counterfactual, inside the exclusive
section) and runs it once with taps where the model's parts meet the
residual: each block's input, its ``mid`` residual (after attention, the
MLP's input), its output before and after the steering hooks, the down
projection's input (the neurons) and output, and the output projection's
input (the heads' ``z``) and output.  Eager attention is switched on for the
forward only when attention patterns are asked for.  The forward stops after
the last block: nothing here needs the vocabulary head over every position.

Two reads are built on it:

- :meth:`NodeReplayer.token_anatomy` — one token, every layer (plan section E
  and "a9's lens correction"): what the residual reads as, what attention and
  the MLP added, which neurons and heads pushed the token, and where the top
  heads looked.  Returned inside the one measurement envelope
  (``scope="replay"``, the envelope's ``anatomy`` key).
- :meth:`NodeReplayer.unit_activity` — every token, every unit: each neuron's
  activation and each head's write norm over its neutral scale (the units
  family's ``activation_over_scale``), reduced on the device to the top units
  by peak, the traces of requested units and one token's whole population.

**Positions.**  Raw token ``i`` is *produced* at position ``prompt_length +
i − 1`` (its input is the token before it), so a token's anatomy and its
unit activity are those of its producing forward, exactly as its loom row's
live readings and the families' token readouts are.  Position 0 is the
attention sink: unit activity there is out of distribution (the unit
statistics never count it) and is reported as 0 and excluded from the peaks,
as the units runtime excludes it.

Every result leaves the device in one transfer (the lens reads, the
contributions, the patterns and the flags are packed together).
"""

from __future__ import annotations

import base64
import math
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn as nn
from typing_extensions import TypedDict

from drowse.errors import RequestError
from drowse.lens.read import LensRead, VectorRead
from drowse.loom.ops import resolve_token_anchor
from drowse.loom.replay import ReplayForward, TokenReplayer
from drowse.model.blocks import BlockSpec, ModelSpec
from drowse.model.layers import LayerSelector, resolve_model_layers
from drowse.model.units import head_write_matrix
from drowse.naming import HeadRef, NeuronRef, UnitRef
from drowse.readout.measurements import (
    AnatomyLayer,
    AnatomyPattern,
    AnatomyUnit,
    LensReadDict,
    MeasurementsEnvelope,
    TokenAnatomy,
    build_measurements,
)
from drowse.units.capture import SCALE_FLOOR
from drowse.units.stats import UNIT_POSITIONS_FROM, UnitStats

if TYPE_CHECKING:
    from drowse.lens.instrument import LensInstrument
    from drowse.loom.tree import LoomTree
    from drowse.units.runtime import UnitsRuntime

#: ``activation_over_scale``: the units family's probe unit.
UNIT_ACTIVITY_UNIT = "activation_over_scale"
#: The population encoding: IEEE float16, little-endian, base64 (RFC 4648),
#: one value per neuron (in index order) or head; values beyond the float16
#: range are clamped to ±65504.
POPULATION_ENCODING = "float16-le-base64"
_F16_MAX = 65504.0
#: Fp32 column chunk for a vector times a model-dtype weight.
_COLUMN_CHUNK = 4096


# --------------------------------------------------------------------------
# Wire shapes (unit activity; the anatomy's live in readout/measurements.py)
# --------------------------------------------------------------------------

class ActivityToken(TypedDict):
    raw_index: int
    token_id: int
    token: str


class ActivityPeak(TypedDict):
    """One unit's peak over the node: its signed value (over scale) at the
    token where ``|value|`` peaks."""

    unit: str
    kind: Literal["neuron", "head"]
    layer: int
    index: int
    peak: float
    raw_index: int


class PopulationLayer(TypedDict):
    """One layer's whole population at one token, packed
    (:data:`POPULATION_ENCODING`): ``neurons`` has ``d_ff`` values, ``heads``
    ``n_heads``."""

    layer: int
    neurons: str
    heads: str
    n_neurons: int
    n_heads: int


class ActivityPopulation(TypedDict):
    raw_index: int
    encoding: str
    layers: list[PopulationLayer]


class UnitActivityJSON(TypedDict):
    """Unit activity over a node's tokens (:meth:`NodeReplayer.unit_activity`).

    ``tokens`` are the node's raw tokens, each read at its producing forward;
    ``sink_raw_index`` names the token produced at the attention sink
    (position 0), whose values are 0 and never peak, or is ``None``.
    ``traces`` maps each requested unit to one value per token.
    """

    node_id: str
    steering: str | None
    unit: str
    tokens: list[ActivityToken]
    sink_raw_index: int | None
    top: list[ActivityPeak]
    traces: dict[str, list[float]]
    population: ActivityPopulation | None


# --------------------------------------------------------------------------
# The tapped forward
# --------------------------------------------------------------------------

class _Stop(BaseException):
    """The last block ran: the replay needs nothing after it."""


def _first(output: Any) -> torch.Tensor:
    return output if isinstance(output, torch.Tensor) else output[0]


def _row(x: torch.Tensor, position: int) -> torch.Tensor:
    """``x[0, position]`` as a fresh fp32 copy (a later in-place hook cannot
    reach it)."""
    return x[0, position].to(dtype=torch.float32, copy=True)


@contextmanager
def eager_attention(spec: ModelSpec, enabled: bool) -> Iterator[None]:
    """Run the forward under eager attention (whose modules return their
    probabilities), restoring the model's implementation after."""
    config = getattr(spec.blocks[0].attn.module, "config", None) if enabled else None
    current = getattr(config, "_attn_implementation", None)
    if config is None or current == "eager":
        yield
        return
    setattr(config, "_attn_implementation", "eager")
    try:
        yield
    finally:
        setattr(config, "_attn_implementation", current)


@contextmanager
def _registered(handles: list[Any]) -> Iterator[list[Any]]:
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def _forward(model: nn.Module, ids: torch.Tensor) -> None:
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    except _Stop:
        pass


@dataclass
class TokenTaps:
    """What one forward carries at one position ``p``, fp32 on the device.

    ``block_in[L]`` is the residual entering block ``L``; ``mid[L]`` the
    residual after its attention; ``out_pre[L]`` its output before the
    steering hooks and ``out[L]`` after them (what block ``L + 1`` reads).
    ``acts[L]`` is the down projection's input (the neurons), ``mlp_out[L]``
    its output; ``z[L]`` the output projection's input (the heads' ``z``),
    ``attn_out[L]`` its output.  ``patterns[L]`` is ``[n_heads, p + 1]``,
    each head's attention from ``p`` over the context (eager attention only).
    """

    block_in: torch.Tensor
    mid: torch.Tensor
    out_pre: torch.Tensor
    out: torch.Tensor
    acts: list[torch.Tensor]
    mlp_out: list[torch.Tensor]
    z: list[torch.Tensor]
    attn_out: list[torch.Tensor]
    patterns: list[torch.Tensor] | None

    @property
    def attention_write(self) -> torch.Tensor:
        """``mid − block_in``: what attention added, ``[L, d]``."""
        return self.mid - self.block_in

    @property
    def mlp_write(self) -> torch.Tensor:
        """``out_pre − mid``: what the MLP added (with Gemma 4's per-layer
        scalar, the block's whole change after ``mid``), ``[L, d]``."""
        return self.out_pre - self.mid

    @property
    def steering_write(self) -> torch.Tensor:
        """``out − out_pre``: what the steering hooks added, ``[L, d]``."""
        return self.out - self.out_pre


def tap_token(
    model: nn.Module, spec: ModelSpec, ids: torch.Tensor, position: int,
    *, patterns: bool,
) -> TokenTaps:
    """One forward of ``ids`` ``[1, T]`` tapped at ``position``.

    ``patterns`` needs the attention modules to return their probabilities
    (run it under :func:`eager_attention`).  Each hook copies one row; the
    block-output tap sits before the steering hooks (``prepend``) and after
    them, so the two together isolate the injection.  Every projection tap
    is a forward hook, so it sees the input the module used (after a unit
    ablation's pre-hook).
    """
    blocks = spec.blocks
    n = len(blocks)
    p = int(position)
    first: dict[str, torch.Tensor] = {}
    mid: dict[int, torch.Tensor] = {}
    pre: dict[int, torch.Tensor] = {}
    post: dict[int, torch.Tensor] = {}
    acts: dict[int, torch.Tensor] = {}
    mlp_out: dict[int, torch.Tensor] = {}
    zs: dict[int, torch.Tensor] = {}
    attn_out: dict[int, torch.Tensor] = {}
    rows: dict[int, torch.Tensor] = {}

    def block_input(_m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        h = args[0] if args else kwargs["hidden_states"]
        first["h"] = _row(h, p)

    def tap(layer: int, store: dict[int, torch.Tensor], *, out: bool = False, stop: bool = False) -> Any:
        def hook(_m: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            store[layer] = _row(_first(output) if out else args[0], p)
            if stop:
                raise _Stop()
        return hook

    def pair(layer: int, inputs: dict[int, torch.Tensor], outputs: dict[int, torch.Tensor]) -> Any:
        def hook(_m: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            inputs[layer] = _row(args[0], p)
            outputs[layer] = _row(_first(output), p)
        return hook

    def pattern(layer: int) -> Any:
        def hook(_m: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError(
                    f"block {layer}'s attention returned no probabilities; patterns "
                    "need eager attention"
                )
            rows[layer] = weights[0, :, p, : p + 1].to(dtype=torch.float32, copy=True)
        return hook

    handles: list[Any] = [
        blocks[0].module.register_forward_pre_hook(block_input, with_kwargs=True),
    ]
    with _registered(handles):
        for block in blocks:
            L = block.layer
            handles.append(block.module.register_forward_hook(tap(L, pre, out=True), prepend=True))
            handles.append(block.module.register_forward_hook(
                tap(L, post, out=True, stop=L == n - 1),
            ))
            handles.append(block.mlp_in_norm.module.register_forward_hook(tap(L, mid)))
            handles.append(block.mlp.down.register_forward_hook(pair(L, acts, mlp_out)))
            handles.append(block.attn.o.register_forward_hook(pair(L, zs, attn_out)))
            if patterns:
                handles.append(block.attn.module.register_forward_hook(pattern(L)))
        _forward(model, ids)
    if "h" not in first or len(post) != n or len(mid) != n:
        raise RuntimeError("the replay forward did not reach every block")
    out = torch.stack([post[L] for L in range(n)])
    return TokenTaps(
        block_in=torch.cat([first["h"][None], out[:-1]]),
        mid=torch.stack([mid[L] for L in range(n)]),
        out_pre=torch.stack([pre[L] for L in range(n)]),
        out=out,
        acts=[acts[L] for L in range(n)],
        mlp_out=[mlp_out[L] for L in range(n)],
        z=[zs[L] for L in range(n)],
        attn_out=[attn_out[L] for L in range(n)],
        patterns=[rows[L] for L in range(n)] if patterns else None,
    )


# --------------------------------------------------------------------------
# Unit math on the description
# --------------------------------------------------------------------------

def _vec_mat(v: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``v @ weight`` in fp32 for a model-dtype ``weight`` ``[d, n]``, chunked
    over columns so the fp32 copy stays small."""
    v = v.to(device=weight.device, dtype=torch.float32)
    out = torch.empty(int(weight.shape[1]), dtype=torch.float32, device=weight.device)
    for start in range(0, int(weight.shape[1]), _COLUMN_CHUNK):
        stop = start + _COLUMN_CHUNK
        out[start:stop] = v @ weight[:, start:stop].detach().to(torch.float32)
    return out


def _norm_scale(norm: Any, row: torch.Tensor) -> torch.Tensor:
    """A sandwich norm's per-position scale ``rsqrt(mean(x²) + eps)`` (held
    constant: the LN rule), or 1 without one."""
    if norm is None:
        return torch.ones((), dtype=torch.float32, device=row.device)
    return torch.rsqrt(row.float().pow(2).mean() + float(norm.eps))


def neuron_contributions(
    block: BlockSpec, g_out: torch.Tensor, acts: torch.Tensor, mlp_out: torch.Tensor,
) -> torch.Tensor:
    """Each neuron's direct contribution to a logit through ``g_out``, ``[d_ff]``.

    ``act_n · (g · write_n)`` with ``write_n = s · γ_out ⊙ W_down[:, n] · r``:
    ``s`` the per-layer scalar, ``γ_out``/``r`` the MLP output norm's gain and
    per-position scale (Gemma's sandwich; absent: 1).  ``g_out`` is the
    logit's cotangent at the block's output (``J_Lᵀ u``).
    """
    g = g_out.float()
    if block.mlp_out_norm is not None:
        g = g * block.mlp_out_norm.gain(device=g.device)
    r = _norm_scale(block.mlp_out_norm, mlp_out) * block.scale()
    return acts.float() * _vec_mat(g, block.mlp.down.weight) * r


def head_contributions(
    block: BlockSpec, g_mid: torch.Tensor, z: torch.Tensor, attn_out: torch.Tensor,
) -> torch.Tensor:
    """Each head's direct contribution through ``g_mid`` (the logit's
    cotangent at the block's ``mid``), ``[n_heads]``:
    ``Σ_i (γ ⊙ g · W_O)[h·hd + i] · z[h·hd + i] · r``."""
    attn = block.attn
    g = g_mid.float()
    if block.attn_out_norm is not None:
        g = g * block.attn_out_norm.gain(device=g.device)
    per = _vec_mat(g, attn.o.weight) * z.float()
    r = _norm_scale(block.attn_out_norm, attn_out)
    return per.reshape(attn.n_heads, attn.head_dim).sum(dim=1) * r


def head_write_norms(block: BlockSpec, z: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
    """``‖W_O^h z_h‖`` per head (times the sandwich scale), ``[n_heads]``: the
    units family's head firing (a deliberate Euclidean read of a scalar)."""
    attn = block.attn
    grams = _grams(block, z.device)
    zh = z.float().reshape(attn.n_heads, attn.head_dim)
    norm = torch.einsum("hi,hij,hj->h", zh, grams, zh).clamp_min(0.0).sqrt()
    return norm * _norm_scale(block.attn_out_norm, attn_out)


_GRAMS: "weakref.WeakKeyDictionary[nn.Module, dict[str, torch.Tensor]]" = weakref.WeakKeyDictionary()


def _grams(block: BlockSpec, device: torch.device) -> torch.Tensor:
    """``[n_heads, hd, hd]`` write grams ``W_O^hᵀ W_O^h`` (with the sandwich
    gain), fp32 on ``device``, built once per output projection and device."""
    per_device = _GRAMS.setdefault(block.attn.o, {})
    cached = per_device.get(str(device))
    if cached is None:
        cached = torch.stack([
            (lambda m: m.T @ m)(head_write_matrix(block, h))
            for h in range(block.attn.n_heads)
        ]).to(device=device, dtype=torch.float32)
        per_device[str(device)] = cached
    return cached


def unembed_cotangent(
    spec: ModelSpec, h_final: torch.Tensor, token_id: int,
) -> torch.Tensor:
    """The cotangent of ``token_id``'s logit at the final residual, ``[d]``.

    ``u = c' · r · γ ⊙ W_U[v]``: the final norm linearized at ``h_final``
    with its scale ``r = rsqrt(mean(h²) + eps)`` held constant (the LN rule
    the RelP lenses read under), ``γ`` its gain, and ``c' = 1 − tanh²(z/c)``
    the final softcap's slope at the token's logit (1 without a cap).
    """
    h = h_final.float()
    norm = spec.final_norm
    row = spec.unembed[int(token_id)].detach().to(device=h.device, dtype=torch.float32)
    gain = norm.gain(device=h.device)
    r = torch.rsqrt(h.pow(2).mean() + float(norm.eps))
    u = r * gain * row
    cap = spec.final_logit_softcap
    if cap is not None:
        logit = (h * r * gain) @ row
        u = u * (1.0 - torch.tanh(logit / float(cap)).pow(2))
    return u


# --------------------------------------------------------------------------
# One packed transfer
# --------------------------------------------------------------------------

_Slot = tuple[int, tuple[int, ...]]


class _Pack:
    """Device tensors flattened into one fp32 vector for a single host copy."""

    def __init__(self, device: torch.device) -> None:
        self._parts: list[torch.Tensor] = []
        self._size = 0
        self._device = device

    def add(self, tensor: torch.Tensor) -> _Slot:
        flat = tensor.detach().to(device=self._device, dtype=torch.float32).reshape(-1)
        slot = (self._size, tuple(int(dim) for dim in tensor.shape))
        self._parts.append(flat)
        self._size += int(flat.numel())
        return slot

    def host(self) -> torch.Tensor:
        """The one device-to-host transfer."""
        if not self._parts:
            return torch.zeros(0)
        return torch.cat(self._parts).cpu()


def _take(host: torch.Tensor, slot: _Slot) -> torch.Tensor:
    start, shape = slot
    return host[start:start + math.prod(shape)].reshape(shape)


def encode_population(values: torch.Tensor) -> str:
    """``values`` as :data:`POPULATION_ENCODING` (clamped to the float16 range)."""
    half = values.float().clamp(-_F16_MAX, _F16_MAX).to(torch.float16).cpu()
    return base64.b64encode(half.numpy().astype("<f2").tobytes()).decode("ascii")


def decode_population(text: str) -> torch.Tensor:
    """The inverse of :func:`encode_population`, fp32."""
    import numpy as np

    raw = np.frombuffer(base64.b64decode(text), dtype="<f2")
    return torch.from_numpy(raw.astype("float32"))


def _unit_name(kind: str, layer: int, index: int) -> str:
    ref: UnitRef = NeuronRef(layer, index) if kind == "neuron" else HeadRef(layer, index)
    return ref.qualified


def _anatomy_unit(
    kind: str, layer: int, index: int, contribution: float, activation: float | None,
) -> AnatomyUnit:
    return {
        "unit": _unit_name(kind, layer, index),
        "kind": "neuron" if kind == "neuron" else "head",
        "layer": int(layer),
        "index": int(index),
        "contribution": round(float(contribution), 6),
        "activation": None if activation is None else round(float(activation), 6),
    }


class _FlatUnits:
    """A flat index over every unit of some layers (``[(kind, layer,
    width)…]`` in order), mapped back to ``(kind, layer, index)``."""

    def __init__(self, groups: Sequence[tuple[str, int, int]]) -> None:
        self._starts: list[tuple[int, str, int]] = []
        start = 0
        for kind, layer, width in groups:
            self._starts.append((start, kind, layer))
            start += int(width)
        self.size = start

    def unit(self, flat: int) -> tuple[str, int, int]:
        for begin, kind, layer in reversed(self._starts):
            if flat >= begin:
                return kind, layer, flat - begin
        raise ValueError(f"flat unit index {flat} is out of range")


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

class NodeReplayer:
    """Token anatomy and unit activity over a loom node's replayed forward.

    Dependencies: the session's :class:`TokenReplayer` (anchor, render,
    steering scope, exclusive section), the model, the lens instrument (its
    resident R and S lenses, its read planning and neutral means), the units
    runtime (the block description and the neutral unit statistics), the
    tokenizer and the loom tree getter.
    """

    def __init__(
        self,
        *,
        replayer: TokenReplayer,
        model: Any,
        lens: "LensInstrument",
        units: "UnitsRuntime",
        tokenizer: Any,
        tree: Callable[[], "LoomTree"],
    ) -> None:
        self._replayer = replayer
        self._model = getattr(model, "_orig_mod", model)
        self._lens = lens
        self._units = units
        self._tokenizer = tokenizer
        self._tree = tree

    def _decode(self, token_id: int) -> str:
        return str(self._tokenizer.decode([int(token_id)]))

    # ------------------------------------------------------------ anatomy

    def token_anatomy(
        self,
        node_id: str,
        raw_index: int,
        *,
        k: int = 8,
        units: int = 8,
        patterns: int = 4,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> MeasurementsEnvelope:
        """One token, every layer, as the ``scope="replay"`` envelope.

        Per layer: the residual at ``out`` read through both lens families
        (``reads_as``; its S half centred on the neutral out mean), what
        attention added (``mid − block input``, read at ``mid`` as a
        direction), what the MLP added (``out − mid`` before the steering
        hooks, read at ``out`` as a direction), what steering added (the
        hooks' injection, read at ``out``; ``None`` where nothing was
        injected), and the top ``units`` neurons and heads by direct
        contribution to the token's logit through the R lens (``act_n · g_L ·
        w_n``, ``g_L = J_Lᵀ u`` with ``u`` the token's norm-linearized
        unembedding row; heads through ``J`` at ``mid``).  Across layers the
        top ``units`` units, and the attention from the token's producing
        position of the top ``patterns`` heads (0: none, and no eager
        attention).  Raises ``UnsupportedBlockError`` (422) for a model the
        block description refuses; without an R lens the contributions are
        absent (``units_absent`` says why).
        """
        if not 1 <= int(k) <= 256:
            raise RequestError("k must be in [1, 256]")
        if not 1 <= int(units) <= 64:
            raise RequestError("units must be in [1, 64]")
        if not 0 <= int(patterns) <= 16:
            raise RequestError("patterns must be in [0, 16]")
        spec = self._units.spec()
        if self._lens.resident_input() is not None:
            # The centring mean is resident before the steering scope opens
            # (a neutral build refuses inside one).
            self._lens.neutral_mean("out", 0)

        def run(forward: ReplayForward) -> TokenAnatomy:
            with eager_attention(spec, patterns > 0):
                taps = tap_token(
                    self._model, spec, forward.ids, forward.position, patterns=patterns > 0,
                )
            with torch.inference_mode():
                return self._anatomy(
                    spec, taps, forward, k=int(k), units=int(units), patterns=int(patterns),
                )

        token, anatomy = self._replayer.forward(
            node_id, raw_index, run=run, apply_steering=apply_steering, raw=raw,
            label="session.token_anatomy",
        )
        anatomy["binding"]["steering"] = token.steering
        return {"measurements": build_measurements(
            scope="replay", provenance="replayed", anatomy=anatomy,
            steering=token.steering,
        )}

    def _anatomy(
        self, spec: ModelSpec, taps: TokenTaps, forward: ReplayForward,
        *, k: int, units: int, patterns: int,
    ) -> TokenAnatomy:
        n = len(spec.blocks)
        device = taps.out.device
        token_id = int(forward.raw_ids[forward.raw_index])
        lens = self._lens
        pack = _Pack(device)
        requests: list[VectorRead] = []
        for L in range(n):
            requests.append(VectorRead(taps.out[L], L, "out", "residual"))
            requests.append(VectorRead(taps.attention_write[L], L, "mid", "direction"))
            requests.append(VectorRead(taps.mlp_write[L], L, "out", "direction"))
            requests.append(VectorRead(taps.steering_write[L], L, "out", "direction"))
        pending = lens.plan_reads(requests, k=k)
        reads_slot = pack.add(pending.tensor)
        injected_slot = pack.add((taps.steering_write.abs().amax(dim=1) > 0).float())
        logits = lens.runtime.readout_modules()[1](taps.out[-1][None])[0].float()
        prob_slot = pack.add(logits.log_softmax(-1)[token_id].exp().reshape(1))

        contributions = self._contributions(spec, taps, token_id)
        absent: str | None = None
        neuron_slots: dict[int, tuple[_Slot, _Slot, _Slot]] = {}
        head_slots: dict[int, tuple[_Slot, _Slot, _Slot]] = {}
        flat = _FlatUnits(())
        top_slots: tuple[_Slot, _Slot] | None = None
        pattern_slots: tuple[_Slot, _Slot, _Slot] | None = None
        if isinstance(contributions, str):
            absent = contributions
        else:
            neuron_c, head_c = contributions
            for L, c in neuron_c.items():
                idx = c.abs().topk(min(units, int(c.numel()))).indices
                neuron_slots[L] = (pack.add(c[idx]), pack.add(idx), pack.add(taps.acts[L][idx]))
            for L, c in head_c.items():
                block = spec.blocks[L]
                idx = c.abs().topk(min(units, int(c.numel()))).indices
                norms = head_write_norms(block, taps.z[L], taps.attn_out[L])
                head_slots[L] = (pack.add(c[idx]), pack.add(idx), pack.add(norms[idx]))
            groups = [("neuron", L, int(c.numel())) for L, c in neuron_c.items()]
            groups += [("head", L, int(c.numel())) for L, c in head_c.items()]
            flat = _FlatUnits(groups)
            every = torch.cat([*neuron_c.values(), *head_c.values()])
            best = every.abs().topk(min(units, int(every.numel()))).indices
            top_slots = (pack.add(every[best]), pack.add(best))
            if patterns > 0 and taps.patterns is not None and head_c:
                heads = torch.cat(list(head_c.values()))
                chosen = heads.abs().topk(min(patterns, int(heads.numel()))).indices
                rows = torch.cat([taps.patterns[L] for L in head_c])          # [ΣH, T]
                base = sum(int(c.numel()) for c in neuron_c.values())
                pattern_slots = (
                    pack.add(rows[chosen]), pack.add(chosen + base), pack.add(heads[chosen]),
                )

        host = pack.host()
        reads = pending.finish(_take(host, reads_slot))
        injected = _take(host, injected_slot).tolist()
        probability = float(_take(host, prob_slot)[0])

        layers: list[AnatomyLayer] = []
        for L in range(n):
            reads_as, attention, mlp, steering = (reads[4 * L + i][0] for i in range(4))
            neurons: list[AnatomyUnit] = []
            heads_out: list[AnatomyUnit] = []
            if L in neuron_slots:
                c, idx, act = (_take(host, slot).tolist() for slot in neuron_slots[L])
                neurons = [
                    _anatomy_unit("neuron", L, int(i), v, a)
                    for v, i, a in zip(c, idx, act, strict=True)
                ]
            if L in head_slots:
                c, idx, norm = (_take(host, slot).tolist() for slot in head_slots[L])
                heads_out = [
                    _anatomy_unit("head", L, int(i), v, w)
                    for v, i, w in zip(c, idx, norm, strict=True)
                ]
            layers.append({
                "layer": L,
                "depth": round(reads_as.depth, 6),
                "reads_as": _read_dict(reads_as),
                "attention": _read_dict(attention),
                "mlp": _read_dict(mlp),
                "steering": _read_dict(steering) if injected[L] > 0 else None,
                "neurons": neurons,
                "heads": heads_out,
            })

        top_units: list[AnatomyUnit] = []
        if top_slots is not None:
            values, flat_idx = (_take(host, slot).tolist() for slot in top_slots)
            for value, index in zip(values, flat_idx, strict=True):
                kind, L, unit = flat.unit(int(index))
                top_units.append(_anatomy_unit(kind, L, unit, value, None))
        pattern_rows: list[AnatomyPattern] | None = None
        if pattern_slots is not None:
            rows_host = _take(host, pattern_slots[0])
            flat_heads = _take(host, pattern_slots[1]).tolist()
            values = _take(host, pattern_slots[2]).tolist()
            pattern_rows = []
            for row, index, value in zip(rows_host, flat_heads, values, strict=True):
                _kind, L, head = flat.unit(int(index))
                pattern_rows.append({
                    "unit": _unit_name("head", L, head),
                    "layer": L,
                    "head": head,
                    "contribution": round(float(value), 6),
                    "weights": [round(float(w), 6) for w in row.tolist()],
                })
        return {
            "token": {
                "raw_index": forward.raw_index,
                "token_id": token_id,
                "token": self._decode(token_id),
                "position": forward.position,
                "prompt_length": forward.prompt_length,
                "probability": round(probability, 6),
            },
            "binding": {
                "sources": {"R": lens.runtime.source, "S": lens.input_runtime.source},
                "steering": None,
            },
            "layers": layers,
            "units": top_units,
            "units_absent": absent,
            "patterns": pattern_rows,
            "context": (
                [self._decode(int(t)) for t in forward.ids[0].tolist()]
                if pattern_rows is not None else None
            ),
        }

    def _contributions(
        self, spec: ModelSpec, taps: TokenTaps, token_id: int,
    ) -> "tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]] | str":
        """``({layer: neuron contributions}, {layer: head contributions})`` at
        the layers the R lens reads (neurons through ``J`` at ``out``, heads
        at ``mid``), or why there are none."""
        lens = self._lens.resident()
        if lens is None:
            return (
                "no R-family lens is selected: the units that pushed the token "
                "are read through it (fetch or fit one)"
            )
        n = len(spec.blocks)
        device = taps.out.device
        u = unembed_cotangent(spec, taps.out[-1], token_id)
        sites = set(getattr(lens, "sites", ("out",)))

        def cotangents(site: str) -> dict[int, torch.Tensor]:
            if site not in sites:
                return {}
            layers = sorted(int(layer) for layer in lens.layers(site))
            if not layers:
                return {}
            stack = self._lens.runtime.transport_stack(lens, layers, device, site=site)
            g = torch.einsum("lij,i->lj", stack, u)                  # J_Lᵀ u, [L, d]
            return {layer: g[i] for i, layer in enumerate(layers)}

        g_out = cotangents("out")
        if g_out and n - 1 not in g_out:
            g_out[n - 1] = u     # the R target: h_final is the last block's output
        g_mid = cotangents("mid")
        neuron_c = {
            L: neuron_contributions(spec.blocks[L], g, taps.acts[L], taps.mlp_out[L])
            for L, g in sorted(g_out.items())
        }
        head_c = {
            L: head_contributions(spec.blocks[L], g, taps.z[L], taps.attn_out[L])
            for L, g in sorted(g_mid.items())
        }
        if not neuron_c and not head_c:
            return "the R lens holds no layer at either site"
        return neuron_c, head_c

    # ------------------------------------------------------------ activity

    def unit_activity(
        self,
        node_id: str,
        *,
        top: int = 32,
        units: Sequence[str] = (),
        raw_index: int | None = None,
        layers: LayerSelector = None,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> UnitActivityJSON:
        """Every neuron's and head's activity over a node's tokens.

        Each token is read at its producing forward, in the units family's
        unit (``activation_over_scale``: a neuron's activation over its
        pole's neutral ``act_top``, signed; a head's write norm over its
        ``act_top``), from one replay forward.  Returns the ``top`` units
        across the node by peak ``|value|`` with their peak token, the
        per-token trace of each unit in ``units``, and, with ``raw_index``,
        that token's whole population at ``layers`` (default every layer),
        packed (:data:`POPULATION_ENCODING`).  Raises
        ``UnitsUnavailableError`` (422) when the model has no unit
        statistics and a 400 for a malformed or out-of-range unit.
        """
        if not 1 <= int(top) <= 1024:
            raise RequestError("top must be in [1, 1024]")
        if layers is not None and raw_index is None:
            raise RequestError("layers selects a population, which needs raw_index")
        refs = [self._units.parse(selector) for selector in units]
        if len({ref.qualified for ref in refs}) != len(refs):
            raise RequestError("units names a unit twice")
        anchor = resolve_token_anchor(
            self._tree(), node_id, 0 if raw_index is None else int(raw_index),
            label="session.unit_activity",
        )
        spec = self._units.spec()
        stats = self._units.stats()
        pop_layers = (
            resolve_model_layers(layers, len(spec.blocks)) if raw_index is not None else []
        )

        def run(forward: ReplayForward) -> UnitActivityJSON:
            return self._activity(
                spec, stats, forward, top=int(top), refs=refs,
                population=None if raw_index is None else int(raw_index),
                pop_layers=pop_layers,
            )

        token, result = self._replayer.forward(
            node_id, len(anchor.raw_ids) - 1, run=run,
            apply_steering=apply_steering, raw=raw, label="session.unit_activity",
        )
        result["node_id"] = node_id
        result["steering"] = token.steering
        return result

    def _activity(
        self, spec: ModelSpec, stats: UnitStats, forward: ReplayForward,
        *, top: int, refs: Sequence[UnitRef], population: int | None,
        pop_layers: Sequence[int],
    ) -> UnitActivityJSON:
        blocks = spec.blocks
        n = len(blocks)
        ids = forward.ids
        device = ids.device
        count = len(forward.raw_ids)
        positions = torch.arange(
            forward.prompt_length - 1, forward.prompt_length - 1 + count, device=device,
        )
        sink = positions < UNIT_POSITIONS_FROM                                   # [N]
        wanted = {int(layer) for layer in pop_layers}
        picks: dict[tuple[str, int], list[int]] = {}
        for ref in refs:
            kind = "neuron" if isinstance(ref, NeuronRef) else "head"
            picks.setdefault((kind, ref.layer), []).append(ref.index)
        peaks: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        traces: dict[tuple[str, int], torch.Tensor] = {}
        pops: dict[tuple[str, int], torch.Tensor] = {}

        def reduce(key: tuple[str, int], values: torch.Tensor) -> None:
            values = values.masked_fill(sink[:, None], 0.0)
            at = values.abs().argmax(dim=0)
            peaks[key] = (values.gather(0, at[None])[0], at)
            chosen = picks.get(key)
            if chosen:
                traces[key] = values[:, torch.tensor(chosen, device=values.device)]
            if population is not None and key[1] in wanted:
                pops[key] = values[population]

        def neuron_hook(block: BlockSpec) -> Any:
            layer_stats = stats.layer(block.layer)
            plus = layer_stats.neuron_scale[0].to(device).clamp_min(SCALE_FLOOR)
            minus = layer_stats.neuron_scale[1].to(device).clamp_min(SCALE_FLOOR)
            last = block.layer == n - 1

            def hook(_m: nn.Module, args: tuple[Any, ...], _output: Any) -> None:
                act = args[0][0].index_select(0, positions).float()              # [N, d_ff]
                reduce(("neuron", block.layer), act / torch.where(act >= 0, plus, minus))
                if last:   # the forward's last tap: nothing after it is read
                    raise _Stop()
            return hook

        def head_hook(block: BlockSpec) -> Any:
            attn = block.attn
            scale = stats.layer(block.layer).head_scale.to(device).clamp_min(SCALE_FLOOR)
            grams = _grams(block, device)
            out_norm = block.attn_out_norm

            def hook(_m: nn.Module, args: tuple[Any, ...], output: Any) -> None:
                z = args[0][0].index_select(0, positions).float()
                zh = z.reshape(-1, attn.n_heads, attn.head_dim)
                norm = torch.einsum("nhi,hij,nhj->nh", zh, grams, zh).clamp_min(0.0).sqrt()
                if out_norm is not None:
                    o = _first(output)[0].index_select(0, positions).float()
                    norm = norm * torch.rsqrt(
                        o.pow(2).mean(-1, keepdim=True) + float(out_norm.eps),
                    )
                reduce(("head", block.layer), norm / scale)
            return hook

        handles: list[Any] = []
        with _registered(handles), torch.inference_mode():
            for block in blocks:
                handles.append(block.mlp.down.register_forward_hook(neuron_hook(block)))
                handles.append(block.attn.o.register_forward_hook(head_hook(block)))
            _forward(self._model, ids)
            if len(peaks) != 2 * n:
                raise RuntimeError("the replay forward did not reach every unit")
            order = [(kind, L) for L in range(n) for kind in ("neuron", "head")]
            flat = _FlatUnits([(kind, L, int(peaks[(kind, L)][0].numel())) for kind, L in order])
            signed = torch.cat([peaks[key][0] for key in order])
            at = torch.cat([peaks[key][1] for key in order])
            best = signed.abs().topk(min(top, int(signed.numel()))).indices
            pack = _Pack(device)
            top_slots = (pack.add(signed[best]), pack.add(best), pack.add(at[best]))
            trace_slots = {key: pack.add(value) for key, value in traces.items()}
            pop_slots = {key: pack.add(value) for key, value in pops.items()}
            host = pack.host()

        values, flat_idx, peak_at = (_take(host, slot).tolist() for slot in top_slots)
        peaks_out: list[ActivityPeak] = []
        for value, index, row in zip(values, flat_idx, peak_at, strict=True):
            kind, L, unit = flat.unit(int(index))
            peaks_out.append({
                "unit": _unit_name(kind, L, unit),
                "kind": "neuron" if kind == "neuron" else "head",
                "layer": L,
                "index": unit,
                "peak": round(float(value), 6),
                "raw_index": int(row),
            })
        traces_out: dict[str, list[float]] = {}
        for ref in refs:
            key = ("neuron" if isinstance(ref, NeuronRef) else "head", ref.layer)
            column = _take(host, trace_slots[key])[:, picks[key].index(ref.index)]
            traces_out[ref.qualified] = [round(float(v), 6) for v in column.tolist()]
        population_out: ActivityPopulation | None = None
        if population is not None:
            population_out = {
                "raw_index": population,
                "encoding": POPULATION_ENCODING,
                "layers": [
                    {
                        "layer": L,
                        "neurons": encode_population(_take(host, pop_slots[("neuron", L)])),
                        "heads": encode_population(_take(host, pop_slots[("head", L)])),
                        "n_neurons": pop_slots[("neuron", L)][1][0],
                        "n_heads": pop_slots[("head", L)][1][0],
                    }
                    for L in sorted(wanted)
                ],
            }
        sink_rows = [i for i, flag in enumerate(sink.tolist()) if flag]
        return {
            "node_id": "",
            "steering": None,
            "unit": UNIT_ACTIVITY_UNIT,
            "tokens": [
                {"raw_index": i, "token_id": int(t), "token": self._decode(int(t))}
                for i, t in enumerate(forward.raw_ids)
            ],
            "sink_raw_index": sink_rows[0] if sink_rows else None,
            "top": peaks_out,
            "traces": traces_out,
            "population": population_out,
        }


def _read_dict(read: LensRead) -> LensReadDict:
    return cast(LensReadDict, read.to_dict())


__all__ = [
    "ActivityPeak",
    "ActivityPopulation",
    "ActivityToken",
    "NodeReplayer",
    "POPULATION_ENCODING",
    "PopulationLayer",
    "TokenTaps",
    "UNIT_ACTIVITY_UNIT",
    "UnitActivityJSON",
    "decode_population",
    "eager_attention",
    "encode_population",
    "head_contributions",
    "head_write_norms",
    "neuron_contributions",
    "tap_token",
    "unembed_cotangent",
]
