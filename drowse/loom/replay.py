"""Loom-anchored token replay: rebuild the forward that produced a token.

:class:`TokenReplayer` is the one replay workflow behind every read family's
``token_readout`` (geometry, lens, SAE).  It resolves the ``(node_id,
raw_index)`` anchor, reopens the node's recorded steering, rebuilds the
node's render (continue-mode, or the flat raw buffer) plus the raw decode
prefix, and runs one capture forward pooled at the producing position.  The
family's own read runs inside the exclusive-GPU section through ``read``, so
device work never races a generation.  A token fork addresses tokens through
the same anchor (:func:`drowse.loom.ops.resolve_token_anchor`).

The instruments take a replayer as their :class:`~drowse.readout.replay.TokenReplay`
dependency; the session builds one over its tree, renderer, steering
composer, exclusive guard and capture forward.  :meth:`TokenReplayer.forward`
is the same rebuild handing the producing ids to a forward of the caller's
own (the node replay of :mod:`drowse.loom.node_replay`, with its own taps).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from drowse.base.exclusive import ExclusiveGuard
from drowse.errors import SteeringCompositionError
from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT
from drowse.loom.ops import TokenAnchor, resolve_token_anchor
from drowse.model.chat import logits_keep_kwarg
from drowse.readout.replay import ReplayedToken, ReplayLayers
from drowse.steering.triggers import Trigger, TriggerContext

if TYPE_CHECKING:
    from drowse.generation.render import PromptRenderer
    from drowse.loom.tree import LoomTree
    from drowse.model.chat import ChatTraits

T = TypeVar("T")


@dataclass(frozen=True)
class ReplayForward:
    """The producing forward's input, as :meth:`TokenReplayer.forward` hands
    it over: ``ids`` ``[1, T]`` (the node's prompt render, then its raw
    decode record before the anchored token), ``prompt_length`` (the render's
    length, so raw token ``i`` is produced at position ``prompt_length + i −
    1``), the node's whole raw decode record and the anchored raw index."""

    ids: torch.Tensor
    prompt_length: int
    raw_ids: tuple[int, ...]
    raw_index: int

    @property
    def position(self) -> int:
        """The producing position of the anchored token (the last one)."""
        return int(self.ids.shape[1]) - 1

    def producing_position(self, raw_index: int) -> int:
        """Where raw token ``raw_index`` is produced (its input is the token
        before it)."""
        return self.prompt_length + int(raw_index) - 1


class _ReplayStop(BaseException):
    """The last block ran: the replay needs nothing after it."""


def _forward(model: nn.Module, ids: torch.Tensor) -> None:
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    except _ReplayStop:
        pass


@dataclass(frozen=True)
class ReplaySegment:
    """One forward of a replay: positions ``[start, stop)`` under one trigger
    state (the prompt's ``prefill``, or decode step ``gen_step``: the forward
    producing raw token ``gen_step``)."""

    start: int
    stop: int
    prefill: bool
    gen_step: int


def replay_segments(
    prompt_length: int, length: int, triggers: Iterable[Trigger], *, can_think: bool,
) -> tuple[ReplaySegment, ...]:
    """The forwards that replay ``length`` positions the way generation ran
    them under ``triggers`` (every open steering term's trigger).

    Generation prefills ``[0, prompt_length)`` in one forward, then decode
    step ``k`` produces raw token ``k`` at position ``prompt_length + k − 1``.
    With every trigger ``Trigger.BOTH`` each position is steered alike and
    one forward over ``[0, length)`` is exact.  Otherwise the prefill is one
    segment and the decode positions split wherever a trigger's activity
    changes (``@first:N``, ``@after:N``), each segment evaluated by
    :meth:`Trigger.active` itself.  Raises
    :class:`~drowse.errors.SteeringCompositionError` (422) for a term whose
    firing depends on state the node does not record: a probe gate (the
    generation's live probe scores) or, when the family can think
    (``can_think``), a thinking-phase window (``@after``, ``@thinking``).
    """
    windowed = list(dict.fromkeys(t for t in triggers if t != Trigger.BOTH))
    if not windowed:
        return (ReplaySegment(0, int(length), prefill=False, gen_step=0),)
    gated = any(t.generated and t.gate is not None for t in windowed)
    phased = can_think and any(t.generated and t.thinking != t.response for t in windowed)
    if gated or phased:
        what = " and ".join(
            label for label, present in (
                ("probe-gated terms (@when:…), which fire on the generation's live "
                 "probe scores", gated),
                ("thinking-phase terms (@after, @thinking), which fire on the decode "
                 "loop's thinking state", phased),
            ) if present
        )
        raise SteeringCompositionError(
            f"the replayed steering has {what}; the node records neither, so a "
            "replay cannot reproduce where they fired. Replay without them "
            "(apply_steering=false drops the node's own steering)."
        )

    def fires(step: int) -> tuple[bool, ...]:
        ctx = TriggerContext(gen_step=step)
        return tuple(t.active(ctx) for t in windowed)

    prefill_stop = min(int(prompt_length), int(length))
    segments: list[ReplaySegment] = []
    if prefill_stop > 0:
        segments.append(ReplaySegment(0, prefill_stop, prefill=True, gen_step=0))
    start = prefill_stop
    while start < length:
        step = start - int(prompt_length) + 1
        state = fires(step)
        stop = start + 1
        while stop < length and fires(stop - int(prompt_length) + 1) == state:
            stop += 1
        segments.append(ReplaySegment(start, stop, prefill=False, gen_step=step))
        start = stop
    return tuple(segments)


class ReplayWindow:
    """The running forward's span of the replayed positions, read by the taps."""

    __slots__ = ("final", "start", "stop")

    def __init__(self) -> None:
        self.start = 0
        self.stop = 0
        self.final = True


def drive_segments(
    model: nn.Module,
    ids: torch.Tensor,
    segments: Sequence[ReplaySegment],
    ctx: TriggerContext | None,
    window: ReplayWindow,
) -> None:
    """Run ``ids`` through ``segments``, keeping ``window`` on the running one.

    The one rest-state segment (every trigger fires everywhere) is one
    cacheless forward, the context untouched.  Otherwise each segment, a
    lone prefill included, runs on one KV cache with the context set as the
    decode loop sets it (the attention sink in column 0 only from position
    0), and the context returns to its rest state after.  The taps raise
    :class:`_ReplayStop` in the last segment only.
    """
    if len(segments) == 1 and not segments[0].prefill:
        segment = segments[0]
        window.start, window.stop, window.final = segment.start, segment.stop, True
        _forward(model, ids[:, segment.start:segment.stop])
        return
    if ctx is None:
        raise ValueError("a segmented replay needs the trigger context")
    keep = logits_keep_kwarg(model)
    past: Any = None
    try:
        with torch.inference_mode():
            for i, segment in enumerate(segments):
                window.start, window.stop = segment.start, segment.stop
                window.final = i == len(segments) - 1
                ctx.is_prefill = segment.prefill
                ctx.thinking = False
                ctx.gen_step = segment.gen_step
                ctx.sink_column = 0 if segment.start == 0 else None
                ctx.probe_scores = {}
                kwargs: dict[str, Any] = {
                    "input_ids": ids[:, segment.start:segment.stop], "use_cache": True,
                }
                if past is not None:
                    kwargs["past_key_values"] = past
                if keep is not None:
                    kwargs[keep] = 1
                try:
                    out = model(**kwargs)
                except _ReplayStop:
                    if not window.final:
                        raise RuntimeError("a replay tap stopped before the last forward") from None
                    return
                past = getattr(out, "past_key_values", None)
                if past is None:
                    raise RuntimeError(
                        "the model returned no KV cache; a segmented replay runs on one"
                    )
                del out
    finally:
        ctx.reset()

def capture_pooled(
    model: nn.Module,
    layers: nn.ModuleList,
    ids: torch.Tensor,
    *,
    layer_indices: Sequence[int],
    pool_index: int,
    segments: Sequence[ReplaySegment],
    ctx: TriggerContext | None,
) -> dict[int, torch.Tensor]:
    """The token replay's capture forward: ``ids`` driven through
    ``segments`` (:func:`drive_segments`), each selected layer's output
    pooled at ``pool_index`` as an fp32 ``(hidden,)`` copy.

    The hooks register after the open steering's, so they read the steered
    residual.  ``pool_index`` lies in the last segment; the forward stops at
    the last selected layer there.
    """
    capture_layers = [int(idx) for idx in layer_indices]
    if not capture_layers:
        raise ValueError("layer_indices must select at least one layer")
    terminal = max(capture_layers)
    pool = int(pool_index)
    window = ReplayWindow()
    captured: dict[int, torch.Tensor] = {}

    def hook_for(idx: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: Any, output: Any) -> None:
            if not window.final or not window.start <= pool < window.stop:
                return
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            captured[idx] = hidden[0, pool - window.start].to(dtype=torch.float32, copy=True)
            if idx == terminal:
                raise _ReplayStop()
        return hook

    handles = [layers[idx].register_forward_hook(hook_for(idx)) for idx in capture_layers]
    try:
        drive_segments(getattr(model, "_orig_mod", model), ids, segments, ctx, window)
    finally:
        for handle in handles:
            handle.remove()
    return captured


class TokenReplayer:
    """Replays loom tokens' producing forwards for the read instruments.

    Dependencies:

    - ``tree``: the live loom tree (a callable: the session swaps its tree
      on restore);
    - ``renderer``: rebuilds the node's prompt render;
    - ``traits``: the chat facts (whether the family can think);
    - ``tokenizer``: decodes the replayed token's text;
    - ``open_scope``: opens a steering scope for an expression;
    - ``exclusive``: the exclusive-GPU guard;
    - ``capture``: ``capture(ids, *, layer_indices, pool_index, segments,
      ctx)`` — the capture forward (:func:`capture_pooled` over the model)
      returning ``{layer: (hidden,) fp32}``;
    - ``open_triggers``: the triggers of every open steering term (read
      inside the replay's scope);
    - ``trigger_context``: the steering's :class:`TriggerContext`, which a
      segmented replay sets as the decode loop set it.
    """

    def __init__(
        self,
        *,
        tree: Callable[[], "LoomTree"],
        renderer: "PromptRenderer",
        traits: "ChatTraits",
        tokenizer: Any,
        open_scope: Callable[[str], AbstractContextManager[Any]],
        exclusive: ExclusiveGuard,
        capture: Callable[..., dict[int, torch.Tensor]],
        open_triggers: Callable[[], Iterable[Trigger]] = tuple,
        trigger_context: TriggerContext | None = None,
    ) -> None:
        self._tree = tree
        self._renderer = renderer
        self._traits = traits
        self._tokenizer = tokenizer
        self._open_scope = open_scope
        self._exclusive = exclusive
        self._capture = capture
        self._open_triggers = open_triggers
        self._ctx = trigger_context

    def segments(self, forward: "ReplayForward") -> tuple[ReplaySegment, ...]:
        """The forwards replaying ``forward`` as the open steering fired
        during generation (:func:`replay_segments`; inside the replay's
        scope).  Raises :class:`~drowse.errors.SteeringCompositionError`
        (422) for a term whose firing the node does not record."""
        return replay_segments(
            forward.prompt_length, int(forward.ids.shape[1]), self._open_triggers(),
            can_think=bool(self._traits.supports_thinking),
        )

    def __call__(
        self,
        node_id: str,
        raw_index: int,
        *,
        layers: ReplayLayers,
        read: Callable[[dict[int, torch.Tensor]], T],
        apply_steering: bool = True,
        raw: bool = False,
        label: str,
    ) -> tuple[ReplayedToken, T]:
        """Replay the forward that produced ``node_id``'s ``raw_index``-th token.

        ``read`` receives ``{layer: (hidden,) fp32}`` pooled at the producing
        position and runs inside the exclusive-GPU section; ``layers`` may be
        a callable, resolved there too.  ``apply_steering`` reopens the node
        recipe's steering expression (a steered generation reads its steered
        state): an always-active affine term reproduces the decode
        injections exactly, and phase-split or probe-gated terms fire as the
        trigger context's rest state decides.  ``raw`` selects the flat
        render, which the node does not record.  Raises ``UnknownNodeError``
        / ``InvalidNodeOperationError`` on a bad anchor and
        ``ConcurrentExtractionError`` while another model op is in flight;
        ``label`` prefixes those messages.
        """

        def capture_and_read(forward: ReplayForward) -> T:
            ids = forward.ids
            layer_indices = list(layers() if callable(layers) else layers)
            hidden = self._capture(
                ids, layer_indices=layer_indices, pool_index=int(ids.shape[1]) - 1,
                segments=self.segments(forward), ctx=self._ctx,
            )
            return read(hidden)

        return self.forward(
            node_id, raw_index, run=capture_and_read,
            apply_steering=apply_steering, raw=raw, label=label,
        )

    def forward(
        self,
        node_id: str,
        raw_index: int,
        *,
        run: Callable[[ReplayForward], T],
        apply_steering: bool = True,
        raw: bool = False,
        label: str,
    ) -> tuple[ReplayedToken, T]:
        """The producing forward of ``node_id``'s ``raw_index``-th token,
        run by ``run`` (the rebuild of :meth:`__call__` without its capture).

        ``run`` receives the :class:`ReplayForward` and executes inside the
        exclusive-GPU section under the node recipe's steering (when
        ``apply_steering``), exactly where :meth:`__call__` runs its capture;
        it owns its forward, hooks and reads.  Anchoring a node's last raw
        token gives the forward that covers every token's producing position.
        """
        anchor = resolve_token_anchor(self._tree(), node_id, raw_index, label=label)
        recipe = anchor.node.recipe
        steering = (
            recipe.steering if (apply_steering and recipe is not None) else None
        )
        # The steering scope opens OUTSIDE the exclusive-GPU section: the
        # composer's push/pop take the generation lock blocking, so nesting
        # the scope inside the exclusive guard would self-deadlock.
        scope = self._open_scope(steering) if steering else nullcontext()
        with scope:
            with self._exclusive(
                f"{label} called while another model use is in flight",
                phase_msg=f"{label} called while a generation is in flight",
            ):
                # The render uploads the ids to the model's device, which is
                # device work like the forward's: outside the section it would
                # share MPS's one command buffer with whatever thread holds it.
                ids = self._producing_ids(anchor, raw=raw)
                result = run(ReplayForward(
                    ids=ids,
                    prompt_length=int(ids.shape[1]) - anchor.raw_index,
                    raw_ids=anchor.raw_ids,
                    raw_index=anchor.raw_index,
                ))
        return (
            ReplayedToken(
                node_id=node_id,
                raw_index=anchor.raw_index,
                token_id=anchor.token_id,
                token_text=str(self._tokenizer.decode([anchor.token_id])),
                steering=steering,
            ),
            result,
        )

    def _producing_ids(self, anchor: TokenAnchor, *, raw: bool) -> torch.Tensor:
        """The token stream the producing forward saw: the node's prompt
        render plus the raw decode prefix before the token."""
        node = anchor.node
        if raw:
            prompt_ids = self._renderer.prepare_input(
                "", raw=True, parent_node_id=node.parent_id,
            )
        else:
            # Continue-mode rebuild: the history walk to the parent carries
            # every prior turn with its stamped label, so the render is
            # byte-identical to the one that produced the node, for any seat.
            recipe = node.recipe
            thinking = recipe.thinking if recipe is not None else None
            supports_thinking = self._traits.supports_thinking
            prompt_ids = self._renderer.prepare_input(
                None,
                thinking=(
                    supports_thinking if thinking is None
                    else bool(thinking) and supports_thinking
                ),
                parent_node_id=node.parent_id,
                system_prompt=(
                    recipe.system_prompt if recipe is not None
                    else INHERIT_SYSTEM_PROMPT
                ),
                user_role=node.role_label if node.role == "user" else None,
                assistant_role=(
                    node.role_label if node.role == "assistant" else None
                ),
                gen_seat=node.role,
            )
        if anchor.raw_index == 0:
            return prompt_ids
        prefix = torch.tensor(
            [anchor.prefix], dtype=prompt_ids.dtype, device=prompt_ids.device,
        )
        return torch.cat([prompt_ids, prefix], dim=1)


__all__ = [
    "ReplayForward",
    "ReplaySegment",
    "ReplayWindow",
    "TokenReplayer",
    "capture_pooled",
    "drive_segments",
    "replay_segments",
]
