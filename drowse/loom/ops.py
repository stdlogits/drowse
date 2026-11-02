"""Conversation operations over a :class:`~drowse.loom.tree.LoomTree`.

The tree owns the data model and its mutation primitives; this module owns
the conversation rules that need a tokenizer or the chat family's scene
grammar on top of it:

- the one authored-append rule (:func:`append_turn`): concatenate onto a
  same-seat, same-label parent, otherwise add a new turn, with one
  validation for both seats;
- the seating checks a legacy (non-scene) chat template imposes
  (:func:`check_user_send_target`) and the committed-thinking check
  (:func:`check_thinking_commit`);
- the loom token anchors: one validation of a ``(node_id, raw_index)``
  address into a generated turn's raw decode record
  (:func:`resolve_token_anchor`, shared by a token fork and the token
  replays) and into an authored turn's captured prompt tokens
  (:func:`resolve_prompt_anchor`, the token replays');
- the forced-prefix plans behind a token fork and an answer prefill
  (:func:`plan_fork`, :func:`plan_prefill`), which the session executes;
- opening the turn a stateful generation streams into
  (:func:`begin_generation_turn`), which also reserves its subtree, and
  recording the conversation a stateful message list answers
  (:func:`split_message_list`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from drowse.generation.sampling import (
    INHERIT_SYSTEM_PROMPT,
    InheritSystemPrompt,
    SamplingConfig,
)
from drowse.loom.errors import InvalidNodeOperationError
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomNode, LoomTree, PromptCapture

if TYPE_CHECKING:
    from drowse.model.scene import TurnGrammar

_SEATS = ("user", "assistant")


def check_user_send_target(tree: LoomTree, parent_node_id: str | None) -> None:
    """Refuse sending a new user turn from a user-role node.

    The resolved parent (``parent_node_id``, else the active node) must not
    itself be a user node: that node is already waiting for an assistant,
    and a user-under-user shape cannot render through a legacy chat
    template.  Regen paths pass the grandparent explicitly so
    ``add_user_turn``'s dedup re-uses the existing user sibling.

    Raises :class:`InvalidNodeOperationError` (HTTP 400).
    """
    target_parent_id = (
        parent_node_id if parent_node_id is not None else tree.active_node_id
    )
    parent = tree.nodes.get(target_parent_id)
    if parent is not None and parent.role == "user":
        raise InvalidNodeOperationError(
            f"cannot send a new user turn from a user node "
            f"({target_parent_id}): the active turn is already "
            f"waiting for an assistant.  Use /regen to redo the "
            f"assistant, or navigate away first."
        )


def check_thinking_commit(grammar: "TurnGrammar | None", thinking: str | None) -> None:
    """Refuse a committed thinking block the renderer can't carry.

    Committed thinking renders through the scene stitcher's family think
    delimiters.  A family without them (gemma), a channel-thinking family
    (gpt-oss) or a template-fallback family (GLM) would raise at the *next*
    render, long after the commit; fail at commit time instead, with the
    renderer's own error.
    """
    if thinking is None:
        return
    if grammar is None or grammar.think_open is None:
        from drowse.model.scene import SceneThinkingUnsupportedError

        raise SceneThinkingUnsupportedError(
            "this model family has no scene-mode think delimiters — "
            "a committed thinking block could not be rendered"
        )


def _encode_nonempty(tokenizer: Any, text: str) -> list[int]:
    ids = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        raise InvalidNodeOperationError(
            f"append_turn: {text!r} tokenized to an empty sequence — "
            f"nothing to append"
        )
    return ids


def append_turn(
    tree: LoomTree,
    tokenizer: Any,
    grammar: "TurnGrammar | None",
    parent_node_id: str | None,
    text: str,
    *,
    seat: str,
    flat: bool = False,
    role_label: str | None = None,
    thinking: str | None = None,
) -> str:
    """Commit one authored span in ``seat`` without generating.

    When the parent (``parent_node_id``, else the active node) already sits
    in ``seat`` under the same ``role_label``, ``text`` is concatenated
    exactly onto it (``thinking`` onto its thinking block) and the node
    becomes an authored message.  Otherwise a new turn lands under the
    parent: a user turn through ``add_user_turn`` (which re-uses an
    identical sibling), an assistant turn as a finished authored node that
    keeps its token ids so it stays forkable.

    Seating is free in scene mode (``grammar`` present) and in ``flat``
    (base-model) mode, where the seat is authorship provenance.  A legacy
    chat template only renders alternating turns, so there a new user turn
    may not hang under a user node and a new assistant turn must hang under
    one.

    Raises :class:`InvalidNodeOperationError` for an unknown seat, empty
    ``text``, text that tokenizes to nothing, or a seating violation, and
    :class:`~drowse.model.scene.SceneThinkingUnsupportedError` when the
    family cannot render committed thinking.  Returns the node id; the
    active node advances to it.
    """
    if seat not in _SEATS:
        raise InvalidNodeOperationError(
            f"append_turn: seat must be 'user' or 'assistant', got {seat!r}"
        )
    if text == "":
        raise InvalidNodeOperationError("append_turn: text must be non-empty")
    check_thinking_commit(grammar, thinking)
    parent_id = parent_node_id or tree.active_node_id
    parent = tree.get(parent_id)
    if parent.role == seat and parent.role_label == role_label:
        return tree.append_authored(
            parent_id,
            text,
            thinking_text=thinking,
            raw_token_ids=_encode_nonempty(tokenizer, parent.text + text),
        )
    if not flat and grammar is None:
        if seat == "user":
            check_user_send_target(tree, parent_id)
        elif parent.role != "user":
            raise InvalidNodeOperationError(
                f"append_turn: {parent_id!r} is a {parent.role} node, not a "
                f"user node — an authored assistant turn hangs off a user "
                f"turn (free seating requires scene mode)"
            )
    raw_token_ids = _encode_nonempty(tokenizer, text)
    if seat == "user":
        return tree.add_user_turn(
            text, parent_id=parent_id, role_label=role_label,
            thinking_text=thinking,
        )
    return tree.add_assistant_turn(
        parent_id,
        text,
        role_label=role_label,
        thinking_text=thinking,
        raw_token_ids=raw_token_ids,
    )


@dataclass(frozen=True)
class TokenAnchor:
    """One decoded token of a loom turn: the node and its raw decode index.

    ``raw_ids`` is the node's raw decode record as it stood when the anchor
    was resolved, so a later edit of the node cannot shift what the anchor
    addresses.
    """

    node: LoomNode
    raw_index: int
    raw_ids: tuple[int, ...]

    @property
    def token_id(self) -> int:
        """The token the decode step at ``raw_index`` produced."""
        return self.raw_ids[self.raw_index]

    @property
    def prefix(self) -> list[int]:
        """The decoded tokens before it (what the producing forward saw)."""
        return list(self.raw_ids[: self.raw_index])


def resolve_token_anchor(
    tree: LoomTree, node_id: str, raw_index: int, *, label: str,
) -> TokenAnchor:
    """Validate a ``(node_id, raw_index)`` token address.

    The node must be a turn (user or assistant seat) with a raw decode
    record, and ``raw_index`` must index into it.  Raises
    :class:`~drowse.loom.errors.UnknownNodeError` for an unknown node and
    :class:`InvalidNodeOperationError` otherwise; ``label`` names the caller
    in the message.
    """
    node = tree.get(node_id)
    if node.role not in _SEATS:
        raise InvalidNodeOperationError(
            f"{label}: {node_id!r} is a {node.role} node — only a turn "
            f"with a decode record has tokens to address"
        )
    raw_ids = node.raw_token_ids
    if not raw_ids:
        raise InvalidNodeOperationError(
            f"{label}: {node_id!r} has no raw token record "
            f"(transcript-loaded node)"
        )
    if not 0 <= raw_index < len(raw_ids):
        raise InvalidNodeOperationError(
            f"{label}: raw_index {raw_index} out of range "
            f"[0, {len(raw_ids)}) for {node_id!r}"
        )
    return TokenAnchor(
        node=node,
        raw_index=int(raw_index),
        raw_ids=tuple(int(token) for token in raw_ids),
    )


@dataclass(frozen=True)
class PromptAnchor:
    """One captured prompt token of an authored turn.

    ``index`` addresses the node's captured rows in render order (thinking
    rows, then visible rows); ``token_ids`` is every captured row's token in
    that order, snapshotted at resolution like a :class:`TokenAnchor`'s raw
    record.  The token sits at render position :attr:`position` of the
    capture's prompt, produced at the position before it.
    """

    node: LoomNode
    index: int
    capture: PromptCapture
    token_ids: tuple[int, ...]
    #: The render position of the token's channel's first token.
    channel_start: int
    #: The index of the channel's first row.
    channel_offset: int
    #: How many of the captured rows are thinking rows (they come first).
    thinking_rows: int

    @property
    def token_id(self) -> int:
        return self.token_ids[self.index]

    @property
    def position(self) -> int:
        """The token's position in the capture's render."""
        return self.channel_start + self.index - self.channel_offset

    @property
    def channel_prefix(self) -> list[int]:
        """The token's channel before it (what follows the channel's head)."""
        return list(self.token_ids[self.channel_offset:self.index])

    @property
    def runs(self) -> tuple[tuple[int, int], ...]:
        """Each captured channel as ``(first_row, render_position)``: the
        thinking rows, then the visible rows, each channel's rows at
        consecutive render positions from its start."""
        out: list[tuple[int, int]] = []
        if self.thinking_rows > 0 and self.capture.thinking_start is not None:
            out.append((0, int(self.capture.thinking_start)))
        if len(self.token_ids) > self.thinking_rows and self.capture.start is not None:
            out.append((self.thinking_rows, int(self.capture.start)))
        return tuple(out)


def resolve_prompt_anchor(
    tree: LoomTree, node_id: str, index: int, *, label: str, produced: bool = True,
) -> PromptAnchor:
    """Validate a ``(node_id, raw_index)`` address into an authored turn's
    captured prompt tokens (``raw_index`` indexes its thinking rows, then
    its visible rows).  Raises
    :class:`~drowse.loom.errors.UnknownNodeError` for an unknown node and
    :class:`InvalidNodeOperationError` for a generated node, an authored
    turn no generation has measured, an out-of-range index or, with
    ``produced`` (the default: the caller reads that token's own forward),
    the render's first token (nothing produces it)."""
    node = tree.get(node_id)
    if node.role not in _SEATS:
        raise InvalidNodeOperationError(
            f"{label}: {node_id!r} is a {node.role} node — only a turn "
            f"has tokens to address"
        )
    if node.recipe is not None:
        raise InvalidNodeOperationError(
            f"{label}: {node_id!r} is a generated turn; its tokens are addressed "
            f"by their raw decode index"
        )
    capture = node.prompt_capture
    if capture is None:
        raise InvalidNodeOperationError(
            f"{label}: {node_id!r} is an authored turn no generation has measured "
            f"yet, so it has no prompt tokens to replay"
        )
    thinking = list(node.thinking_tokens or ())
    visible = list(node.tokens or ())
    token_ids = tuple(int(row["token_id"]) for row in thinking + visible)
    if not 0 <= index < len(token_ids):
        raise InvalidNodeOperationError(
            f"{label}: raw_index {index} out of range [0, {len(token_ids)}) for "
            f"the captured prompt tokens of {node_id!r}"
        )
    in_thinking = index < len(thinking)
    start = capture.thinking_start if in_thinking else capture.start
    if start is None:  # _check_prompt_capture keeps rows and starts together
        raise InvalidNodeOperationError(
            f"{label}: {node_id!r} has rows its prompt capture does not place"
        )
    anchor = PromptAnchor(
        node=node,
        index=int(index),
        capture=capture,
        token_ids=token_ids,
        channel_start=int(start),
        channel_offset=0 if in_thinking else len(thinking),
        thinking_rows=len(thinking),
    )
    if produced and anchor.position < 1:
        raise InvalidNodeOperationError(
            f"{label}: the token at raw_index {index} of {node_id!r} opens the "
            f"prompt, so no position produced it"
        )
    return anchor


@dataclass(frozen=True)
class ForcedTurn:
    """A generation whose first decode steps replay ``forced_prefix``.

    The plan a token fork or an answer prefill hands to the session's
    generation core: ``input`` is the new user text (``None`` continues the
    path under ``parent_node_id``) and the token budget already covers the
    forced span.
    """

    input: str | None
    parent_node_id: str | None
    forced_prefix: list[int]
    sampling: SamplingConfig
    steering: Any = None  # str | Steering | None
    thinking: bool | None = None
    gen_seat: str = "assistant"
    system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT

    def generate_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the generation core (all but ``input``)."""
        return {
            "steering": self.steering,
            "sampling": self.sampling,
            "thinking": self.thinking,
            "parent_node_id": self.parent_node_id,
            "forced_prefix": self.forced_prefix,
            "gen_seat": self.gen_seat,
            "system_prompt": self.system_prompt,
        }


def plan_fork(
    tree: LoomTree,
    tokenizer: Any,
    node_id: str,
    raw_index: int,
    alt_token_id: int | None = None,
    *,
    replacement_text: str | None = None,
    seed: int | None = None,
    default_max_tokens: int,
) -> ForcedTurn:
    """Plan regenerating ``node_id`` as a sibling with one token replaced.

    The fork replays the node's raw decode sequence up to ``raw_index``,
    forces either ``alt_token_id`` or the tokenization of
    ``replacement_text`` there, then samples freely.  It reuses the node's
    steering, sampling, thinking and system-prompt recipe; by default the
    original seed re-applies and the sampler still draws every forced step,
    keeping the RNG stream aligned through the fork point.  ``seed`` asks
    for a new ending instead.  The sibling lands under the same parent, in
    the same seat and under the same cast label as the source node.

    Raises :class:`InvalidNodeOperationError` when the address is not a
    token (see :func:`resolve_token_anchor`), the node has no parent, or not
    exactly one replacement is given.
    """
    anchor = resolve_token_anchor(tree, node_id, raw_index, label="fork_from_token")
    node = anchor.node
    if node.parent_id is None:
        raise InvalidNodeOperationError(
            f"fork_from_token: {node_id!r} has no parent to anchor "
            f"the forked sibling under"
        )
    if (alt_token_id is None) == (replacement_text is None):
        raise InvalidNodeOperationError(
            "fork_from_token: provide exactly one alternative token id "
            "or replacement text"
        )
    if replacement_text is not None:
        replacement_ids = [
            int(t) for t in tokenizer.encode(
                replacement_text, add_special_tokens=False,
            )
        ]
        if not replacement_ids:
            raise InvalidNodeOperationError(
                "fork_from_token: replacement text tokenized to an empty "
                "sequence"
            )
    else:
        assert alt_token_id is not None
        replacement_ids = [int(alt_token_id)]
    forced_prefix = anchor.prefix + replacement_ids

    recipe = node.recipe
    base = (
        recipe.sampling
        if recipe is not None and recipe.sampling is not None
        else SamplingConfig()
    )
    # The forced span burns ``len(forced_prefix)`` decode steps before the
    # continuation starts; extend the budget so a deep fork keeps the
    # original continuation headroom.
    sampling = replace(
        base,
        max_tokens=len(forced_prefix) + (base.max_tokens or default_max_tokens),
        **({"seed": seed} if seed is not None else {}),
    )
    # The sibling carries the source node's label on its own seat; the
    # tree's stamped label wins over whatever the recipe's sampling held.
    if node.role == "user":
        sampling = replace(sampling, user_role=node.role_label)
    else:
        sampling = replace(sampling, assistant_role=node.role_label)
    return ForcedTurn(
        input=None,
        parent_node_id=node.parent_id,
        forced_prefix=forced_prefix,
        sampling=sampling,
        steering=recipe.steering if recipe is not None else None,
        thinking=recipe.thinking if recipe is not None else None,
        gen_seat=node.role,
        system_prompt=(
            recipe.system_prompt if recipe is not None else INHERIT_SYSTEM_PROMPT
        ),
    )


def plan_prefill(
    tree: LoomTree,
    tokenizer: Any,
    node_id: str,
    text: str,
    *,
    steering: Any = None,
    sampling: SamplingConfig | None = None,
    default_max_tokens: int,
) -> ForcedTurn:
    """Plan an answer prefill: an assistant reply to ``node_id`` opening with ``text``.

    ``text`` is the start of the response, so the thinking channel is
    skipped.  The plan re-sends the user turn under its own parent, where
    ``add_user_turn``'s dedup re-uses it, so the reply lands as a sibling
    assistant node under ``node_id``.  ``steering``/``sampling`` apply as in
    an ordinary generation, with the budget extended by the forced span.

    Raises :class:`InvalidNodeOperationError` when ``node_id`` is not a user
    node or ``text`` tokenizes to nothing.
    """
    node = tree.get(node_id)
    if node.role != "user":
        raise InvalidNodeOperationError(
            f"prefill_assistant: {node_id!r} is a {node.role} node, "
            f"not a user node — prefill seeds the assistant reply "
            f"that hangs off a user turn"
        )
    forced_prefix = list(tokenizer.encode(text, add_special_tokens=False))
    if not forced_prefix:
        raise InvalidNodeOperationError(
            f"prefill_assistant: {text!r} tokenized to an empty "
            f"sequence — nothing to prefill"
        )
    base = sampling if sampling is not None else SamplingConfig()
    return ForcedTurn(
        input=node.text,
        parent_node_id=node.parent_id,
        forced_prefix=forced_prefix,
        sampling=replace(
            base,
            max_tokens=len(forced_prefix) + (base.max_tokens or default_max_tokens),
        ),
        steering=steering,
        thinking=False,
    )


@dataclass(frozen=True)
class MessageTurn:
    """One validated turn of a stateful message list.

    :func:`split_message_list` makes them; :func:`begin_generation_turn`
    records them.  ``token_ids`` is the text's tokenization, which an
    authored assistant turn keeps so it stays forkable.
    """

    seat: str
    text: str
    label: str | None
    thinking: str | None
    token_ids: tuple[int, ...]


def split_message_list(
    messages: list[Any],
    *,
    system_prompt: str | None | InheritSystemPrompt,
    parent_node_id: str | None,
    raw: bool,
    tokenizer: Any,
    grammar: "TurnGrammar | None",
) -> tuple[list[MessageTurn], str | None | InheritSystemPrompt]:
    """A stateful message list's turns and the call's system prompt.

    A message list is a whole conversation, and a stateful generation records
    the one it answers, so the reply's node replays the prompt it was
    generated from: :func:`begin_generation_turn` records the turns from the
    root and the reply renders from them.  A leading ``system`` message is the
    call's system prompt (the reply's recipe keeps it); passing a different
    one as well is ambiguous.  Every other message is a ``user`` or
    ``assistant`` turn with non-empty string ``content`` and optional string
    ``label`` and ``thinking``; each message stays its own turn, as it does in
    a stateless render.  A legacy chat template (no scene ``grammar``) only
    renders alternating turns, so there the list must alternate starting
    with ``user``.  The list renders from the root, so ``parent_node_id``
    does not apply, and raw completion takes text, not messages.

    The whole list is checked here, before anything is recorded, so a
    refused list leaves the tree as it was.

    Raises :class:`InvalidNodeOperationError` (HTTP 400) and
    :class:`~drowse.model.scene.SceneThinkingUnsupportedError` when a turn
    carries thinking the family cannot render.
    """
    if raw:
        raise InvalidNodeOperationError(
            "raw completion takes text, not a message list"
        )
    if parent_node_id is not None:
        raise InvalidNodeOperationError(
            "a message list is the whole conversation and is recorded from "
            "the root; parent_node_id does not apply (send None or a string "
            "to continue a node)"
        )
    messages = list(messages)
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise InvalidNodeOperationError(
                "each message needs a role and string content"
            )
    if messages and messages[0].get("role") == "system":
        system = messages.pop(0)["content"]
        if system_prompt is not INHERIT_SYSTEM_PROMPT and system_prompt != system:
            raise InvalidNodeOperationError(
                "the system prompt is given twice: as the list's system "
                "message and as system_prompt"
            )
        system_prompt = system
    turns: list[MessageTurn] = []
    previous = "system"  # the root
    for index, message in enumerate(messages):
        seat = message.get("role")
        if seat not in _SEATS:
            raise InvalidNodeOperationError(
                "a recorded message list holds one leading system message, "
                f"then user and assistant turns; got role {seat!r}"
            )
        text = message["content"]
        if text == "":
            raise InvalidNodeOperationError(
                f"message {index} ({seat}) has empty content"
            )
        label = message.get("label")
        thinking = message.get("thinking")
        if label is not None and not isinstance(label, str):
            raise InvalidNodeOperationError(f"message {index}: label must be a string")
        if thinking is not None and not isinstance(thinking, str):
            raise InvalidNodeOperationError(f"message {index}: thinking must be a string")
        if grammar is None and (seat == previous or (seat == "assistant" and previous != "user")):
            raise InvalidNodeOperationError(
                f"message {index}: this model's chat template renders only "
                f"alternating turns starting with user, and a {seat} turn "
                f"cannot follow a {previous} one (free seating requires "
                f"scene mode)"
            )
        check_thinking_commit(grammar, thinking)
        turns.append(MessageTurn(
            seat=seat, text=text, label=label, thinking=thinking,
            token_ids=tuple(_encode_nonempty(tokenizer, text)),
        ))
        previous = seat
    return turns, system_prompt


def _record_message_turns(tree: LoomTree, turns: list[MessageTurn]) -> str:
    """Record validated turns from the root; returns the last one's id.

    Each turn reuses an identical child of the one before (same seat, text,
    label and thinking, so the render is the same) and otherwise lands as a
    new node, so the list's whole matching path is shared: replies to the
    same list are siblings, and no existing node is edited.
    """
    anchor_id = tree.root_id
    for turn in turns:
        if turn.seat == "user":
            anchor_id = tree.add_user_turn(
                turn.text, parent_id=anchor_id, role_label=turn.label,
                thinking_text=turn.thinking,
            )
        else:
            anchor_id = tree.add_assistant_turn(
                anchor_id, turn.text, role_label=turn.label,
                thinking_text=turn.thinking,
                raw_token_ids=list(turn.token_ids), dedup_existing=True,
            )
    return anchor_id


def begin_generation_turn(
    tree: LoomTree,
    input: Any,
    *,
    raw: bool,
    parent_node_id: str | None,
    sampling: SamplingConfig | None,
    recipe: Recipe,
    gen_seat: str = "assistant",
    continuation_node_id: str | None = None,
) -> str:
    """Open the node a stateful generation streams into; reserve its subtree.

    A ``str`` input commits the new user turn first (under ``sampling``'s
    ``user_role`` label); in flat (``raw``) mode a non-empty span lands as
    its own node and an empty one just extends the parent, with no seating
    check.  ``None`` anchors on the parent directly.  A message list (the
    :class:`MessageTurn` list :func:`split_message_list` returns) is
    recorded from the root, reusing the path it matches, and the reply
    hangs under its last turn.  The generated node
    takes ``gen_seat`` and that seat's label from ``sampling``;
    ``continuation_node_id`` reuses an existing same-seat message whose text
    the caller replays as the forced decode prefix.
    """
    # Per-message labels ride this send's sampling box and are stamped at
    # creation, so a turn keeps its label regardless of later box changes.
    # Flat mode has no chat template, so labels don't apply there.
    user_role = sampling.user_role if sampling is not None and not raw else None
    assistant_role = (
        sampling.assistant_role if sampling is not None and not raw else None
    )
    if isinstance(input, str) and raw:
        anchor_id = (
            tree.add_user_turn(input, parent_id=parent_node_id) if input != ""
            else parent_node_id or tree.active_node_id
        )
    elif isinstance(input, str):
        if gen_seat == "assistant":
            check_user_send_target(tree, parent_node_id)
        anchor_id = tree.add_user_turn(
            input, parent_id=parent_node_id, role_label=user_role,
        )
    elif isinstance(input, list):
        anchor_id = _record_message_turns(tree, input)
    else:
        anchor_id = parent_node_id or tree.active_node_id
    tree.reserve(anchor_id)
    if continuation_node_id is not None:
        return tree.begin_continuation(continuation_node_id, recipe=recipe)
    return tree.begin_assistant(
        anchor_id,
        recipe=recipe,
        role_label=user_role if gen_seat == "user" else assistant_role,
        seat=gen_seat,
    )


__all__ = [
    "ForcedTurn",
    "MessageTurn",
    "PromptAnchor",
    "TokenAnchor",
    "append_turn",
    "begin_generation_turn",
    "check_thinking_commit",
    "check_user_send_target",
    "plan_fork",
    "plan_prefill",
    "resolve_prompt_anchor",
    "resolve_token_anchor",
    "split_message_list",
]
