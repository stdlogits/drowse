"""drowse.loom.ops: fork/prefill plans and the generation turn opener (CPU, no model)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, SamplingConfig
from drowse.loom.errors import InvalidNodeOperationError, MutationDuringGenerationError
from drowse.loom.ops import (
    MessageTurn,
    begin_generation_turn,
    plan_fork,
    plan_prefill,
    resolve_token_anchor,
    split_message_list,
)
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomTree
from drowse.model.scene import SceneThinkingUnsupportedError


class _Tok:
    """Maps known phrases to fixed ids; anything else to one id per char."""

    def __init__(self, table: dict[str, list[int]] | None = None) -> None:
        self.table = table or {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.table.get(text, [ord(c) for c in text]))


def _generated(
    tree: LoomTree,
    *,
    seat: str = "assistant",
    label: str | None = "guide",
    recipe: Recipe | None = None,
) -> tuple[str, str]:
    parent = tree.add_user_turn("hi")
    node = tree.begin_assistant(
        parent,
        recipe=recipe if recipe is not None else Recipe(
            steering="0.2 calm",
            sampling=SamplingConfig(max_tokens=4, seed=9),
            thinking=False,
            system_prompt="be terse",
        ),
        role_label=label,
        seat=seat,
    )
    tree.finalize_assistant(node, text="abc", raw_token_ids=[10, 11, 12])
    return parent, node


def test_fork_forces_a_multitoken_replacement_under_the_source_recipe() -> None:
    tree = LoomTree()
    parent, node = _generated(tree)
    tok = _Tok({"replacement phrase": [41, 42]})

    turn = plan_fork(
        tree, tok, node, 1, replacement_text="replacement phrase",
        default_max_tokens=16,
    )
    assert turn.input is None
    assert turn.forced_prefix == [10, 41, 42]
    assert turn.parent_node_id == parent
    assert turn.gen_seat == "assistant"
    assert turn.steering == "0.2 calm"
    assert turn.thinking is False
    assert turn.system_prompt == "be terse"
    assert turn.sampling.max_tokens == 7  # forced span + the recipe's 4
    assert turn.sampling.seed == 9
    assert turn.sampling.assistant_role == "guide"

    reseeded = plan_fork(tree, tok, node, 1, 11, seed=123, default_max_tokens=16)
    assert reseeded.forced_prefix == [10, 11]
    assert reseeded.sampling.seed == 123
    assert reseeded.generate_kwargs()["forced_prefix"] == [10, 11]


def test_fork_of_a_user_seat_node_stays_in_the_user_seat() -> None:
    tree = LoomTree()
    _parent, node = _generated(
        tree, seat="user", label="narrator", recipe=Recipe(),
    )
    turn = plan_fork(tree, _Tok(), node, 0, 7, default_max_tokens=16)
    assert turn.gen_seat == "user"
    assert turn.sampling.user_role == "narrator"
    assert turn.sampling.max_tokens == 17  # no recipe budget: session default
    assert turn.system_prompt is INHERIT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"raw_index": 3, "alt_token_id": 1}, "out of range"),
        ({"raw_index": 0}, "exactly one"),
        ({"raw_index": 0, "alt_token_id": 1, "replacement_text": "x"}, "exactly one"),
        ({"raw_index": 0, "replacement_text": "gone"}, "empty"),
    ],
)
def test_fork_refuses_bad_targets(kwargs: dict[str, Any], match: str) -> None:
    tree = LoomTree()
    _parent, node = _generated(tree)
    with pytest.raises(InvalidNodeOperationError, match=match):
        plan_fork(
            tree, _Tok({"gone": []}), node, kwargs.pop("raw_index"),
            default_max_tokens=8, **kwargs,
        )


def test_fork_refuses_a_node_without_raw_ids() -> None:
    tree = LoomTree()
    user = tree.add_user_turn("transcript-loaded")
    with pytest.raises(InvalidNodeOperationError, match="no raw token record"):
        plan_fork(tree, _Tok(), user, 0, 1, default_max_tokens=8)


def test_a_token_anchor_addresses_one_decode_step() -> None:
    tree = LoomTree()
    _parent, node = _generated(tree)
    anchor = resolve_token_anchor(tree, node, 2, label="probe")
    assert anchor.node.id == node
    assert anchor.token_id == 12
    assert anchor.prefix == [10, 11]
    # The anchor snapshots the decode record: a later edit of the node
    # cannot shift what it addresses.
    tree.get(node).raw_token_ids = None
    assert anchor.token_id == 12
    with pytest.raises(InvalidNodeOperationError, match="probe: .*no raw token record"):
        resolve_token_anchor(tree, node, 0, label="probe")


def test_fork_refuses_a_node_that_is_not_a_turn() -> None:
    """A fork addresses its token through the same anchor as the token
    replays, so the system root is refused as a non-turn."""
    tree = LoomTree()
    assert tree.root_id is not None
    with pytest.raises(InvalidNodeOperationError, match="fork_from_token: .*only a turn"):
        plan_fork(tree, _Tok(), tree.root_id, 0, 1, default_max_tokens=8)


def test_prefill_opens_the_reply_with_the_text_and_skips_thinking() -> None:
    tree = LoomTree()
    root = tree.root_id
    user = tree.add_user_turn("question")
    turn = plan_prefill(
        tree, _Tok({"Sure,": [5, 6]}), user, "Sure,",
        steering="0.3 calm", sampling=SamplingConfig(max_tokens=10, seed=1),
        default_max_tokens=64,
    )
    assert turn.input == "question"
    assert turn.parent_node_id == root  # dedup re-uses the user turn
    assert turn.forced_prefix == [5, 6]
    assert turn.thinking is False
    assert turn.steering == "0.3 calm"
    assert turn.sampling == SamplingConfig(max_tokens=12, seed=1)
    assert turn.gen_seat == "assistant"
    assert turn.system_prompt is INHERIT_SYSTEM_PROMPT

    default_budget = plan_prefill(
        tree, _Tok({"Sure,": [5, 6]}), user, "Sure,", default_max_tokens=64,
    )
    assert default_budget.sampling.max_tokens == 66


def test_prefill_refuses_non_user_nodes_and_empty_text() -> None:
    tree = LoomTree()
    _parent, node = _generated(tree)
    with pytest.raises(InvalidNodeOperationError, match="not a user node"):
        plan_prefill(tree, _Tok(), node, "x", default_max_tokens=8)
    user = tree.add_user_turn("q")
    with pytest.raises(InvalidNodeOperationError, match="empty"):
        plan_prefill(tree, _Tok({"x": []}), user, "x", default_max_tokens=8)


def test_generation_turn_commits_the_user_turn_and_reserves_it() -> None:
    tree = LoomTree()
    node = begin_generation_turn(
        tree, "hello", raw=False, parent_node_id=None,
        sampling=SamplingConfig(user_role="captain", assistant_role="deer"),
        recipe=Recipe(),
    )
    user = tree.nodes[node].parent_id
    assert user is not None
    assert tree.nodes[user].role_label == "captain"
    assert tree.nodes[node].role_label == "deer"
    assert tree.reservation == user
    with pytest.raises(MutationDuringGenerationError):
        tree.edit(user, "changed mid-generation")


def test_generation_turn_refuses_a_user_under_user_send() -> None:
    tree = LoomTree()
    tree.add_user_turn("waiting")
    with pytest.raises(InvalidNodeOperationError, match="cannot send"):
        begin_generation_turn(
            tree, "again", raw=False, parent_node_id=None, sampling=None,
            recipe=Recipe(),
        )
    assert tree.reservation is None


def test_generation_turn_user_seat_and_flat_paths() -> None:
    tree = LoomTree()
    user = tree.add_user_turn("waiting")
    # A user-seat generation may hang under a user node (scene seating).
    node = begin_generation_turn(
        tree, None, raw=False, parent_node_id=user,
        sampling=SamplingConfig(user_role="narrator"), recipe=Recipe(),
        gen_seat="user",
    )
    assert tree.nodes[node].role == "user"
    assert tree.nodes[node].role_label == "narrator"

    # Flat mode: no labels, an empty span just extends the parent.
    flat = LoomTree()
    span = flat.add_user_turn("once")
    node = begin_generation_turn(
        flat, "", raw=True, parent_node_id=span,
        sampling=SamplingConfig(assistant_role="deer"), recipe=Recipe(),
    )
    assert flat.nodes[node].parent_id == span
    assert flat.nodes[node].role_label is None
    assert flat.reservation == span


_SCENE = SimpleNamespace(think_open="<think>")  # a scene grammar: free seating, thinking


def _split(messages: list[Any], **kwargs: Any) -> tuple[list[MessageTurn], Any]:
    call: dict[str, Any] = {
        "system_prompt": INHERIT_SYSTEM_PROMPT, "parent_node_id": None, "raw": False,
        "tokenizer": _Tok(), "grammar": _SCENE,
    }
    return split_message_list(messages, **{**call, **kwargs})


def _record(tree: LoomTree, messages: list[Any], *, grammar: Any = _SCENE) -> str:
    turns, system = _split(messages, grammar=grammar)
    return begin_generation_turn(
        tree, turns, raw=False, parent_node_id=None, sampling=None,
        recipe=Recipe(system_prompt=None if system is INHERIT_SYSTEM_PROMPT else system),
    )


def _reply(tree: LoomTree, node: str, text: str = "r") -> None:
    tree.reserve(None)
    tree.finalize_assistant(node, text=text, raw_token_ids=[1])


def test_message_list_splits_its_system_message_into_the_system_prompt() -> None:
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi there", "label": "captain"},
    ]
    turns, system = _split(messages)
    assert turns == [MessageTurn("user", "Hi there", "captain", None, tuple(map(ord, "Hi there")))]
    assert system == "Be brief."
    assert messages[0]["role"] == "system", "the caller's list is left as it was"
    # The same system prompt passed twice is one prompt; a different one is ambiguous.
    assert _split(messages, system_prompt="Be brief.")[1] == "Be brief."
    with pytest.raises(InvalidNodeOperationError, match="given twice"):
        _split(messages, system_prompt="Be long.")
    # Without a system message the call's own system prompt stands.
    assert _split(messages[1:], system_prompt=None) == (turns, None)


@pytest.mark.parametrize(
    ("messages", "kwargs", "match"),
    [
        ([{"role": "user", "content": "hi"}], {"parent_node_id": "n1"}, "whole conversation"),
        ([{"role": "user", "content": "hi"}], {"raw": True}, "raw completion"),
        ([{"role": "user", "content": 3}], {}, "string content"),
        (["hi"], {}, "string content"),
        ([{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}], {}, "role 'system'"),
        ([{"role": "tool", "content": "{}"}], {}, "role 'tool'"),
        ([{"role": "user", "content": "kept"}, {"role": "assistant", "content": ""}], {}, "empty content"),
        ([{"role": "user", "content": "x"}], {"tokenizer": _Tok({"x": []})}, "tokenized to an empty"),
        ([{"role": "user", "content": "hi", "label": 3}], {}, "label must be a string"),
        ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "A", "label": "Not a slug"}], {}, "message 1"),
        ([{"role": "user", "content": "hi", "thinking": ["no"]}], {}, "thinking must be a string"),
        # A legacy chat template renders only alternating turns from a user one.
        ([{"role": "assistant", "content": "first"}], {"grammar": None}, "alternating"),
        ([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], {"grammar": None}, "alternating"),
        (
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
             {"role": "assistant", "content": "c"}],
            {"grammar": None},
            "alternating",
        ),
    ],
)
def test_message_list_refuses_what_it_cannot_record(
    messages: list[Any], kwargs: dict[str, Any], match: str,
) -> None:
    with pytest.raises(InvalidNodeOperationError, match=match):
        _split(messages, **kwargs)


def test_message_list_refuses_thinking_the_family_cannot_render() -> None:
    with pytest.raises(SceneThinkingUnsupportedError):
        _split([{"role": "user", "content": "hi", "thinking": "hmm"}], grammar=None)
    turns, _ = _split([{"role": "user", "content": "hi", "thinking": "hmm"}])
    assert turns[0].thinking == "hmm"


def test_a_refused_message_list_records_nothing() -> None:
    # Validation covers the whole list before the tree changes: a bad last
    # turn must not leave the earlier ones committed and active.
    tree = LoomTree()
    rev, active = tree.rev, tree.active_node_id
    with pytest.raises(InvalidNodeOperationError):
        _record(tree, [{"role": "user", "content": "kept"}, {"role": "assistant", "content": ""}])
    assert (tree.rev, tree.active_node_id, len(tree.nodes)) == (rev, active, 1)


def test_a_message_list_is_recorded_and_reserved_as_one() -> None:
    # An edit racing the recording waits for the tree's lock, then meets the
    # reservation, so the reply answers the turns the list sent.
    import threading

    tree = LoomTree()
    first = _record(tree, [{"role": "user", "content": "Q"}])
    _reply(tree, first, "A")
    answer = first
    record = tree.add_assistant_turn
    outcome: list[BaseException | None] = []

    def edit_answer() -> None:
        try:
            tree.edit(answer, "CHANGED")
            outcome.append(None)
        except BaseException as error:  # noqa: BLE001 - the test reads it
            outcome.append(error)

    def racing_record(*args: Any, **kwargs: Any) -> str:
        node = record(*args, **kwargs)
        racer = threading.Thread(target=edit_answer)
        racer.start()
        racer.join(timeout=0.2)
        assert racer.is_alive(), "the edit waits for the recording's hold of the lock"
        threads.append(racer)
        return node

    threads: list[threading.Thread] = []
    tree.add_assistant_turn = racing_record  # type: ignore[method-assign]
    reply = _record(tree, [
        {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Q2"},
    ])
    threads[0].join(timeout=5)
    assert isinstance(outcome[0], MutationDuringGenerationError)
    path = [tree.nodes[n].text for n in tree.ancestors_of(reply) if n != tree.root_id]
    assert "CHANGED" not in path and tree.nodes[answer].text == "A"


def test_generation_turn_records_a_message_list_from_the_root() -> None:
    tree = LoomTree()
    earlier = tree.add_user_turn("an earlier conversation")
    messages = [
        {"role": "user", "content": "Hi there", "label": "captain"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "And again"},
    ]
    node = _record(tree, messages, grammar=None)
    path = tree.path_to(node)
    assert path[0].id == tree.root_id and earlier not in {n.id for n in path}
    assert [(n.role, n.text, n.role_label, n.thinking_text) for n in path[1:-1]] == [
        ("user", "Hi there", "captain", None),
        ("assistant", "Hello.", None, None),
        ("user", "And again", None, None),
    ]
    assert path[2].raw_token_ids == list(map(ord, "Hello.")), "an authored turn stays forkable"
    assert tree.messages_for(path[-2].id) == [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    assert tree.reservation == path[-2].id


def test_a_resent_message_list_reuses_its_whole_path() -> None:
    # Re-sending a multi-turn list (a regen, or each row of a fan or sweep)
    # reuses every matching turn, assistant ones included, so the replies are
    # siblings under one copy of the history.
    tree = LoomTree()
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Q2"},
    ]
    first = _record(tree, messages)
    _reply(tree, first)
    second = _record(tree, messages)
    _reply(tree, second)
    assert tree.nodes[first].parent_id == tree.nodes[second].parent_id
    assert sum(n.text == "A" for n in tree.nodes.values()) == 1
    # A generated reply with the same text is history too: answering it reuses it.
    third = _record(tree, [*messages, {"role": "assistant", "content": "r"},
                           {"role": "user", "content": "Q3"}])
    assert tree.path_to(third)[-3].id in {first, second}
    # A turn still streaming is not history yet, even when its text so far matches.
    _reply(tree, third)
    q3 = tree.path_to(third)[-2].id
    tree.reserve(q3)
    streaming = tree.begin_assistant(q3)
    tree.nodes[streaming].text = "s"
    longer = [*messages, {"role": "assistant", "content": "r"}, {"role": "user", "content": "Q3"}]
    fourth = _record(tree, [*longer, {"role": "assistant", "content": "s"}, {"role": "user", "content": "Q4"}])
    assert tree.path_to(fourth)[-3].id != streaming


def test_message_list_keeps_message_boundaries_and_never_edits_history() -> None:
    # Two adjacent user messages are two turns (as a stateless render has
    # them), and an existing node the list shares is never edited.
    tree = LoomTree()
    first = _record(tree, [{"role": "user", "content": "Hi"}])
    _reply(tree, first, "yo")
    hi = tree.path_to(first)[-2].id
    second = _record(tree, [{"role": "user", "content": "Hi"}, {"role": "user", "content": " more"}])
    assert tree.nodes[hi].text == "Hi" and tree.nodes[first].parent_id == hi
    assert [(n.role, n.text) for n in tree.path_to(second)[1:-1]] == [("user", "Hi"), ("user", " more")]
    assert tree.nodes[first].text == "yo" and tree.nodes[first].recipe is not None


def test_generation_turn_continues_an_existing_message() -> None:
    tree = LoomTree()
    parent, node = _generated(tree)
    recipe = Recipe(steering="0.1 warm")
    returned = begin_generation_turn(
        tree, None, raw=False, parent_node_id=node, sampling=None,
        recipe=recipe, continuation_node_id=node,
    )
    assert returned == node
    assert tree.nodes[node].recipe == recipe
    assert tree.nodes[node].text == ""
    assert tree.reservation == node
    assert tree.nodes[node].parent_id == parent
