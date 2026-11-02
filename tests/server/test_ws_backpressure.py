"""A slow WebSocket reader never slows or cancels generation.

Past the outbound bound the connection drops its token and tree frames,
generation goes on into the tree, and once the client drains it receives one
``resync`` frame; the reply finishes with its own reason and the client
rebuilds what it missed from the tree.  Stop and a real disconnect still
cancel.  The reader is stalled at the ASGI ``send`` (``_socket``'s gate), and
the decode is paused deterministically inside the token callback.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, cast

import pytest
from starlette.types import Message

from drowse.server import create_app
from drowse.server import ws_stream
from drowse.server.tree_models import tree_to_json
from drowse.session.session import DrowseSession

STREAM_PATH = "/drowse/v1/sessions/default/stream"
BOUND = 8
TOKENS = 40


@asynccontextmanager
async def _socket(app: Any, gate: asyncio.Event) -> AsyncIterator[tuple[Any, Any, asyncio.Task[Any]]]:
    incoming: asyncio.Queue[Message] = asyncio.Queue()
    outgoing: asyncio.Queue[Message] = asyncio.Queue()
    scope = {
        "type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "scheme": "ws", "path": STREAM_PATH, "raw_path": STREAM_PATH.encode(),
        "root_path": "", "query_string": b"", "headers": [(b"host", b"localhost")],
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 80),
        "subprotocols": [], "state": {},
    }
    incoming.put_nowait({"type": "websocket.connect"})

    async def send(message: Message) -> None:
        if message["type"] == "websocket.send":
            await gate.wait()
        await outgoing.put(message)

    task = asyncio.create_task(app(scope, incoming.get, send))
    assert (await asyncio.wait_for(outgoing.get(), 2))["type"] == "websocket.accept"
    try:
        yield incoming, outgoing, task
    finally:
        gate.set()
        incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)


class _Pause:
    """Hold the decode inside the WS token callback at token ``at``."""

    def __init__(self, at: int | None) -> None:
        self.at = at
        self.count = 0
        self.paused = threading.Event()
        self.release = threading.Event()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        build = ws_stream.build_token_event

        def paced(session: Any, holder: Any, event: Any) -> Any:
            frame = build(session, holder, event)
            if self.count == self.at:
                self.paused.set()
                assert self.release.wait(10)
            self.count += 1
            return frame

        monkeypatch.setattr(ws_stream, "build_token_event", paced)


def _generate(**extra: Any) -> Message:
    body = {
        "type": "generate", "input": "Tell me a long story.", "stateless": False,
        "sampling": {"max_tokens": TOKENS, "temperature": 0}, **extra,
    }
    return {"type": "websocket.receive", "text": json.dumps(body)}


def _reply(session: Any) -> Any:
    nodes = [node for node in session.tree.nodes.values() if node.role == "assistant"]
    return nodes[-1] if nodes else None


async def _until(predicate: Any, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def _frames_until(outgoing: Any, kind: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    async with asyncio.timeout(10):
        while True:
            message = await outgoing.get()
            assert message["type"] == "websocket.send", message
            frame = json.loads(message["text"])
            frames.append(frame)
            if frame["type"] == kind:
                return frames


def _rebuild(frames: list[dict[str, Any]], snapshot: dict[str, Any] | None, node_id: str) -> list[dict[str, Any]]:
    """What a client holds: live rows, the tree's rows past them at resync, then deduped live rows."""
    rows: list[dict[str, Any]] = []

    def last() -> int:
        return rows[-1]["raw_index"] if rows else -1

    for frame in frames:
        if frame["type"] == "resync":
            assert snapshot is not None
            node = next(node for node in snapshot["nodes"] if node["id"] == node_id)
            rows.extend(row for row in node["tokens"] if row["raw_index"] > last())
        elif frame["type"] == "token" and frame["raw_index"] > last():
            rows.append({"raw_index": frame["raw_index"], "text": frame["text"]})
    return rows


def _rows(node: Any) -> list[tuple[int, str]]:
    return [(row["raw_index"], row["text"]) for row in node.tokens or []]


def test_stalled_reader_lets_the_reply_finish(tiny_session: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader that reads nothing for a whole reply: it finishes, unlocked, then resyncs once."""
    monkeypatch.setattr(ws_stream, "MAX_WS_OUTBOUND_EVENTS", BOUND)
    session = cast(DrowseSession, tiny_session)

    async def scenario() -> None:
        app = create_app(session)
        gate = asyncio.Event()
        async with _socket(app, gate) as (incoming, outgoing, _task):
            incoming.put_nowait(_generate())
            await _until(lambda: (node := _reply(session)) is not None and node.finish_reason is not None)
            await _until(lambda: not app.state.session_lock.locked())
            node = _reply(session)
            assert node.finish_reason in ("length", "stop")
            assert outgoing.empty(), "nothing reached the stalled reader"
            gate.set()
            frames = await _frames_until(outgoing, "done")
        kinds = [frame["type"] for frame in frames]
        assert kinds.count("resync") == 1
        resync = frames[kinds.index("resync")]
        assert resync["dropped"] > 0
        # The bound held: at most BOUND droppable frames waited for the reader.
        assert sum(kind in ("token", "tree_mutated") for kind in kinds[:kinds.index("resync")]) <= BOUND
        assert kinds.index("resync") < kinds.index("done")
        done = frames[-1]
        assert done["result"]["finish_reason"] == node.finish_reason
        assert done["result"]["text"] == node.text
        snapshot = cast(dict[str, Any], tree_to_json(session))
        rebuilt = _rebuild(frames, snapshot, node.id)
        assert [(row["raw_index"], row["text"]) for row in rebuilt] == _rows(node)
        assert "".join(row["text"] for row in rebuilt) == node.text

    asyncio.run(scenario())


def test_reader_that_recovers_mid_reply_resyncs_then_streams_live(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_stream, "MAX_WS_OUTBOUND_EVENTS", BOUND)
    pause = _Pause(20)
    pause.install(monkeypatch)
    session = cast(DrowseSession, tiny_session)

    async def scenario() -> None:
        app = create_app(session)
        gate = asyncio.Event()
        async with _socket(app, gate) as (incoming, outgoing, _task):
            incoming.put_nowait(_generate())
            assert await asyncio.to_thread(pause.paused.wait, 10)
            gate.set()
            before = await _frames_until(outgoing, "resync")
            # The client fetches the tree on resync, mid-reply.
            assert app.state.session_lock.locked()
            snapshot = cast(dict[str, Any], tree_to_json(session))
            pause.release.set()
            after = await _frames_until(outgoing, "done")
        node = _reply(session)
        assert node.finish_reason in ("length", "stop")
        live = [frame for frame in after if frame["type"] == "token"]
        assert live, "live token frames follow the resync"
        assert "resync" not in [frame["type"] for frame in after]
        rebuilt = _rebuild(before + after, snapshot, node.id)
        indices = [row["raw_index"] for row in rebuilt]
        assert indices == sorted(set(indices)), "no token is duplicated"
        assert [(row["raw_index"], row["text"]) for row in rebuilt] == _rows(node)

    asyncio.run(scenario())


def test_stop_during_a_lag_cancels_at_once(tiny_session: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_stream, "MAX_WS_OUTBOUND_EVENTS", BOUND)
    pause = _Pause(20)
    pause.install(monkeypatch)
    session = cast(DrowseSession, tiny_session)

    async def scenario() -> None:
        app = create_app(session)
        gate = asyncio.Event()
        async with _socket(app, gate) as (incoming, outgoing, _task):
            incoming.put_nowait(_generate(request_id="lagging-stop"))
            assert await asyncio.to_thread(pause.paused.wait, 10)
            incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": "stop"})})
            await asyncio.sleep(0.05)
            pause.release.set()
            # The stop lands while the reader still reads nothing.
            await _until(lambda: _reply(session).finish_reason is not None)
            assert _reply(session).finish_reason == "cancelled"
            await _until(lambda: not app.state.session_lock.locked())
            assert outgoing.empty()
            gate.set()
            frames = await _frames_until(outgoing, "request_complete")
        kinds = [frame["type"] for frame in frames]
        assert "resync" in kinds
        done = next(frame for frame in frames if frame["type"] == "done")
        assert done["result"]["finish_reason"] == "cancelled"
        assert frames[-1]["state"] == "cancelled"
        assert app.state.operations.request_status("lagging-stop")["state"] == "cancelled"

    asyncio.run(scenario())


def test_disconnect_during_a_lag_cancels(tiny_session: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_stream, "MAX_WS_OUTBOUND_EVENTS", BOUND)
    pause = _Pause(20)
    pause.install(monkeypatch)
    session = cast(DrowseSession, tiny_session)

    async def scenario() -> None:
        app = create_app(session)
        gate = asyncio.Event()
        async with _socket(app, gate) as (incoming, _outgoing, task):
            incoming.put_nowait(_generate(request_id="lagging-gone"))
            assert await asyncio.to_thread(pause.paused.wait, 10)
            incoming.put_nowait({"type": "websocket.disconnect", "code": 1001})
            await asyncio.sleep(0.05)
            pause.release.set()
            await asyncio.wait_for(task, 5)
            assert _reply(session).finish_reason == "cancelled"
            assert not app.state.session_lock.locked()
        assert app.state.operations.request_status("lagging-gone")["state"] == "cancelled"

    asyncio.run(scenario())


def test_outbound_queue_lags_then_resyncs_once() -> None:
    """Producers on another thread never block; controls are never dropped."""
    from drowse.server.ws_stream import MAX_WS_OUTBOUND_EVENTS, _OutboundQueue

    async def scenario() -> None:
        queue = _OutboundQueue()

        def produce() -> None:
            for i in range(10_000):
                queue.put({"type": "token", "i": i}, droppable=True)
            queue.put({"type": "done"})

        await asyncio.to_thread(produce)
        await asyncio.sleep(0)
        got = [await queue.get() for _ in range(MAX_WS_OUTBOUND_EVENTS)]
        assert [frame["i"] for frame in got] == list(range(MAX_WS_OUTBOUND_EVENTS))
        assert await queue.get() == {"type": "resync", "dropped": 10_000 - MAX_WS_OUTBOUND_EVENTS}
        assert await queue.get() == {"type": "done"}
        # Live again after the resync.
        queue.put({"type": "token", "i": -1}, droppable=True)
        await asyncio.sleep(0)
        assert await queue.get() == {"type": "token", "i": -1}
        flushed = asyncio.create_task(queue.flushed())
        await asyncio.sleep(0)
        assert not flushed.done(), "frames taken but not yet written"
        for _ in range(MAX_WS_OUTBOUND_EVENTS + 3):
            queue.sent()
        await asyncio.wait_for(flushed, 1)
        queue.close()
        queue.put({"type": "done"})
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), 0.01)

    asyncio.run(scenario())
