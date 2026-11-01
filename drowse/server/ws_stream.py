"""Native WebSocket token+probe co-stream (``WS /drowse/v1/sessions/{id}/stream``).

Bidirectional token + per-token probe co-stream — the killer feature of
the native tree API.  The connection runs a single perpetual reader, one
sender over an ordered outbound buffer (:class:`_OutboundQueue`), and a
per-generate-turn worker; see :func:`_ws_stream_generation` for the
concurrency design.  A client that reads slower than the model decodes
never slows or cancels generation: past the buffer's bound its token and
tree frames are dropped, and once it drains it receives one ``resync`` frame
and rebuilds the rest from ``GET /tree``.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections import deque
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, cast

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from anyio import CancelScope

from drowse.events import LoomMutated
from drowse.generation.results import GenerationResult, TokenEvent
from drowse.generation.sampling import SamplingConfig
from drowse.session.session import DrowseSession
from drowse.generation.token_callback import TokenConsumer, TokenConsumerOptions
from drowse.steering.terms import Steering
from drowse.server.errors import describe_error, validation_message
from drowse.server.operations import OperationLedger
from drowse.server.request_helpers import compose_request_steering, validate_logit_bias
from drowse.server.runtime import (
    SINGLE_SESSION_ID,
    SessionLockTimeout,
    finish_worker,
    run_in_thread,
    session_lock,
    session_router,
)
from drowse.server.security import ws_auth_ok
from drowse.server.tree_models import cast_json, node_json
from drowse.server.ws_events import build_token_event
from drowse.server.ws_models import (
    WSGenerateMessage,
    WSSubmitMessage,
    build_input,
    build_sampling_config,
    result_to_json,
)

_logger = logging.getLogger(__name__)

MAX_WS_MESSAGE_BYTES = 1024 * 1024
MAX_WS_PENDING_MESSAGES = 16
MAX_WS_PENDING_BYTES = 4 * MAX_WS_MESSAGE_BYTES
#: Droppable frames (``token``, ``tree_mutated``) one connection may hold
#: queued or pending; past it the connection lags and later gets a ``resync``.
MAX_WS_OUTBOUND_EVENTS = 256
WS_SEND_TIMEOUT_SECONDS = 30
#: The server's WebSocket keepalive (``uvicorn.run(ws_ping_interval=...,
#: ws_ping_timeout=...)``).  Pings still go out (they keep idle connections
#: open through proxies, and a peer that is gone fails the TCP write), but a
#: late pong never closes the socket: a reader slower than generation answers
#: the ping only after the token backlog queued in front of it, which would
#: close a live page about 40 s into a long reply and record the reply
#: ``cancelled``.  ``WS_SEND_TIMEOUT_SECONDS`` is the one dead-peer rule.
WS_PING_INTERVAL_SECONDS = 20.0
WS_PING_TIMEOUT_SECONDS: float | None = None

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]


@dataclass(frozen=True)
class _Stop:
    request_id: str | None = None


@dataclass(frozen=True)
class _Disconnect:
    code: int = 1000
    reason: str = ""


@dataclass(frozen=True)
class _InvalidInbound:
    message: str
    request_id: str | None = None


_Inbound = (
    WSGenerateMessage | WSSubmitMessage | _Stop | _Disconnect
    | _InvalidInbound
)


class _InboundBuffer:
    """One bounded FIFO, with coalesced controls that never consume its budget."""

    def __init__(self) -> None:
        self.pending: deque[tuple[_Inbound, int]] = deque()
        self.pending_bytes = 0
        self.stop: _Stop | None = None
        self.terminal: _Disconnect | None = None
        self.changed = asyncio.Event()
        self.closed = asyncio.Event()

    def put(self, message: _Inbound, size: int = 0) -> bool:
        if self.terminal is not None:
            return False
        if isinstance(message, _Stop):
            if self.stop is None or self.stop.request_id is not None:
                self.stop = message
        else:
            if len(self.pending) >= MAX_WS_PENDING_MESSAGES or self.pending_bytes + size > MAX_WS_PENDING_BYTES:
                return False
            self.pending.append((message, size))
            self.pending_bytes += size
        self.changed.set()
        return True

    def close(self, code: int = 1000, reason: str = "") -> None:
        if self.terminal is None:
            self.terminal = _Disconnect(code, reason)
        self.pending.clear()
        self.pending_bytes = 0
        self.closed.set()
        self.changed.set()

    async def get(self, *, control_only: bool = False) -> _Inbound:
        while True:
            if self.terminal is not None:
                return self.terminal
            if self.stop and (control_only or not self.pending):
                stop = self.stop
                self.stop = None
                return stop
            if self.pending and not control_only:
                message, size = self.pending.popleft()
                self.pending_bytes -= size
                return message
            self.changed.clear()
            await self.changed.wait()


class _OutboundQueue:
    """The connection's one ordered outbound buffer; a slow reader never blocks a producer.

    Every frame the connection sends goes through it and one sender task, so
    wire order is put order: ``started`` precedes the tree delta that opens
    its node, a reply's tree finalization precedes its ``done``, and
    ``request_complete`` comes last.  ``put`` is safe from any thread (the
    generation worker and EventBus callbacks call it) and never waits.

    Droppable frames (``token`` and ``tree_mutated``) count against
    :data:`MAX_WS_OUTBOUND_EVENTS`, including frames still pending as loop
    callbacks.  Past the bound the queue *lags*: it drops every droppable frame
    until the sender has taken all it had queued, then :meth:`get` returns one
    ``{type: "resync", dropped}`` frame and live frames follow.  Generation
    goes on into the tree meanwhile; the client rebuilds what it missed from
    ``GET /tree``.  Control frames (``started``, ``done``, ``error``,
    ``request_complete``) are never dropped: a turn adds a few, and the turn
    loop waits for :meth:`flushed` before it takes the next message, so the
    buffer stays bounded.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[tuple[JSONObject, bool]] = asyncio.Queue()
        self._lock = threading.Lock()
        self._droppable = 0
        self._unsent = 0
        self._dropped = 0
        self._lagging = False
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()

    def put(self, frame: JSONObject, *, droppable: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            if droppable:
                if self._lagging or self._droppable >= MAX_WS_OUTBOUND_EVENTS:
                    self._lagging = True
                    self._dropped += 1
                    return
                self._droppable += 1
            self._unsent += 1
        with suppress(RuntimeError):  # the loop closed under a late producer
            self._loop.call_soon_threadsafe(self._enqueue, frame, droppable)

    def _enqueue(self, frame: JSONObject, droppable: bool) -> None:
        self._idle.clear()
        self._queue.put_nowait((frame, droppable))

    async def get(self) -> JSONObject:
        """The next frame to send: a ``resync`` once a lag has drained."""
        with self._lock:
            if self._lagging and self._droppable == 0:
                self._lagging = False
                dropped, self._dropped = self._dropped, 0
                self._unsent += 1
                return {"type": "resync", "dropped": dropped}
        frame, droppable = await self._queue.get()
        if droppable:
            with self._lock:
                self._droppable -= 1
        return frame

    def sent(self) -> None:
        """Mark the frame the sender last took as written to the socket."""
        with self._lock:
            self._unsent -= 1
            idle = self._unsent == 0
        if idle:
            self._idle.set()

    async def flushed(self) -> None:
        """Wait until every frame put so far has been written."""
        while True:
            with self._lock:
                if self._unsent == 0:
                    return
            self._idle.clear()
            await self._idle.wait()

    def close(self) -> None:
        with self._lock:
            self._closed = True


def _error_frame(
    message: str,
    *,
    code: str = "RequestError",
    status: int = 400,
    node_id: str | None = None,
    sibling_index: int = 0,
) -> JSONObject:
    """Build the one ``{type: "error", …}`` frame shape this stream emits.

    Every error frame carries the same five keys, so a client never has to
    branch on which of them happens to be present.  ``node_id`` is filled only
    once a generation has produced its assistant node.
    """
    return {
        "type": "error",
        "message": message,
        "code": code,
        "status": status,
        "node_id": node_id,
        "sibling_index": sibling_index,
    }


class _WSRequestError(Exception):
    """A request-shape problem to report as an error frame, not a close.

    Raised by the pre-dispatch validators so the field-consistency rules read
    as straight-line checks instead of a ``send_json``/``return`` pair each.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "RequestError",
        status: int = 400,
        node_id: str | None = None,
        sibling_index: int = 0,
    ) -> None:
        super().__init__(message)
        self.frame = _error_frame(
            message, code=code, status=status,
            node_id=node_id, sibling_index=sibling_index,
        )


@dataclass(frozen=True)
class _AuthoredTurn:
    """The turn a ``submit`` commits before its generation runs."""

    text: str
    role: Literal["user", "assistant"]
    thinking: str | None


def register_ws_stream(app: FastAPI) -> None:
    """Mount the bidirectional WebSocket token+probe co-stream and its receipts."""
    session = app.state.session
    ledger: OperationLedger = app.state.operations
    router = session_router()

    @router.get("/requests/{request_id}")
    def request_status(request_id: str) -> dict[str, Any]:
        return ledger.request_status(request_id)

    app.include_router(router)

    @app.websocket("/drowse/v1/sessions/{session_id}/stream")
    async def session_stream(websocket: WebSocket, session_id: str):
        if not ws_auth_ok(websocket):
            await websocket.close(code=1008, reason="unauthorized")
            return
        protocol = "drowse.v1" if "drowse.v1" in websocket.scope.get("subprotocols", []) else None
        if session_id != SINGLE_SESSION_ID:
            await websocket.accept(subprotocol=protocol)
            await websocket.close(code=1008, reason="session not found")
            return
        await websocket.accept(subprotocol=protocol)

        # Single perpetual reader.  ``websocket.receive_json()`` is bound
        # to a per-connection ``recv_in_progress`` flag in the underlying
        # ``websockets`` library; cancelling a pending receive doesn't
        # clear the flag immediately, so any handler that called
        # ``receive_json()`` while another concurrent (even just-cancelled)
        # caller was pending tripped a "cannot call recv while another
        # coroutine is already waiting" RuntimeError.  Routing every
        # incoming frame through one queue lets both the outer dispatch
        # loop and the in-flight generation share the read side without
        # ever overlapping calls into the WS.
        incoming = _InboundBuffer()

        async def _reader():
            try:
                while True:
                    frame = await websocket.receive_text()
                    size = len(frame.encode("utf-8"))
                    if size > MAX_WS_MESSAGE_BYTES:
                        incoming.close(1009, "message too large")
                        return
                    message: _Inbound
                    raw = None
                    try:
                        raw = json.loads(frame)
                        if not isinstance(raw, dict):
                            message = _InvalidInbound("message must be an object")
                        elif raw.get("type") in ("generate", "submit"):
                            message = (
                                WSSubmitMessage(**raw)
                                if raw.get("type") == "submit"
                                else WSGenerateMessage(**raw)
                            )
                        elif raw.get("type") == "stop":
                            request_id = raw.get("request_id")
                            if request_id is not None and (
                                not isinstance(request_id, str) or not request_id or len(request_id) > 128
                            ):
                                message = _InvalidInbound("stop request_id must be a nonempty string of at most 128 characters")
                            else:
                                message = _Stop(request_id)
                        else:
                            message = _InvalidInbound("unknown message type")
                    except ValidationError as exc:
                        request_id = raw.get("request_id") if isinstance(raw, dict) else None
                        message = _InvalidInbound(
                            validation_message(exc),
                            request_id if isinstance(request_id, str) and len(request_id) <= 128 else None,
                        )
                    except (ValueError, RecursionError):
                        message = _InvalidInbound("invalid JSON message")
                    if not incoming.put(message, size):
                        incoming.close(1013, "too many pending messages")
                        return
            except WebSocketDisconnect:
                incoming.close()
            except Exception:
                _logger.exception("native WebSocket reader failed")
                incoming.close(1011, "request failed")

        reader_task = asyncio.create_task(_reader())

        # Every frame this connection sends (token frames from the worker
        # thread, tree deltas from EventBus callbacks on any thread, the
        # turn's control frames) goes through one ordered outbound buffer
        # and one sender task, so no producer ever waits on this client.
        outbound = _OutboundQueue()
        # ``websocket.send_json`` is not safe for concurrent callers; the
        # sender and the terminal error path below share this lock.
        ws_send_lock = asyncio.Lock()

        async def _send_json(payload: JSONObject) -> None:
            async with asyncio.timeout(WS_SEND_TIMEOUT_SECONDS):
                async with ws_send_lock:
                    await websocket.send_json(payload)

        def _on_loom_event(event: object) -> None:
            if not isinstance(event, LoomMutated):
                return
            added_nodes = [node_json(session, nid) for nid in event.added]
            mutated_payload = cast(JSONObject, {
                "type": "tree_mutated",
                "op": event.op,
                "rev": event.rev,
                "added": added_nodes,
                "removed": list(event.removed),
                "updated": [
                    node_json(session, nid)
                    for nid in event.updated
                    if session.tree.has(nid)
                ],
                "active_node_id": event.active_node_id,
            })
            # The roster is derived from the full tree's observed labels as
            # well as explicit configuration.  Inline the small effective
            # roster on every mutation so adds, deletes, and restores reconcile
            # identity without a refetch or provenance inference client-side.
            mutated_payload["cast"] = cast(JSONValue, cast_json(session))
            outbound.put(mutated_payload, droppable=True)
        loom_unsub = session.events.subscribe(_on_loom_event)

        async def _sender() -> None:
            """Write the outbound buffer to the socket for the connection's lifetime.

            A send the client accepts nothing of for ``WS_SEND_TIMEOUT_SECONDS``
            means the peer is gone: the connection closes, which cancels a turn
            still generating like any disconnect.
            """
            try:
                while True:
                    payload = await outbound.get()
                    try:
                        await _send_json(payload)
                    except Exception:
                        incoming.close(1013, "client is not consuming events")
                        return
                    outbound.sent()
            except asyncio.CancelledError:
                return

        sender_task = asyncio.create_task(_sender())

        async def _send(payload: JSONObject) -> None:
            outbound.put(payload)

        def _send_token(payload: JSONObject) -> None:
            outbound.put(payload, droppable=True)

        async def _turn(msg: _Inbound) -> None:
            """Handle one inbound message, then wait until its frames are written.

            Waiting here keeps the buffer bounded: a client that stops reading
            holds at most one turn's control frames, never a growing backlog.
            Generation never waits on it: the session lock is released when
            the last sibling's worker ends.
            """
            if isinstance(msg, (WSGenerateMessage, WSSubmitMessage)):
                await _ws_handle_generate(app, msg, incoming, _send, _send_token)
            elif isinstance(msg, _InvalidInbound):
                frame = _error_frame(msg.message, code="ValidationError", status=400)
                if msg.request_id is not None:
                    frame["request_id"] = msg.request_id
                outbound.put(frame)
                if msg.request_id is not None:
                    outbound.put({
                        "type": "request_complete", "request_id": msg.request_id,
                        "state": "failed", "completed_siblings": 0,
                    })
            await outbound.flushed()

        closed_task = asyncio.create_task(incoming.closed.wait())
        request_task: asyncio.Task[None] | None = None
        try:
            while True:
                msg = await incoming.get()
                if isinstance(msg, _Disconnect):
                    with suppress(Exception):
                        await websocket.close(code=msg.code, reason=msg.reason)
                    return
                if isinstance(msg, _Stop):
                    continue
                request_task = asyncio.create_task(_turn(msg))
                finished, _ = await asyncio.wait(
                    {request_task, closed_task}, return_when=asyncio.FIRST_COMPLETED,
                )
                if closed_task in finished:
                    request_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await finish_worker(request_task)
                else:
                    await request_task
                request_task = None
        except WebSocketDisconnect:
            return
        except Exception as e:
            info = describe_error(e, failure="Request failed")
            try:
                await _send_json(_error_frame(info.message, code=info.code, status=info.status))
            finally:
                with suppress(Exception):
                    await websocket.close(code=1011)
        finally:
            loom_unsub()
            outbound.close()
            with CancelScope(shield=True):
                if request_task is not None:
                    request_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await finish_worker(request_task)
                for task in (sender_task, reader_task, closed_task):
                    task.cancel()
                await asyncio.gather(sender_task, reader_task, closed_task, return_exceptions=True)


def _normalize_submit(
    submit: WSSubmitMessage,
) -> tuple[WSGenerateMessage, _AuthoredTurn | None, bool]:
    """Lower a ``submit`` frame onto the specialist ``generate`` schema.

    ``submit`` is the ONE authored-turn contract on this wire: an explicit
    authored role plus an optional generated role.  Append-only submissions
    take the no-decode commit path; append+generate keeps the authored turn
    for one atomic commit inside the generation worker and generates from
    ``input=None`` once that turn has landed.

    Returns ``(generate_message, authored_turn, commit_only)``.  When
    ``commit_only`` the authored turn is the whole request and the returned
    generate message carries only the commit's context (parent, sampling
    labels, raw); otherwise the authored turn — if any — is committed inside
    the generation worker.  Raises :class:`_WSRequestError` on any
    field-consistency violation.
    """
    if submit.text is None:
        if submit.authored_role is not None:
            raise _WSRequestError("authored_role requires non-empty text")
        if submit.authored_thinking is not None:
            raise _WSRequestError("authored_thinking requires non-empty text")
        authored: _AuthoredTurn | None = None
    else:
        if submit.text == "":
            raise _WSRequestError("submit text must be non-empty when present")
        if submit.authored_role is None:
            raise _WSRequestError("text requires authored_role")
        authored = _AuthoredTurn(
            text=submit.text,
            role=submit.authored_role,
            thinking=submit.authored_thinking,
        )
    if submit.generated_role is None and authored is None:
        raise _WSRequestError("submit requires text or generated_role")

    if submit.generated_role is None:
        assert authored is not None  # guarded by the check above
        return (
            WSGenerateMessage(
                type="generate",
                request_id=submit.request_id,
                parent_node_id=submit.parent_node_id,
                sampling=submit.sampling,
                raw=submit.raw,
            ),
            authored,
            True,
        )
    return (
        WSGenerateMessage(
            type="generate",
            request_id=submit.request_id,
            input=None,
            steering=submit.steering,
            sampling=submit.sampling,
            thinking=submit.thinking,
            **cast(dict[str, Any], {"system_prompt": submit.system_prompt} if "system_prompt" in submit.model_fields_set else {}),
            stateless=False,
            raw=submit.raw,
            parent_node_id=submit.parent_node_id,
            n=submit.n,
            recipe_override=submit.recipe_override,
            generate_seat=submit.generated_role,
        ),
        authored,
        False,
    )


def _commit_authored(
    session: DrowseSession, msg: WSGenerateMessage, authored: _AuthoredTurn,
) -> str:
    """Append the authored turn of a ``submit``; returns its node id.

    Its cast label rides the sampling block (``user_role`` / ``assistant_role``
    by seat).  A raw (flat-buffer) commit has no chat template, so it carries
    neither a label nor committed thinking.
    """
    label = None
    if not msg.raw and msg.sampling is not None:
        label = (msg.sampling.user_role if authored.role == "user" else msg.sampling.assistant_role) or None
    return session.append_turn(
        msg.parent_node_id,
        authored.text,
        role=authored.role,
        raw=msg.raw,
        role_label=label,
        thinking=None if msg.raw else (authored.thinking or None),
    )


async def _ws_handle_commit(
    app: FastAPI,
    msg: WSGenerateMessage,
    authored: _AuthoredTurn,
    send_json: Callable[[JSONObject], Awaitable[None]],
) -> None:
    """Land one authored turn with no decode.

    One tree mutation, one ``started`` (``node_id`` null — the id is only
    known after the append), one ``done`` carrying the new node; no token
    frames.  Seating is free: ``append_turn`` is role-neutral.
    """
    await send_json({
        "type": "started",
        "generation_id": uuid.uuid4().hex[:12],
        "node_id": None,
        "sibling_index": 0,
        "sibling_count": 1,
    })
    try:
        async with session_lock(app):
            new_id = await run_in_thread(_commit_authored, app.state.session, msg, authored)
    except Exception as e:  # noqa: BLE001 - reported as an error frame; the socket stays open
        info = describe_error(e, failure="Commit failed")
        await send_json(_error_frame(info.message, code=info.code, status=info.status))
        return
    await send_json({
        "type": "done",
        "result": {
            "role": authored.role,
            "text": authored.text,
            "node_id": new_id,
            "finish_reason": "stop",
            "mean_logprob": None,
            "mean_surprise": None,
        },
        "node_id": new_id,
        "sibling_index": 0,
        "sibling_count": 1,
    })


async def _ws_handle_generate(
    app: FastAPI,
    msg: WSGenerateMessage | WSSubmitMessage,
    incoming: _InboundBuffer,
    send_json: Callable[[JSONObject], Awaitable[None]],
    send_token: Callable[[JSONObject], None],
) -> None:
    """Run one inbound turn; with a ``request_id``, keep its receipt in the ledger.

    A correlated request is admitted into ``app.state.operations`` first:
    a resubmitted id whose run already finished replays its recorded frames,
    a live or conflicting one is refused, and a full ledger refuses before
    dispatch.  Every frame of an admitted request carries its id and the run
    ends with one ``request_complete`` frame.
    """
    if msg.request_id is None:
        await _ws_dispatch_generate(app, msg, incoming, send_json, send_token)
        return
    request_id = msg.request_id
    ledger: OperationLedger = app.state.operations
    admission = ledger.admit(request_id, msg.model_dump(mode="json", exclude_unset=True), kind="generate")
    if admission == "full":
        await send_json({**_error_frame(
            "Too many active generation requests; wait for one to finish.",
            code="REQUEST_QUEUE_FULL", status=503,
        ), "request_id": request_id})
        await send_json({"type": "request_complete", "request_id": request_id,
                         "state": "failed", "completed_siblings": 0})
        return
    if admission != "new":
        status = ledger.request_status(request_id)
        if admission == "existing" and status["state"] in ("completed", "cancelled", "failed"):
            for frame in status["results"]:
                await send_json(frame)
            if "error" in status:
                await send_json(status["error"])
            await send_json({"type": "request_complete", "request_id": request_id,
                             "state": status["state"], "completed_siblings": len(status["results"])})
        else:
            await send_json({"type": "error", "request_id": request_id,
                             "code": "REQUEST_ID_CONFLICT" if admission == "conflict" else "REQUEST_RECONCILE_REQUIRED",
                             "message": "This request ID was already admitted; query its recorded status before retrying."})
        return
    state = "completed"
    completed = 0

    async def correlated_send(frame: JSONObject) -> None:
        nonlocal state, completed
        if frame.get("type") == "error":
            state = "failed"
        elif frame.get("type") == "done":
            completed += 1
            result = frame.get("result")
            if isinstance(result, dict) and result.get("finish_reason") == "cancelled":
                state = "cancelled"
        correlated = {**frame, "request_id": request_id}
        if frame.get("type") in ("done", "error"):
            ledger.observe(request_id, correlated)
        await send_json(correlated)

    def correlated_token(frame: JSONObject) -> None:
        send_token({**frame, "request_id": request_id})

    def record_only(frame: JSONObject) -> None:
        ledger.observe(request_id, {**frame, "request_id": request_id})

    try:
        stopped = await _ws_dispatch_generate(
            app, msg, incoming, correlated_send, correlated_token, record_only,
        )
        # A stop that raced a reply's natural end leaves that reply's reason,
        # but the request it cut short is still cancelled.
        if stopped and state == "completed":
            state = "cancelled"
    except (WebSocketDisconnect, asyncio.CancelledError):
        ledger.settle_disconnected(request_id, msg.n)
        raise
    except Exception as exc:
        info = describe_error(exc, failure="Generation failed")
        try:
            await correlated_send(_error_frame(info.message, code=info.code, status=info.status))
        except BaseException:
            ledger.interrupt(request_id, "The connection closed before the failure was reported.")
            raise
    terminal_frame: JSONObject = {
        "type": "request_complete", "request_id": request_id,
        "state": state, "completed_siblings": completed,
    }
    ledger.observe(request_id, terminal_frame)
    await send_json(terminal_frame)


async def _ws_dispatch_generate(
    app: FastAPI,
    msg: WSGenerateMessage | WSSubmitMessage,
    incoming: _InboundBuffer,
    send_json: Callable[[JSONObject], Awaitable[None]],
    send_token: Callable[[JSONObject], None],
    record_event: Callable[[JSONObject], None] | None = None,
) -> bool:
    """Validate one inbound turn and dispatch it to its mode handler.

    ``generate``'s own field-consistency rules already ran in the schema
    (``WSGenerateMessage`` model validators, rejected by the reader), so the
    only checks left here are ``submit``'s cross-field rules, which have no
    schema home: they decide *which* generate frame to lower onto
    (:func:`_normalize_submit`).  Then the steering expression is composed
    over the server default and the turn goes to exactly one of two handlers:
    :func:`_ws_handle_commit` (no decode) or :func:`_ws_stream_generation`
    (the token fan-out).  Every rejection here is an error frame on a
    connection that stays open — a 400-grade user mistake must not close the
    socket.  Returns whether a client stop ended the run.
    """
    authored: _AuthoredTurn | None = None
    commit_only = False
    if isinstance(msg, WSSubmitMessage):
        try:
            msg, authored, commit_only = _normalize_submit(msg)
        except _WSRequestError as e:
            await send_json(e.frame)
            return False

    if commit_only:
        assert authored is not None  # set together by ``_normalize_submit``
        await _ws_handle_commit(app, msg, authored, send_json)
        return False

    # FastAPI's exception handlers do not apply to WebSocket routes: a
    # malformed or unresolvable expression is an error frame on a connection
    # that stays open.
    session = app.state.session
    sampling = build_sampling_config(msg.sampling)
    try:
        validate_logit_bias(sampling, session)
        steering = compose_request_steering(msg.steering, app.state.default_steering)
    except Exception as e:  # noqa: BLE001 - reported as an error frame
        info = describe_error(e, failure="Steering failed")
        await send_json(_error_frame(info.message, code=info.code, status=info.status))
        return False

    return await _ws_stream_generation(
        app, msg, steering, sampling, authored,
        incoming, send_json, send_token, record_event,
    )


async def _ws_stream_generation(
    app: FastAPI,
    msg: WSGenerateMessage,
    steering: "Steering | None",
    sampling: SamplingConfig | None,
    authored: _AuthoredTurn | None,
    incoming: _InboundBuffer,
    send_json: Callable[[JSONObject], Awaitable[None]],
    send_token: Callable[[JSONObject], None],
    record_event: Callable[[JSONObject], None] | None = None,
) -> bool:
    """Run one generate turn and stream its started/token/done/error frames.

    Concurrency design: synchronous generation runs in a worker thread under
    its own cancellation scope.  Its ``on_token`` callback hands each token
    frame to ``send_token``, the connection's ordered outbound buffer, which
    never blocks it (past its bound a slow reader's frames are dropped and it
    is told to resync; see :class:`_OutboundQueue`).  This coroutine waits on
    the worker and, beside it, on controls from the shared ``incoming``
    buffer (populated by the connection's single reader task), so an
    in-flight ``{type: "stop"}`` reaches the worker however far behind the
    client's reading is.  The session lock is released once the last
    sibling's worker ends, not once its frames are written.

    **Loom**: ``parent_node_id`` attaches the assistant node under a specific
    tree node; ``n>1`` fans out N siblings serially.  Each sibling produces
    its own ``started`` / token-stream / ``done`` triplet, all tagged with
    the assistant node id.  ``tree_mutated`` frames ride the
    connection-level subscription through the same buffer, so a reply's
    finalization precedes its ``done``.  ``authored`` (from a ``submit``
    that also generates) is committed once inside the worker, under the
    same session lock as the whole fan.  Returns whether a client stop ended
    the run.
    """
    session: DrowseSession = app.state.session
    n = msg.n
    parent_node_id = msg.parent_node_id
    submitted_parent_holder: list[str] = []

    # n>1 derives deterministic per-sibling seeds from the request seed (or
    # fresh entropy); a single stream uses the request seed verbatim.
    from drowse.loom.recipe import derive_seed_schedule
    base_seed = sampling.seed if sampling is not None else None
    seeds: list[int | None]
    seeds = [base_seed] if n == 1 else list(derive_seed_schedule(base_seed, n))

    # The session lock spans the whole N-way fan so concurrent clients
    # serialize FIFO; a timeout is an error frame.
    async with AsyncExitStack() as held:
        try:
            await held.enter_async_context(session_lock(app))
        except SessionLockTimeout as e:
            info = describe_error(e)
            await send_json(_error_frame(info.message, code=info.code, status=info.status))
            return False
        for sibling_idx, seed_i in enumerate(seeds):
            generation_id = uuid.uuid4().hex[:12]

            # Per-sibling sampling override carrying the derived seed.
            if n == 1 and seed_i is None:
                per_sibling_sampling = sampling
            else:
                from dataclasses import replace as _dc_replace
                base_sc = sampling if sampling is not None else SamplingConfig()
                per_sibling_sampling = _dc_replace(base_sc, seed=seed_i)

            cancelled = threading.Event()
            # The tree assigns the assistant node id at ``begin_assistant``
            # time inside the generation; we don't know it before the
            # gen starts.  The token consumer reads the live active
            # node off the tree (which is set to the streaming assistant
            # node for the lifetime of the gen).
            current_node_holder: list[str | None] = [None]
            # Set once this sibling's worker has ended: a callback that
            # outlives it (a consumer held past its generation) emits nothing.
            sealed = threading.Event()

            def _on_event(
                event: TokenEvent,
                _node_holder: list[str | None] = current_node_holder,
                _sealed: threading.Event = sealed,
            ) -> None:
                if _sealed.is_set():
                    return
                send_token(cast(JSONObject, build_token_event(session, _node_holder, event)))
            consumer = TokenConsumer(
                _on_event,
                TokenConsumerOptions(
                    live_scores=True,
                    lens_readout=True,
                    sae_readout=True,
                    perplexity=True,
                ),
            )
            # Live instrument readouts ride each token's ``measurements``
            # envelope when the family's live toggle is on and this consumer
            # declares interest (``TokenConsumerOptions``).

            result_holder: list[GenerationResult] = []
            error_holder: list[BaseException] = []

            # Recipe override: a mode string or a partial-recipe expression,
            # passed through ``generate`` so the engine composes it onto the
            # anchor's recipe (``drowse.loom.recipe.compose_call``).
            recipe_override = msg.recipe_override

            def _worker(
                _sampling: SamplingConfig | None = per_sibling_sampling,
                _on_token: TokenConsumer = consumer,
                _result_holder: list[GenerationResult] = result_holder,
                _error_holder: list[BaseException] = error_holder,
                _recipe_override: str | None = recipe_override,
            ) -> None:
                try:
                    with session.generation_state.cancellation_scope(cancelled):
                        effective_parent = parent_node_id
                        if authored is not None:
                            if not submitted_parent_holder:
                                submitted_parent_holder.append(_commit_authored(session, msg, authored))
                            effective_parent = submitted_parent_holder[0]
                        if msg.fork_node_id is not None:
                            # Fork: recipe / sampling / parent all come from
                            # the source node inside ``fork_from_token``; the
                            # WS-level steering/sampling/n fields are ignored.
                            result = session.fork_from_token(
                                msg.fork_node_id,
                                int(msg.fork_raw_index),  # pyright: ignore[reportArgumentType]  # guarded non-None by is_fork check above; int() accepts int|None only at runtime with None already excluded
                                alt_token_id=msg.fork_alt_token_id,
                                replacement_text=msg.fork_replacement_text,
                                **({"seed": msg.fork_seed} if msg.fork_seed is not None else {}),
                                on_token=_on_token,
                            )
                        else:
                            gen_kwargs: dict[str, Any] = {
                                "steering": steering,
                                "sampling": _sampling,
                                "stateless": msg.stateless,
                                "raw": msg.raw,
                                "thinking": msg.thinking,
                                "on_token": _on_token,
                                "parent_node_id": effective_parent,
                            }
                            if "system_prompt" in msg.model_fields_set:
                                gen_kwargs["system_prompt"] = msg.system_prompt
                            if _recipe_override is not None:
                                gen_kwargs["recipe_override"] = _recipe_override
                            if msg.generate_seat is not None:
                                gen_kwargs["gen_seat"] = msg.generate_seat
                            # Weave exploration preserves the source even for one
                            # candidate; ordinary chat keeps its coalescing default.
                            if n > 1 or not msg.append_same_role:
                                gen_kwargs["append_same_role"] = False
                            result = session.generate(
                                build_input(msg.input), **gen_kwargs,
                            ).first
                        _result_holder.append(result)
                except BaseException as e:
                    _error_holder.append(e)

            await send_json({
                "type": "started",
                "generation_id": generation_id,
                # ``node_id`` is filled in lazily by the first token
                # event (the assistant node is created inside
                # the generation); ``started`` includes the request-
                # level context the client needs to allocate state.
                "node_id": None,
                "sibling_index": sibling_idx,
                "sibling_count": n,
            })

            worker_task = asyncio.create_task(asyncio.to_thread(_worker))

            # Race the worker against client controls from the connection's
            # perpetual reader.  Nothing here waits on the socket, so a stop
            # lands however far behind the client's reading is.
            stop_signaled = False
            joined = False
            client_get = asyncio.create_task(incoming.get(control_only=True))
            try:
                while not worker_task.done():
                    finished, _pending = await asyncio.wait(
                        {worker_task, client_get}, return_when=asyncio.FIRST_COMPLETED,
                    )
                    if client_get in finished:
                        incoming_msg = client_get.result()
                        if isinstance(incoming_msg, _Disconnect):
                            raise WebSocketDisconnect(code=incoming_msg.code)
                        if isinstance(incoming_msg, _Stop) and (
                            incoming_msg.request_id is None or incoming_msg.request_id == msg.request_id
                        ):
                            cancelled.set()
                            stop_signaled = True
                        client_get = asyncio.create_task(incoming.get(control_only=True))
                joined = True
            finally:
                cancelled.set()
                with CancelScope(shield=True):
                    # A stop racing the reply's end must still abort the fan.
                    if client_get.done() and not client_get.cancelled():
                        pending_control = client_get.result()
                        if isinstance(pending_control, _Stop) and (
                            pending_control.request_id is None or pending_control.request_id == msg.request_id
                        ):
                            stop_signaled = True
                    client_get.cancel()
                    try:
                        await asyncio.gather(client_get, return_exceptions=True)
                    finally:
                        await finish_worker(worker_task)
                        sealed.set()
                        if not joined and record_event is not None:
                            if result_holder:
                                record_event({"type": "done", "result": result_to_json(result_holder[0]),
                                              "node_id": current_node_holder[0], "sibling_index": sibling_idx,
                                              "sibling_count": n})
                            elif error_holder:
                                info = describe_error(error_holder[0], failure="Generation failed")
                                record_event(_error_frame(info.message, code=info.code, status=info.status))

            if error_holder and not result_holder:
                info = describe_error(error_holder[0], failure="Generation failed")
                await send_json(_error_frame(
                    info.message,
                    code=info.code,
                    status=info.status,
                    node_id=current_node_holder[0],
                    sibling_index=sibling_idx,
                ))
                # An error inside one sibling aborts the rest of the fan.
                return False

            if not result_holder:
                raise RuntimeError("generation completed without a result")
            result = result_holder[0]
            # The engine reports ``cancelled`` when the stop cut the reply, and
            # the loom node records the same reason; a stop that raced the
            # reply's own end leaves its natural reason.
            result_json = result_to_json(result)
            # The aggregate-scope measurements envelope the engine built at
            # finalize, per sibling.
            if result.measurements:
                result_json["measurements"] = result.measurements
            # The per-turn logprob rollup, read off the finalized loom node
            # (``None`` for stateless generations).
            mean_logprob_out: float | None = None
            mean_surprise_out: float | None = None
            finalized_node_id = current_node_holder[0]
            if finalized_node_id is not None:
                node = session.tree.get(finalized_node_id)
                mean_logprob_out = node.mean_logprob
                mean_surprise_out = node.mean_surprise
            result_json["mean_logprob"] = mean_logprob_out
            result_json["mean_surprise"] = mean_surprise_out
            receipt: JSONObject = {
                "type": "done", "result": result_json, "node_id": current_node_holder[0],
                "sibling_index": sibling_idx, "sibling_count": n,
            }
            if record_event is not None:
                record_event(receipt)
            # The finalize tree delta was put on the worker thread before the
            # worker ended, so this ``done`` follows it on the wire.
            await send_json(receipt)

            # Mid-batch stop honors the batch plan: cancel the currently
            # streaming sibling and skip every remaining queued sibling.
            if stop_signaled:
                return True
    return False
