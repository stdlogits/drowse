"""Native unit routes under ``/drowse/v1/sessions/{id}/instruments/units``.

What a neuron or head is about, from its weights, over HTTP (the CLI's
``drowse unit``):

- ``GET units/read?unit=neuron/L8n344&k=8&hide_hubs=false`` — the unit view:
  per neuron pole its writes (``out``) and what it fires on (``mid``), per
  head its copies, suppressions and inputs, each through the depth rule's S
  and R halves; scalars, firing statistics, the catalog label and the unit's
  neuron-lens.com page.
- ``GET units/top?token=Paris&by=writes&layers=all&k=20`` — the neuron poles
  that write or fire on a token most specifically.
- ``POST units/find {selector | vector + layer, k, layers}`` — a direction
  split over each layer's neuron writes.
- ``GET units/catalog`` — the model's neuron-lens catalog binding.

The catalog fetch is the units family's ``fetch`` preparation
(``POST .../instruments/units/preparations {operation: "fetch"}``, polled like
the lens fetch; :mod:`drowse.server.instrument_routes`).  Reads run model work
under the session lock; the catalog status is disk only.
"""

from __future__ import annotations

from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import Field

from drowse.model.layers import parse_layer_selector
from drowse.server.response_models import (
    HeadReadJSON,
    NeuronReadJSON,
    UnitCatalogJSON,
    UnitFindJSON,
    UnitTopJSON,
)
from drowse.server.runtime import NativeRequest, run_in_thread, session_lock, session_router


class UnitFindRequest(NativeRequest):
    """Exactly one target: a steerable ``selector`` (one term's selector, as
    ``lens decompose`` takes) or a residual ``vector`` at ``layer``."""

    selector: str | None = Field(default=None, min_length=1)
    vector: list[float] | None = Field(default=None, min_length=1)
    layer: int | None = Field(default=None, ge=0)
    k: int = Field(default=16, ge=1, le=256)
    layers: str | None = None


class UnitCatalogFetchRequest(NativeRequest):
    """The units ``fetch`` preparation's fields: ``force`` refetches a
    current binding."""

    force: bool = False


def _layers(raw: str | None) -> "list[int] | str | None":
    try:
        return parse_layer_selector(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def register_unit_routes(app: FastAPI) -> None:
    session = app.state.session
    router = session_router()

    @router.get("/instruments/units/read")
    async def unit_read(
        unit: str, k: int = 8, hide_hubs: bool = False,
    ) -> NeuronReadJSON | HeadReadJSON:
        """One unit from its weights (the unit view)."""
        async with session_lock(app):
            read = await run_in_thread(session.units.read, unit, k=k, hide_hubs=hide_hubs)
        return read.to_dict()  # type: ignore[return-value]

    @router.get("/instruments/units/top")
    async def unit_top(
        token: str,
        by: Literal["writes", "fires"] = "writes",
        layers: str | None = None,
        k: int = 20,
    ) -> UnitTopJSON:
        """Neuron poles that write (``by=writes``) or fire on (``by=fires``)
        a single-token word (or ``#<id>``), by the depth-weighted
        layer-contrastive z."""
        selected = _layers(layers)
        async with session_lock(app):
            result = await run_in_thread(session.units.top, token, by=by, layers=selected, k=k)
        return result.to_dict()  # type: ignore[return-value]

    @router.post("/instruments/units/find")
    async def unit_find(body: UnitFindRequest) -> UnitFindJSON:
        """A direction split over each layer's neuron writes, with each
        neuron's pole and label."""
        if (body.selector is None) == (body.vector is None):
            raise HTTPException(400, "find takes exactly one of selector or vector")
        if body.vector is not None and body.layer is None:
            raise HTTPException(400, "a vector needs the layer it lives at")
        target: str | torch.Tensor = (
            body.selector if body.selector is not None
            else torch.tensor(body.vector, dtype=torch.float32)
        )
        selected = _layers(body.layers)
        async with session_lock(app):
            result = await run_in_thread(
                session.units.find, target, layer=body.layer, k=body.k, layers=selected,
            )
        return result.to_dict()  # type: ignore[return-value]

    @router.get("/instruments/units/catalog")
    def unit_catalog() -> UnitCatalogJSON:
        """The model's neuron-lens catalog binding (disk only, no lock)."""
        return session.units.catalog_status()

    app.include_router(router)
