"""The autograd lens oracle: every kind and site by brute-force autograd.

The transport estimator (:mod:`drowse.lens.transport`) writes each block's
lens map out by hand from its :class:`~drowse.model.blocks.BlockSpec`.  This
module computes the same per-document estimates the slow, independent way:
autograd through the model's own ``transformers`` modules, with the RelP
rules (R, S) applied as derivative patches on those modules
(:func:`relp_backward_rules`, drowse's estimator before transport).  The two
share nothing but the definition of the lens:

- output kinds: ``J_L[r] = mean_{t∈V} Σ_{t'∈V} ∂h_final(t')[r] / ∂h_L(t)``,
  read through a zero perturbation added to the site;
- input kinds: ``K_L[r] = Σ_{s∈V} mean_{t∈V} ∂h_L(t)[r] / ∂e(s)``, ``e`` the
  residual stream's input (``inputs_embeds``, the scaled embedding).

``V`` is the valid positions ``[skip_first, T − 1)``.  A site's residual is
reached through the module whose output is added to it: the block itself
(``out``), or the attention (its output norm on a sandwich block) for
``mid``.  Rows come ``batch`` at a time from replicated forwards, so the
oracle also runs on a real model for a handful of layers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from types import MethodType
from typing import Any, cast

import torch
from torch import nn

from drowse.lens.key import LensKey
from drowse.lens.relp import HALF_RULE_BETA
from drowse.model.blocks import model_spec

__all__ = ["RelpOracleError", "oracle_estimate", "oracle_lens", "relp_backward_rules"]


class RelpOracleError(RuntimeError):
    """A module the oracle's RelP patches cannot reproduce."""


# -- RelP as derivative patches ------------------------------------------------------

_VERIFY_RTOL = 1e-2
_VERIFY_ATOL = 1e-2


class _DetachedFactorProduct(torch.autograd.Function):
    """Return ``value`` untouched; backpropagate ``grad · factor`` to ``x``."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, value: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        del x
        ctx.save_for_backward(factor)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (grad,) = grad_outputs
        (factor,) = ctx.saved_tensors
        work = torch.promote_types(grad.dtype, torch.float32)
        return (grad.to(work) * factor.to(work)).to(grad.dtype), None, None


class _HalfProduct(torch.autograd.Function):
    """Forward ``a · b`` exactly; backward halves each factor (β = 0.5)."""

    @staticmethod
    def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (grad,) = grad_outputs
        a, b = ctx.saved_tensors
        return HALF_RULE_BETA * (grad * b), HALF_RULE_BETA * (grad * a)


def _norm_factor(module: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The LN rule's diagonal ``(offset + w)·rsqrt(mean(x²)+eps)``, offset found
    by reconstructing the module's own output."""
    eps = getattr(module, "variance_epsilon", getattr(module, "eps", None))
    weight = getattr(module, "weight", None)
    if not isinstance(eps, float) or not isinstance(weight, torch.Tensor):
        raise RelpOracleError(f"{type(module).__name__} exposes no eps/weight")
    work = torch.promote_types(x.dtype, torch.float32)
    xf = x.detach().to(work)
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    wf = weight.detach().to(work)
    yf = y.detach().to(work)
    matches = [
        offset for offset in (0.0, 1.0)
        if torch.allclose(xf * ((offset + wf) * scale), yf, rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL)
    ]
    if len(matches) != 1:
        raise RelpOracleError(f"{type(module).__name__} is not a plain or 1+w RMSNorm")
    return (matches[0] + wf) * scale


def _act_factor(act: Any, x: torch.Tensor) -> torch.Tensor:
    xf = x.detach().to(torch.promote_types(x.dtype, torch.float32))
    name = type(act).__name__.casefold().replace("activation", "")
    if "silu" in name or "swish" in name:
        return torch.sigmoid(xf)
    if "gelu" in name and ("tanh" in name or getattr(act, "approximate", None) == "tanh"):
        return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (xf + 0.044715 * xf.pow(3))))
    raise RelpOracleError(f"no identity-rule factor for {type(act).__name__}")


def _relp_norm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        y = type(self).forward(self, x.detach())
    return cast(torch.Tensor, _DetachedFactorProduct.apply(x, y, _norm_factor(self, x, y)))


def _relp_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate_proj = cast(nn.Module, getattr(self, "gate_proj"))
    up_proj = cast(nn.Module, getattr(self, "up_proj"))
    down_proj = cast(nn.Module, getattr(self, "down_proj"))
    act_fn = cast(Callable[[torch.Tensor], torch.Tensor], getattr(self, "act_fn"))
    gate_pre = cast(torch.Tensor, gate_proj(x))
    act_value = act_fn(gate_pre.detach())
    gate = cast(torch.Tensor, _DetachedFactorProduct.apply(
        gate_pre, act_value, _act_factor(act_fn, gate_pre),
    ))
    up = cast(torch.Tensor, up_proj(x))
    result = cast(torch.Tensor, down_proj(_HalfProduct.apply(gate, up)))
    with torch.no_grad():
        expected = type(self).forward(self, x.detach())
    if not torch.allclose(result.detach(), expected, rtol=1e-6, atol=1e-8):
        raise RelpOracleError(f"{type(self).__name__} is not the canonical gated MLP")
    return result


@contextmanager
def relp_backward_rules(layer_modules: Sequence[nn.Module]) -> Iterator[None]:
    """Patch every block's residual RMSNorms (direct children) and gated MLP
    so autograd takes the RelP derivative; forwards are unchanged."""
    patched: list[nn.Module] = []

    def patch(module: nn.Module, fn: Any) -> None:
        if "forward" in module.__dict__:
            raise RelpOracleError(f"{type(module).__name__} has an instance-level forward")
        patched.append(module)
        object.__setattr__(module, "forward", MethodType(fn, module))

    try:
        for layer in layer_modules:
            for _name, child in layer.named_children():
                if "rmsnorm" in type(child).__name__.casefold():
                    patch(child, _relp_norm_forward)
                elif all(hasattr(child, a) for a in ("gate_proj", "up_proj", "down_proj", "act_fn")):
                    patch(child, _relp_mlp_forward)
        yield
    finally:
        for module in reversed(patched):
            if "forward" in module.__dict__:
                object.__delattr__(module, "forward")


# -- the brute-force estimate --------------------------------------------------------

def _first(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _with_first(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


def _mid_writer(model: Any, layer: int) -> nn.Module:
    """The module whose output the block adds to form its ``mid`` residual."""
    block = model_spec(model).blocks[layer]
    if block.attn_out_norm is not None:
        return block.attn_out_norm.module
    return block.attn.module


def _decoder(model: Any) -> nn.Module:
    return cast(nn.Module, model.get_decoder()) if hasattr(model, "get_decoder") else model


def oracle_estimate(
    model: Any,
    ids: torch.Tensor,
    *,
    kind: str,
    layers: Mapping[str, Sequence[int]],
    skip_first: int,
    batch: int = 64,
) -> dict[str, dict[int, torch.Tensor]]:
    """One document's lens estimate at ``layers`` (``{site: layers}``) by
    autograd, oriented as :class:`~drowse.lens.jlens.JacobianLens` holds it.

    Runs in the model's own precision (an fp64 model gives fp64 truth);
    returns CPU tensors in ``promote(dtype, fp32)``.
    """
    key = LensKey(kind)
    spec = model_spec(model)
    blocks = [block.module for block in spec.blocks]
    n_layers = len(blocks)
    device = spec.embed.weight.device
    dtype = spec.embed.weight.dtype
    work = torch.promote_types(dtype, torch.float32)
    ids = ids.to(device=device, dtype=torch.long)
    T = int(ids.numel())
    valid = torch.arange(skip_first, T - 1, device=device)
    if not valid.numel():
        raise ValueError("document leaves no valid position")
    d = spec.d_model
    decoder = _decoder(model)
    out: dict[str, dict[int, torch.Tensor]] = {
        site: {int(layer): torch.zeros(d, d, dtype=work) for layer in wanted}
        for site, wanted in layers.items()
    }
    with ExitStack() as stack, torch.enable_grad():
        if key.relp:
            stack.enter_context(relp_backward_rules(blocks))
        for start in range(0, d, batch):
            rows = list(range(start, min(start + batch, d)))
            b = len(rows)
            if key.side == "output":
                _output_rows(
                    decoder, blocks, model, ids, valid, rows, out,
                    n_layers=n_layers, d=d, dtype=dtype, work=work, b=b,
                )
            else:
                _input_rows(
                    decoder, blocks, model, spec, ids, valid, rows, out,
                    d=d, dtype=dtype, work=work, b=b,
                )
    return out


def _output_rows(
    decoder: nn.Module, blocks: list[nn.Module], model: Any, ids: torch.Tensor,
    valid: torch.Tensor, rows: list[int], out: dict[str, dict[int, torch.Tensor]],
    *, n_layers: int, d: int, dtype: torch.dtype, work: torch.dtype, b: int,
) -> None:
    T = int(ids.numel())
    device = ids.device
    probes: dict[tuple[str, int], torch.Tensor] = {}
    handles = []
    final: list[torch.Tensor] = []

    def add_probe(site: str, layer: int) -> Callable[..., Any]:
        def hook(_m: nn.Module, _args: Any, output: Any) -> Any:
            probe = torch.zeros(b, T, d, device=device, dtype=dtype, requires_grad=True)
            probes[(site, layer)] = probe
            return _with_first(output, _first(output) + probe)
        return hook

    try:
        for site, wanted in out.items():
            for layer in wanted:
                module = blocks[layer] if site == "out" else _mid_writer(model, layer)
                handles.append(module.register_forward_hook(add_probe(site, layer)))
        handles.append(blocks[n_layers - 1].register_forward_hook(
            lambda _m, _a, output: final.append(_first(output)),
        ))
        decoder(input_ids=ids[None].expand(b, T), use_cache=False)
        cot = torch.zeros(b, T, d, device=device, dtype=final[-1].dtype)
        index = torch.arange(b, device=device)
        cot[index[:, None], valid[None, :], torch.tensor(rows, device=device)[:, None]] = 1
        names = list(probes)
        grads = torch.autograd.grad(final[-1], [probes[n] for n in names], grad_outputs=cot)
    finally:
        for handle in handles:
            handle.remove()
    for (site, layer), grad in zip(names, grads):
        out[site][layer][rows] = grad[:, valid].to(work).mean(1).cpu()


def _input_rows(
    decoder: nn.Module, blocks: list[nn.Module], model: Any, spec: Any, ids: torch.Tensor,
    valid: torch.Tensor, rows: list[int], out: dict[str, dict[int, torch.Tensor]],
    *, d: int, dtype: torch.dtype, work: torch.dtype, b: int,
) -> None:
    T = int(ids.numel())
    device = ids.device
    index = torch.arange(b, device=device)
    row_index = torch.tensor(rows, device=device)
    with torch.no_grad():
        embeds = spec.embed(ids)
    for site, wanted in out.items():
        for layer in wanted:
            e = embeds[None].expand(b, T, d).clone().requires_grad_(True)
            captured: list[torch.Tensor] = []
            if site == "out":
                handle = blocks[layer].register_forward_hook(
                    lambda _m, _a, output: captured.append(_first(output)),
                )
            else:
                norm = spec.blocks[layer].mlp_in_norm.module
                handle = norm.register_forward_pre_hook(
                    lambda _m, args: captured.append(args[0]),
                )
            try:
                decoder(inputs_embeds=e, use_cache=False)
            finally:
                handle.remove()
            target = captured[0]
            cot = torch.zeros_like(target)
            cot[index[:, None], valid[None, :], row_index[:, None]] = 1.0 / valid.numel()
            (grad,) = torch.autograd.grad(target, [e], grad_outputs=cot)
            out[site][layer][rows] = grad[:, valid].to(work).sum(1).cpu()


def oracle_lens(
    model: Any,
    rows: Sequence[Sequence[int]],
    *,
    kind: str,
    layers: Mapping[str, Sequence[int]],
    skip_first: int,
    batch: int = 64,
) -> dict[str, dict[int, torch.Tensor]]:
    """The equal-weight mean of :func:`oracle_estimate` over documents."""
    usable = [list(row) for row in rows if len(row) - 1 - skip_first > 0]
    total: dict[str, dict[int, torch.Tensor]] | None = None
    for row in usable:
        est = oracle_estimate(
            model, torch.tensor(row), kind=kind, layers=layers,
            skip_first=skip_first, batch=batch,
        )
        if total is None:
            total = est
        else:
            for site, matrices in est.items():
                for layer, matrix in matrices.items():
                    total[site][layer] += matrix
    if total is None:
        raise ValueError("no usable document")
    for matrices in total.values():
        for matrix in matrices.values():
            matrix /= len(usable)
    return total
