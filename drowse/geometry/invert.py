"""Nearest-point projection onto a curved manifold surface (the foot solve).

The paper's ``s⁻¹`` map: minimize ``‖s(p) − q‖`` over authoring coords ``p``
by damped Gauss-Newton (Levenberg-Marquardt) with the analytic RBF Jacobian
chained through the domain's embedding Jacobian.  :func:`_gn_step` is the one
LM body, shared by :func:`invert_parameterization` (the read side, the fit
origin, transfer) and :func:`~drowse.geometry.inject.subspace_inject` (one
warm-started step per token on the steering hot path).
"""
from __future__ import annotations

import torch

from drowse.geometry.domains import ManifoldDomain
from drowse.geometry.rbf import eval_rbf, eval_rbf_jet
from drowse.geometry.subspace import LayerSubspace, StackedRbf

# The solve is warm-started from the nearest fit node(s) and Marquardt-damped,
# so a fixed dozen iterations converges in authoring-coord space independent
# of intrinsic dimension.
DEFAULT_INVERSION_MAX_ITER = 12
DEFAULT_INVERSION_RESTARTS = 3
# Warm-started inversion (curved-probe per-token foot-follow): a one-decode-step
# activation drift leaves the carried foot near this token's nearest point, so
# a handful of LM steps (plus one nearest-node restart as a basin-jump safety
# net) converges where the cold 12-iter / 3-restart search would.
DEFAULT_INVERSION_WARM_ITER = 4
DEFAULT_INVERSION_DAMPING = 1e-3
# Absolute floor added to the LM normal-equation diagonal so a locally
# rank-deficient Jacobian (a fold/pinch, or a flat authoring direction)
# still yields a solvable, well-conditioned system.
_INVERSION_DIAG_FLOOR = 1e-9


def _gn_step(
    p: torch.Tensor,
    q: torch.Tensor,
    node_params: torch.Tensor,
    rbf_weights: torch.Tensor,
    poly_coeffs: torch.Tensor,
    coord_offset: torch.Tensor,
    coord_scale: torch.Tensor,
    domain: ManifoldDomain,
    damping: float,
) -> torch.Tensor:
    """One damped Gauss-Newton (Levenberg-Marquardt) step of ``argmin_p ‖s(p) − q‖``.

    ``p`` ``(.., n)`` are authoring coords, ``q`` ``(.., R)`` the reduced-space
    targets (broadcastable against ``p``'s leading dims — the inversion fans
    ``p`` over ``S`` restarts against a ``(.., 1, R)`` query).  The RBF tensors
    are already on ``p``'s device/dtype.  Returns the retracted post-step
    position.  The ``n ≤ 2`` normal equations are solved in closed form: a
    batched ``linalg.solve`` checks for singularity with a host sync, which on
    the steering hot path would drain the device queue mid-forward.

    Stacked parameters (a :class:`~drowse.geometry.subspace.StackedRbf`'s
    leading dim ``L``) take ``p`` ``(L, X, n)`` against ``q`` ``(L, 1, R)``:
    every stack entry steps against its own surface.
    """
    if coord_offset.ndim > 1:
        coord_offset = coord_offset.unsqueeze(-2)                 # (L, 1, m)
        coord_scale = coord_scale.unsqueeze(-2)
    emb = (domain.embed(p) - coord_offset) / coord_scale          # (.., m)
    value, jac = eval_rbf_jet(node_params, rbf_weights, poly_coeffs, emb)
    resid = value - q                                             # (.., R)
    j_auth = (jac / coord_scale.unsqueeze(-2)) @ domain.tangent_jacobian(p)  # (.., R, n)
    jt = j_auth.transpose(-1, -2)                                 # (.., n, R)
    jtj = jt @ j_auth                                             # (.., n, n)
    jtr = jt @ resid.unsqueeze(-1)                                # (.., n, 1)
    n_dim = jtj.shape[-1]
    diag = torch.diagonal(jtj, dim1=-2, dim2=-1)                  # (.., n)
    reg = damping * diag.clamp(min=_INVERSION_DIAG_FLOOR) + _INVERSION_DIAG_FLOOR
    if n_dim == 1:
        step = jtr[..., 0] / (jtj[..., 0] + reg)                  # (.., 1)
    elif n_dim == 2:
        # Damped JᵀJ is SPD, so Cramer's rule is safe — and, unlike a batched
        # ``linalg.solve``, it never syncs the device mid-forward.
        a00 = jtj[..., 0, 0] + reg[..., 0]
        a11 = jtj[..., 1, 1] + reg[..., 1]
        a01 = jtj[..., 0, 1]
        r0, r1 = jtr[..., 0, 0], jtr[..., 1, 0]
        det = a00 * a11 - a01 * jtj[..., 1, 0]
        step = torch.stack((a11 * r0 - a01 * r1, a00 * r1 - jtj[..., 1, 0] * r0), dim=-1)
        step = step / det.unsqueeze(-1)                           # (.., 2)
    else:
        A = jtj + torch.diag_embed(reg)                           # (.., n, n)
        bsz = A.shape[:-2]
        # The flattened batch also dodges ``linalg.solve``'s size-1-leading-
        # batch out-resize warning on MPS.
        step = torch.linalg.solve(
            A.reshape(-1, n_dim, n_dim).contiguous(),
            jtr.reshape(-1, n_dim, 1).contiguous(),
        ).reshape(*bsz, n_dim)                                    # (.., n)
    return domain.retract_position(p, -step)


def invert_parameterization(
    subspace: LayerSubspace,
    domain: ManifoldDomain,
    query: torch.Tensor,
    node_coords: torch.Tensor,
    *,
    max_iter: int = DEFAULT_INVERSION_MAX_ITER,
    n_restarts: int = DEFAULT_INVERSION_RESTARTS,
    warm_start: torch.Tensor | None = None,
    warm_iter: int = DEFAULT_INVERSION_WARM_ITER,
    damping: float = DEFAULT_INVERSION_DAMPING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point projection of ``query`` ``(.., R)`` onto the fitted surface.

    Returns ``(positions (.., n), dist (..,))``: the authoring coords whose
    interpolant value is nearest each query in reduced-coordinate space, and
    that distance.  ``node_coords`` ``(K, n)`` are the manifold's authoring
    node coordinates (the warm-start seeds).

    Cold path: seed each query at its ``n_restarts`` nearest fit nodes (in
    reduced space) and take ``max_iter`` LM steps; the best final residual per
    query wins, guarding against a fold/periodic local minimum.  Warm path
    (``warm_start``, the curved-probe foot-follower): ``warm_iter`` steps over
    two restarts — the carried previous-token foot and the single nearest fit
    node.
    """
    n = domain.intrinsic_dim
    R = query.shape[-1]
    lead = query.shape[:-1]
    device = query.device
    dtype = query.dtype if query.dtype.is_floating_point else torch.float32
    flat = query.reshape(-1, R).to(dtype)  # (N, R)

    # The caller may hand a subspace still resident on its load device.
    np_, rw, pc = subspace.rbf_params()
    np_ = np_.to(device=device, dtype=dtype)  # (K, m)
    rw = rw.to(device=device, dtype=dtype)
    pc = pc.to(device=device, dtype=dtype)
    offset = subspace.coord_offset.to(device=device, dtype=dtype)  # (m,)
    scale = subspace.coord_scale.to(device=device, dtype=dtype)  # (m,)
    # Surface values at the fit nodes: the seed metric for the restarts.
    node_vals = eval_rbf(np_, rw, pc, np_)  # (K, R)
    ws = (
        warm_start.to(device=device, dtype=dtype).reshape(flat.shape[0], n)
        if warm_start is not None else None
    )
    pos, dist = _foot_solve(
        np_, rw, pc, offset, scale, node_vals, domain, flat,
        node_coords.to(device=device, dtype=dtype), torch.cdist(flat, node_vals),
        warm_start=ws, max_iter=max_iter, n_restarts=n_restarts,
        warm_iter=warm_iter, damping=damping,
    )
    return pos.reshape(*lead, n), dist.reshape(lead)


def invert_stacked(
    surfaces: StackedRbf,
    domain: ManifoldDomain,
    query: torch.Tensor,
    node_coords: torch.Tensor,
    *,
    max_iter: int = DEFAULT_INVERSION_MAX_ITER,
    n_restarts: int = DEFAULT_INVERSION_RESTARTS,
    warm_start: torch.Tensor | None = None,
    warm_iter: int = DEFAULT_INVERSION_WARM_ITER,
    damping: float = DEFAULT_INVERSION_DAMPING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`invert_parameterization` for one query per stacked surface.

    ``query`` ``(L, R)`` holds one reduced-space target per layer of
    ``surfaces`` (on its device, fp32; zero in a layer's padded rank columns);
    ``warm_start`` ``(L, n)`` the carried feet.  Returns ``(positions (L, n),
    dist (L,))``.  The same restarts and LM steps as a per-layer solve, run as
    one batched solve whose every kernel computes each stack entry as the
    per-layer kernel does, so the monitor's curved read costs a fixed handful
    of kernels however many layers the probe spans.
    """
    flat = query.to(surfaces.node_params.dtype)  # (L, R)
    np_, rw, pc = surfaces.node_params, surfaces.rbf_weights, surfaces.poly_coeffs
    # Each layer's surface at its fit nodes, evaluated as the one-layer solve
    # does (per stack entry, on the read device): the restart seed metric.
    node_vals = eval_rbf(np_, rw, pc, np_)  # (L, K, R)
    seed_dist = torch.cdist(flat.unsqueeze(-2), node_vals).squeeze(-2)  # (L, K)
    return _foot_solve(
        np_, rw, pc, surfaces.coord_offset, surfaces.coord_scale, node_vals,
        domain, flat, node_coords.to(device=flat.device, dtype=flat.dtype), seed_dist,
        warm_start=warm_start, max_iter=max_iter, n_restarts=n_restarts,
        warm_iter=warm_iter, damping=damping,
    )


def _foot_solve(
    np_: torch.Tensor,
    rw: torch.Tensor,
    pc: torch.Tensor,
    offset: torch.Tensor,
    scale: torch.Tensor,
    node_vals: torch.Tensor,
    domain: ManifoldDomain,
    flat: torch.Tensor,
    node_coords: torch.Tensor,
    seed_dist: torch.Tensor,
    *,
    warm_start: torch.Tensor | None,
    max_iter: int,
    n_restarts: int,
    warm_iter: int,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The one restarted LM solve over ``N`` queries ``flat`` ``(N, R)``.

    The surface parameters are one layer's (shared by every query) or a stack
    with one entry per query (leading dim ``N``); ``seed_dist`` ``(N, K)`` is
    each query's reduced-space distance to the fit nodes' surface values.
    Returns ``(positions (N, n), dist (N,))``.
    """
    n = domain.intrinsic_dim
    N = flat.shape[0]
    K = node_coords.shape[0]
    if warm_start is not None:
        ws = warm_start.reshape(N, n)  # (N, n)
        near1 = seed_dist.argmin(dim=-1)  # (N,)
        p = domain.clamp_position(
            torch.stack([ws, node_coords[near1]], dim=1)  # (N, 2, n)
        )
        iters = int(warm_iter)
    else:
        restarts = max(1, min(int(n_restarts), K))
        seed_idx = seed_dist.topk(
            restarts, dim=-1, largest=False,
        ).indices  # (N, S)
        p = domain.clamp_position(node_coords[seed_idx])  # (N, S, n)
        iters = int(max_iter)
    q = flat.unsqueeze(1)  # (N, 1, R) -- broadcasts over the S restarts

    for _ in range(iters):
        p = _gn_step(p, q, np_, rw, pc, offset, scale, domain, damping)

    if offset.ndim > 1:
        offset, scale = offset.unsqueeze(-2), scale.unsqueeze(-2)  # (N, 1, m)
    final = eval_rbf(np_, rw, pc, (domain.embed(p) - offset) / scale)
    final_res = torch.linalg.vector_norm(final - q, dim=-1)  # (N, S)
    best = final_res.argmin(dim=-1)  # (N,)
    pos = p.gather(1, best[:, None, None].expand(N, 1, n)).squeeze(1)  # (N, n)
    dist = final_res.gather(1, best[:, None]).squeeze(1)  # (N,)
    return pos, dist


def neutral_origin(
    subspace: LayerSubspace,
    domain: ManifoldDomain,
    neutral_mean: torch.Tensor,
    node_coords: torch.Tensor,
) -> torch.Tensor:
    """A curved layer's origin: the foot of the neutral mean on its surface.

    Projects ``neutral_mean`` ``(D,)`` into the layer frame and inverts it onto
    the surface; returns the ``(n,)`` fp32 CPU authoring coords that seed the
    steering foot-follower and anchor ``along``'s translate.
    """
    sub = subspace.to(device=torch.device("cpu"), dtype=torch.float32)
    mean = neutral_mean.to(device="cpu", dtype=torch.float32).reshape(-1)
    if mean.shape != sub.mean.shape:
        raise ValueError(
            f"neutral mean has shape {tuple(mean.shape)}, expected "
            f"{tuple(sub.mean.shape)}"
        )
    query = (mean - sub.mean) @ sub.read.T
    origin, _distance = invert_parameterization(
        sub, domain, query, node_coords.to(device="cpu", dtype=torch.float32),
    )
    return origin.reshape(-1)


__all__ = [
    "DEFAULT_INVERSION_DAMPING",
    "DEFAULT_INVERSION_MAX_ITER",
    "DEFAULT_INVERSION_RESTARTS",
    "DEFAULT_INVERSION_WARM_ITER",
    "invert_parameterization",
    "invert_stacked",
    "neutral_origin",
]
