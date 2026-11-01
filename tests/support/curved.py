"""A curved manifold whose layers differ in scale, rank and σ thickness."""
from __future__ import annotations

import math

import torch

from drowse.geometry.domains import BoxAxis, BoxDomain
from drowse.geometry.manifold import Manifold
from drowse.geometry.rbf import RbfFitPlan
from drowse.geometry.subspace import LayerSubspace, fit_curved_subspace

DIM = 8


def curved_stack_manifold(n_layers: int = 4, *, periodic: bool = False) -> Manifold:
    """A curved 1-D manifold whose layers differ in scale, rank and σ-field.

    Even layers bend into a second direction (rank 2); odd layers add a third
    bend (rank 3), so the stack pads ranks.  Every layer carries a σ-field of
    its own thickness.
    """
    axis = (
        BoxAxis("t", periodic=True, period=6.0) if periodic
        else BoxAxis("u", periodic=False, lo=-1.0, hi=1.0)
    )
    domain = BoxDomain([axis])
    if periodic:
        coords = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
        ang = coords.reshape(-1) * (2 * math.pi / 6)
        u, v = torch.cos(ang), torch.sin(ang)
    else:
        coords = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        u = coords.reshape(-1)
        v = u ** 2
    e = torch.eye(DIM)
    layers: dict[int, LayerSubspace] = {}
    share: dict[int, float] = {}
    for layer in range(n_layers):
        s = 1.0 + 0.3 * layer
        cents = s * (u.unsqueeze(1) * e[0] + v.unsqueeze(1) * e[1])
        if layer % 2:
            cents = cents + 0.4 * (u ** 3).unsqueeze(1) * e[2 + layer % 3]
        cents = cents + 0.05 * layer * e[5]
        fit = fit_curved_subspace(cents, domain.embed(coords))
        sub = fit.subspace
        np_, _rw, _pc = sub.rbf_params()
        log_sigma = torch.log(torch.full((np_.shape[0], 1), 0.2 + 0.05 * layer))
        sigma = RbfFitPlan.build(np_).fit(log_sigma)
        sub.sigma_rbf_weights, sub.sigma_poly_coeffs = sigma.weights, sigma.poly
        layers[layer] = sub
        share[layer] = 0.5 + 0.1 * layer
    return Manifold(
        name="curve", domain=domain,
        node_labels=[f"n{i}" for i in range(coords.shape[0])],
        node_coords=coords, layers=layers, mahalanobis_share=share,
        origin={layer: torch.zeros(1) for layer in layers},
    )


