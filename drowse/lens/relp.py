"""The RelP derivative rules the R- and S-lenses are taken under.

The R-lens ("R-lens: making J-lens more faithful on early layers",
LessWrong 2026) and the S-lens keep the J- and K-lens estimators and read
the Jacobian through an LRP-modified derivative (RelP, arXiv:2508.21258).
Forwards are unchanged; only the linear map each decoder block contributes
differs.  Three rules, dense-model scope:

- **LN rule.** Each residual RMSNorm (the attention's and the MLP's input
  norms, and Gemma's sandwich norms on their outputs) holds its ``rsqrt``
  constant: its map is the diagonal ``gain·rsqrt(mean(x²)+eps)`` and the
  rank-one ``d rms / d x`` term is dropped.
- **Identity rule.** The gated activation ``act(g) = g·φ(g)`` is linearized
  as ``φ(g)·dg`` (``sigmoid`` for SiLU, the tanh-GELU CDF factor) instead of
  ``act'(g)``: :func:`~drowse.model.blocks.activation_factor`.
- **Half rule.** ``act(gate)·up`` passes half of each factor's change
  (β = 0.5) instead of the full product rule.

Norms inside attention (q/k norms, Gemma 4's weightless value norm),
softmax, RoPE and the logit softcap keep their true derivative
(``include_qk_norms: false``).  :mod:`drowse.lens.transport` writes these
maps out per document (``rules="relp"``); the standard J/K-lenses take the
same maps with every derivative true (``rules="standard"``).

``RELP_RULES`` is recorded in every RelP lens's estimator policy, so a rule
change reads as a different estimator and every stored R/S-lens stops
loading.  Change it only for a genuinely new estimator.
"""

from __future__ import annotations

#: The half rule's share per factor.
HALF_RULE_BETA = 0.5

RELP_RULES = {
    "ln_rule": True,
    "identity_rule": True,
    "half_rule": True,
    "half_rule_beta": HALF_RULE_BETA,
    "include_qk_norms": False,
}

__all__ = ["HALF_RULE_BETA", "RELP_RULES"]
