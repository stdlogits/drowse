"""``drowse unit <verb>`` — a model's MLP neurons and attention heads as
objects: what a unit is about from its weights (``read``, ``stats``), which
units write or fire on a token (``top``), which neurons a direction is made of
(``find``), and the neuron-lens catalog's labels (``fetch``, ``ls``, ``rm``:
the catalog binding is a model's label source, listed and removed as ``lens``
and ``sae`` list and remove theirs)."""

from __future__ import annotations

import argparse
from typing import Any

from drowse.cli.context import CliError, emit_json, note, open_session, parse_layers, progress
from drowse.cli.spec import FORCE, JSON, MODEL, MODEL_LOAD, MODEL_REQ, YES, Verb, arg, layers_arg, positive_int

_CREDIT = "Labels by Claude Haiku 5.5"


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _tokens(half: dict[str, Any], width: int) -> str:
    return "  ".join(t["token"].strip() or repr(t["token"]) for t in half["tokens"][:width])


def _half(half: dict[str, Any], width: int, *, at: str = "") -> str:
    """``S:out 1.00  tok tok tok`` or why the half is absent."""
    head = f"{half['lens'] or half['family']}{at}"
    if half["absent"] is not None:
        return f"{head:<9} {half['weight']:.2f}  (absent: {half['absent']})"
    return f"{head:<9} {half['weight']:.2f}  {_tokens(half, width)}"


def _place(title: str, place: dict[str, Any], width: int, *, at: str = "") -> list[str]:
    hubs = "  [hub tokens hidden]" if place["hubs"] == "hidden" else ""
    return [
        f"  {title:<11} {_half(place['s'], width, at=at)}{hubs}",
        f"  {'':<11} {_half(place['r'], width, at=at)}",
    ]


def _label_line(label: dict[str, Any] | None) -> str | None:
    if not label:
        return None
    parts = []
    for key in ("fires", "attends", "writes"):
        if label.get(key):
            conf = label.get(f"{key}_conf")
            parts.append(f"{key}: {label[key]}" + (f" ({conf:.2f})" if conf is not None else ""))
    return "  label  " + " · ".join(parts) if parts else None


def _firing_line(firing: dict[str, Any], *, head: bool) -> str:
    bits = []
    if firing["rate"] is not None:
        bits.append(f"fires on {100 * firing['rate']:.2f}% of tokens (catalog)")
    else:
        bits.append(f"rate: {firing['rate_absent']}")
    scale = firing["scale"]
    if scale is not None:
        # The scale a reading of this unit divides by, and where it came from.
        key = "write_top" if head else "act_top"
        relative = ", relative to the neutral text" if scale["source"] == "neutral" else ""
        bits.append(
            f"{key} {scale['top']:.3g} ({scale['detail']}) · readings over "
            f"{scale['value']:.3g}{relative}"
        )
    elif firing["neutral_absent"]:
        bits.append(f"neutral: {firing['neutral_absent']}")
    return "  " + " · ".join(bits)


def _catalog_line(block: dict[str, Any]) -> str:
    if block["bound"]:
        return f"  {block['url']}  ({_CREDIT})"
    return f"  catalog: {block['reason']}"


def _print_neuron(read: dict[str, Any], width: int) -> None:
    s = read["scalars"]
    mean = "" if s["mean_act"] is None else f" · mean act {s['mean_act']:.3g}"
    print(
        f"{read['unit']}  depth {read['depth']:.2f} · w_norm {s['w_norm']:.3g} · "
        f"cos(a,b) {s['cos_ab']:+.3f} · null share {s['null_share']:.3f}{mean}"
    )
    print(_catalog_line(read["catalog"]))
    for pole in read["poles"]:
        print(
            f"\n{pole['pole']} pole  logit-lens skew {pole['vocab_skew']:+.2f} · "
            f"kurtosis {pole['vocab_kurt']:.1f}"
        )
        print(_firing_line(pole["firing"], head=False))
        label = _label_line(pole["label"])
        if label:
            print(label)
        for line in _place("writes", pole["writes"], width) + _place("fires on", pole["fires_on"], width):
            print(line)


def _print_head(read: dict[str, Any], width: int) -> None:
    score = read["copy_score"]
    copies = " · ".join(
        f"{family} {value:+.2f}" for family, value in score.items() if value is not None
    ) or "none"
    print(f"{read['unit']}  kv head {read['kv_head']} · depth {read['depth']:.2f} · copy score {copies}")
    print(_catalog_line(read["catalog"]))
    print(_firing_line(read["firing"], head=True))
    label = _label_line(read["label"])
    if label:
        print(label)
    for line in _place("copies", read["copies"], width) + _place("suppresses", read["suppresses"], width):
        print(line)
    at = f"@L{read['inputs']['query']['layer']}" if read["layer"] > 0 else ""
    for role, place in read["inputs"].items():
        for line in _place(role, place, width, at=at):
            print(line)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_read(args: argparse.Namespace) -> None:
    with open_session(args) as session:
        read = session.units.read(args.unit, k=args.top_k, hide_hubs=args.hide_hubs).to_dict()
    if args.json_output:
        emit_json({"model": args.model, **read})
        return
    if read["kind"] == "neuron":
        _print_neuron(read, args.top_k)
    else:
        _print_head(read, args.top_k)


def run_top(args: argparse.Namespace) -> None:
    layers = parse_layers(args.layers)
    if layers == "sample":
        raise CliError("--layers sample is a readout mode; name layers, workspace or all")
    with open_session(args) as session:
        result = session.units.top(args.token, by=args.by, layers=layers, k=args.top_k).to_dict()
    if args.json_output:
        emit_json({"model": args.model, **result})
        return
    verb = "write" if result["by"] == "writes" else "fire on"
    print(
        f"neuron poles that {verb} {result['token']!r} (token {result['token_id']}), "
        "by depth-weighted layer z:"
    )
    for hit in result["hits"]:
        r, s = hit["r"], hit["s"]
        halves = []
        if r["z"] is not None:
            halves.append(f"R z {r['z']:+.2f} p {r['probability']:.3g}")
        if s["z"] is not None:
            halves.append(f"S z {s['z']:+.2f}")
        label = hit["label"]
        tail = f"  {label['fires'] if result['by'] == 'fires' else label['writes']}" if label else ""
        print(f"  {hit['unit']:<18} {hit['pole']}  {hit['score']:+7.2f}   {' · '.join(halves)}{tail}")
    missing = {
        layer: info["absent"] for layer, info in result["layers"].items()
        if info["absent"]["R"] and info["absent"]["S"]
    }
    if missing:
        note(f"no lens read layers {', '.join(sorted(missing, key=int))}: {next(iter(missing.values()))['R']}")


def run_find(args: argparse.Namespace) -> None:
    layers = parse_layers(args.layers)
    with open_session(args) as session:
        result = session.units.find(args.selector, k=args.top_k, layers=layers).to_dict()
    if args.json_output:
        emit_json({"model": args.model, **result})
        return
    print(f"neuron share of {args.selector!r} (k={args.top_k}): mean {result['mean_share']:.1%}")
    for layer, row in result["layers"].items():
        neurons = "  ".join(
            f"{n['pole']}n{n['index']} {abs(n['coeff']):.2f}" for n in row["neurons"][:6]
        )
        print(f"  L{int(layer):>3}  {row['share']:>6.1%}  {neurons}")


def run_stats(args: argparse.Namespace) -> None:
    with open_session(args) as session:
        stats = session.units.stats(args.unit)
    if args.json_output:
        emit_json({"model": args.model, **stats})
        return
    print(stats["unit"])
    print(_catalog_line(stats["catalog"]))
    if stats["kind"] == "neuron":
        for pole in stats["poles"]:
            print(f"{pole['pole']} pole")
            print(_firing_line(pole["firing"], head=False))
            catalog = pole["firing"]["catalog"]
            if catalog:
                print("  " + " · ".join(f"{k} {v:.3g}" for k, v in catalog.items() if isinstance(v, float)))
            label = _label_line(pole["label"])
            if label:
                print(label)
        return
    print(_firing_line(stats["firing"], head=True))
    catalog = stats["firing"]["catalog"]
    if catalog:
        print("  " + " · ".join(f"{k} {v:.3g}" for k, v in catalog.items() if isinstance(v, float)))
    label = _label_line(stats["label"])
    if label:
        print(label)


def run_fetch(args: argparse.Namespace) -> None:
    from drowse.units.catalog import fetch_catalog

    record = fetch_catalog(args.model, force=args.force, on_progress=progress)
    if args.json_output:
        emit_json(record)
        return
    print(
        f"Bound {record['repo']}@{record['revision'][:12]} for {record['model_id']}: "
        f"{len(record['files'])} tables under {record['site']} ({record['credit']})."
    )


def run_ls(args: argparse.Namespace) -> None:
    from drowse.units.catalog import catalog_status, list_catalogs

    rows = [catalog_status(args.model)] if args.model else list_catalogs()
    if args.json_output:
        emit_json(rows[0] if args.model else rows)
        return
    for row in rows:
        state = "bound" if row["bound"] else ("available" if row["available"] else "none")
        line = f"{row['model_id']:<24} {state:<10}"
        if row["binding"]:
            line += f" {row['repo']}@{str(row['binding']['revision'])[:12]} · {row['binding']['files']} tables"
            line += f" · fetched {row['binding']['fetched_at']}"
        elif row["reason"]:
            line += f" {row['reason']}"
        print(line)


def run_rm(args: argparse.Namespace) -> None:
    from drowse.units.catalog import remove_catalog, require_catalog_model

    require_catalog_model(args.model)
    if not args.yes:
        raise CliError(f"refusing to remove the unit catalog of {args.model}; pass -y/--yes to confirm")
    if not remove_catalog(args.model):
        raise CliError(f"no unit catalog is bound for {args.model}", exit_code=1)
    print(f"Removed the unit catalog of {args.model}.")


_UNIT = arg("unit", help="neuron/L<layer>n<index> or head/L<layer>h<index> (e.g. neuron/L8n344)")

GROUP = Verb(
    "unit",
    "MLP neurons and attention heads (read/top/find/stats/fetch/ls/rm)",
    description="A model's neurons and heads from their weights, read through "
                "the depth rule (S early, R late): what a neuron writes and fires "
                "on per pole, what a head copies and reads, which neurons write a "
                "token or make up a direction; with the neuron-lens catalog's "
                "labels where one exists (Qwen3-8B, Gemma 4 12B).",
    verbs=(
        Verb(
            "read", "What one unit is about, from its weights",
            run=run_read,
            description="A neuron per pole: its write read at the block output "
                        "(writes) and its read direction at the MLP input (fires "
                        "on), each through the S and R lenses with the depth "
                        "rule's weights; scalars, firing statistics and the "
                        "catalog label. A head: the tokens its OV circuit copies "
                        "and suppresses (at mid), what its query, key and value "
                        "pick up at the previous block's output, its copy score.",
            args=(
                _UNIT, MODEL_REQ,
                arg("-k", "--top-k", type=positive_int, default=8, metavar="K",
                    help="Tokens per read (default 8)"),
                arg("--hide-hubs", action="store_true",
                    help="Rank by the layer-contrastive z (each token z-scored across "
                         "the layer's units), hiding the hub tokens that top many "
                         "raw reads; the first use of a layer builds its statistics"),
                MODEL_LOAD, JSON,
            ),
        ),
        Verb(
            "top", "Neuron poles that write or fire on a token",
            run=run_top,
            description="Every pole of each layer scored by the token's "
                        "layer-contrastive z through the S and R lenses, weighted "
                        "by the depth rule: the neurons that write the token "
                        "(--by writes, at the block output) or fire on it (--by "
                        "fires, their read direction at the MLP input) most "
                        "specifically.",
            args=(
                arg("token", help="A single-token word, or #<id>"),
                MODEL_REQ,
                arg("--by", choices=("writes", "fires"), default="writes",
                    help="Rank by the pole's write (default) or its read direction"),
                arg("-k", "--top-k", type=positive_int, default=20, metavar="K",
                    help="Poles to list (default 20)"),
                layers_arg("Layers to search (default: all)"),
                MODEL_LOAD, JSON,
            ),
        ),
        Verb(
            "find", "Which neurons a steerable direction is made of",
            run=run_find,
            description="Sparse signed split of the direction at each layer over "
                        "that layer's neuron writes, in the Mahalanobis metric "
                        "(lens decompose --dictionary neurons, with each "
                        "neuron's pole and label).",
            args=(
                arg("selector", help="Steerable direction (concept, ns/name, "
                                     "jlens/<word>, slens/<word>, neuron/…)"),
                MODEL_REQ,
                arg("-k", "--top-k", type=positive_int, default=16, metavar="K",
                    help="Sparsity budget (default 16)"),
                layers_arg("Layer indices or a mode (default: every layer the "
                           "direction covers)"),
                MODEL_LOAD, JSON,
            ),
        ),
        Verb(
            "stats", "How a unit fires: neutral statistics and the catalog's",
            run=run_stats,
            args=(_UNIT, MODEL_REQ, MODEL_LOAD, JSON),
        ),
        Verb(
            "fetch", "Bind the neuron-lens catalog for a model",
            run=run_fetch,
            description="Copy the labels (by Claude Haiku 5.5) and the firing and "
                        "legibility tables of stdlogits/neuron-lens, pinned to a "
                        "commit, for Qwen/Qwen3-8B or google/gemma-4-12B; never "
                        "the bulk tables. Other models have no catalog. Needs "
                        "pyarrow (drowse.ai[hf]); no model load.",
            args=(MODEL, FORCE.but(help="Fetch again over a current binding"), JSON),
        ),
        Verb(
            "ls", "List the catalog binding of one model, or of every catalogued model",
            run=run_ls,
            args=(arg("model", nargs="?", default=None, help="Model ID (default: every catalogued model)"), JSON),
        ),
        Verb(
            "rm", "Remove a model's catalog binding",
            run=run_rm,
            args=(MODEL, YES),
        ),
    ),
)
