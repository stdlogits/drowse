# Drowse

[![CI](https://github.com/stdlogits/drowse/actions/workflows/ci.yml/badge.svg)](https://github.com/stdlogits/drowse/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Drowse is a local workbench for mechanistic interpretability on large language
models.

It comes with a local dashboard, along with a Python API, and a server compatible
with both OpenAI and Ollama.

## Quick start

The PyPI distribution name is `drowse.ai`; the Python import and CLI are
`drowse`.

```bash
pip install drowse.ai
drowse serve google/gemma-3-4b-it
```

Open [http://localhost:8000](http://localhost:8000).

The first launch downloads the model and fits the 17 bundled concept probes. This
can take a while; they get stored in `~/.drowse/` for future launches.

For NVIDIA CUDA:

```bash
pip install "drowse.ai[cuda,flash]"
drowse serve google/gemma-3-4b-it --device cuda
```

## Browser edition

[drowse.ai](https://drowse.ai) runs the same dashboard in the browser, with
inference, steering, probes, and the R lens on the device through WebGPU and no
server. Its source, the in-browser runtime, and its release tooling live in
[stdlogits/drowse.ai](https://github.com/stdlogits/drowse.ai), which pins this
repository as a submodule for the shared dashboard and the Python reference
implementation. The Python server and the bundled dashboard described here are
independent of it.

## The dashboard

The header holds six rooms: **Workbench**, **Units**, **Lens**, **Probes**,
**Artifacts** and **Model**, and its search (⌘K) finds any room, tool or kind
of steering by what it does ("fit a lens", "steer a neuron"). The workbench is
where you talk to the model: the thread (the
path you are on through the conversation, and your earlier chats), the page
(the conversation, or the loom: every turn as a node in a branching tree you
can reroll, compare and fork from) and the Inspector. They sit side by side on
a wide screen and fold into one column on a phone, with the Inspector as a
sheet.

The Inspector shows what you selected. With nothing selected it holds the
instruments, neurons first: the unit probes, the live lens readout, the
geometry probes, the SAE's features, sampling, and one line for each source the
lens and SAE read through.

- **A token.** Click any token for its alternatives with logprobs (and a fork
  from any of them), its anatomy (the neurons and heads that pushed it, each
  pushed or silenced in a click, then what the residual reads as through both
  lenses layer by layer and what attention and the MLP, and the steering,
  added) and its probe and feature readings. The readings can be replayed
  without the steering for comparison.
- **A probe** opens its fitted geometry, layer by layer.
- **A unit** (a neuron or head) opens what it writes and fires on, its label
  and its activity over the reply.

**Steering** is built from chips, in one place over the composer. **+ Steer**
adds a neuron or head (by ID or by a word it writes), a lens token, a concept,
a manifold position or an SAE feature; the instruments' **Push** and
**Silence** add one in a click, and the chip opens under the strip without
leaving what you were looking at. A chip's editor sets a signed strength, the
layers it acts at (a depth strip), when it acts (the prompt, the response, the
first or all-but-the-first N tokens) and "Only while…", a gate on an attached
probe's reading; a concept can also be kept along, or removed from, another
(`~`, `|`), and any direction or unit can be removed (`!`). The dashboard never
takes typed recipe text: a recipe it cannot draw as chips (from an agent or the
API) shows as one read-only chip and is sent as written. Agents, the HTTP APIs,
the CLI and Python keep the full expression language below.

A reply cut short by Stop (or a dropped connection) is kept and marked
"stopped early"; a browser that reads slower than the model writes never cuts
one, and catches up from the server's conversation tree. Saved chats live in the browser (saved automatically between
replies) and as backup files you download; the server keeps no conversations.

The rooms beside the workbench read the conversation token by token. A strip
of its tokens, prompt tokens included, sits at the top of each: step with ←
and →, jump with Home and End, click a token, or read the whole reply. The
workbench and the rooms share that cursor, so a token chosen in one is the
token in the other.

| Room | Purpose |
|---|---|
| **Units** | The neurons and heads strongest at the token (or at their peak over the reply), a whole layer at once, and the chosen unit's activity, writes, Push and Silence |
| **Lens** | Each layer at the token through both lenses (what the S lens reads, what the R lens says, weighted by depth), steered or not, with the lens probes |
| **Probes** | The linear probes and vectors (the value at the token, the trace over the conversation), then the manifolds (where the token sits in the fitted geometry), then SAE features |
| **Artifacts** | Where everything is prepared: the neuron-lens unit catalog, the lenses (fetched or fitted, both families at both sites), probes (built, installed, combined), templates, roles and SAEs |
| **Model** | The model's layers, neurons and heads, device and versions, what is prepared for it, health and API access |

Chat mode renders the model's chat template as turns, with collapsible
thinking, any role label a compatible template supports, and a cast that
assigns steering to roles; raw mode is one buffer for base models.

## Concepts, subspaces, and manifolds

In Drowse, you extract concepts as **manifolds** or **subspaces**.

- A 1D flat subspace is just a steering vector: the difference of the two
  poles' means.
- A higher-rank flat subspace is a group of steering vectors, orthogonal in
  the model's Mahalanobis metric.
- A curved manifold fits a nonlinear surface and lets you steer along it.

### Bundled concepts

Drowse comes with 17 concepts that are attached as probes by default: 14
two-pole axes and 3 one-pole concepts (`deceptive`, `refusing`,
`manipulative`):

| Category | Concepts |
|---|---|
| **Epistemic** | `confident.uncertain`, `deceptive`, `curious.disinterested` |
| **Alignment** | `refusing`, `sycophantic.blunt`, `manipulative` |
| **Register** | `formal.casual`, `direct.indirect`, `verbose.concise`, `creative.conventional`, `humorous.serious`, `warm.clinical`, `technical.accessible` |
| **Cultural** | `masculine.feminine`, `individualist.collectivist`, `traditional.progressive`, `religious.secular` |

A one-pole concept is measured from the model's own default answers, which
are already honest, compliant and sincere, so it has no opposite pole to
write: a positive strength pushes toward it and a negative one past the
default (`-0.5 deceptive` is more honest than the model's default).

Three larger concept sets ship as well, but aren't fitted by default:

- **`personas`** — 107 personas;
- **`emotions`** — 20 emotional states;
- **`months`** — 12 months.

## Steering expressions

Every surface uses the same expression format. The same recipe can be used
across Python, YAML, OpenAI, Ollama, or the native API.

```text
-0.3 deceptive + 0.4 warm
0.5 formal - 0.2 verbose
0.3 deceptive|sycophantic.blunt
0.3 deceptive~confident.uncertain
!sycophantic.blunt
0.5 personas%pirate
0.7,0.4 months%january@after
0.4 warm@when:confident.uncertain>0.4
0.3 jlens/orange + 0.2 sae/9143
0.5 deceptive{12-20} + 0.4 personas%hacker{workspace}
!jlens/Paris{24-30}@response
```

| Syntax | Meaning |
|---|---|
| `+`, `-` | Add or subtract terms |
| leading number | Steering coefficient; omitted terms default to `0.5` |
| `~` | Keep the component shared with another direction (Mahalanobis) |
| `\|` | Remove the component shared with another direction (Mahalanobis) |
| `!` | Ablate a direction toward neutral (LEACE: reads with the whitened covector, edits along the direction) or mean-ablate a unit; a bare `!x` ablates fully |
| `%label` or `%x,y,…` | Choose a named node or coordinates on a manifold |
| `{12-20}`, `{4-8,20}`, `{workspace}` | Act only at these layers (after the selector, before `@`) |
| `@prompt`, `@response`, `@after`, `@thinking`, … | Restrict the token phase where a term applies |
| `@first:N`, `@after:N` | Restrict a term to a counted decode window |
| `@when:<probe><op><value>` | Apply a term only while a live probe gate is true |
| `jlens/<word>`, `jlens/#<id>`, `sae/<id>` | Steer toward saying a token (the whitener-raised vector of its R-lens read) or along an SAE decoder row |
| `slens/<word>`, `slens/#<id>` | Steer toward having read a token: its S-lens footprint at each layer, raised to a vector |
| `neuron/L8n344`, `head/L20h17` | A unit: push along a neuron's write, or mean-ablate a neuron or head (`!head/L20h17`) |

A bare pole or node label such as `formal` or `pirate` resolves to its concept
or manifold; an ambiguous name raises instead of guessing. The operands of `~`,
`|` and `!` must reduce to one direction: a whole two-pole concept
(`sycophantic.blunt`, not `sycophantic`), a one-pole concept (`deceptive`), an
extracted vector, or a
`jlens/`/`slens/`/`sae/` atom (`!` also takes a unit). A head has no push
direction, so `head/` takes only `!`. `@response` covers every generated token, thinking
included; `@after` applies only after the thinking block closes. A gate names
an attached probe, so `confident.uncertain` above reads the bundled probe. The
larger sets (`personas`, `emotions`, `months`) steer once they are fitted for
the model.

Without a layer clause a term acts on every layer it is fitted at. A clause
names layers (`{12-20}`, `{8,10,12}`), the model's 40–90% depth band
(`{workspace}`, intersected with the fitted layers) or `{all}`; naming a layer
the term is not fitted at is an error. The coefficient keeps its meaning under
a clause: a one-layer selection steers about as hard per layer as the whole
fit. `neuron/` and `head/` atoms carry their own layer and take no clause.

Manifold coefficients use two coordinates: `along` and `onto`. `along` controls
movement within the manifold toward the target; `onto` reduces the off-surface
component inside the manifold's fitted tube. For a flat subspace, aim for a
coefficient near `0.5` and raise it toward `1.0` for stronger expression.

## How Drowse works

### Extraction

Drowse first has the model answer a shared set of baseline prompts as each concept,
then it takes the resulting hidden states and fits them to either a curved manifold
or a flat subspace.

Fitting, layer allocation and reads use a Mahalanobis metric estimated from
neutral activations. Each fitted frame keeps two row sets: write rows, the
directions steering adds (for a two-pole concept, the difference of means),
and read rows, the whitened covectors that measure a coordinate (the LDA
direction). A full `!` ablation reads with one and edits along the other,
which is LEACE.
Discriminative layer selection removes flat axes that fail to straddle the neutral
baseline across the fitted nodes.

The full data flow, artifact boundaries, instrument protocol, and concurrency
invariants are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

### Monitoring

A reading includes fitted coordinates, the centered activation's subspace
fraction, the nearest nodes and their soft assignment, and, for curved manifolds,
an off-surface residual and tube membership.

### Jacobian lenses

The Jacobian lens implementation follows Gurnee et al.'s
[work](https://transformer-circuits.pub/2026/workspace/index.html).
Drowse lets you use one of the published lens artifacts, or fit your own.

A lens reads a vector inside the model as a ranking over tokens, in one of two
ways. The R-lens (or the standard J-lens) reads what the vector is disposed
to make the model *say*; the S-lens (or the standard K-lens) reads which input
tokens it looks like it has *read*. Each comes at two places in a layer: `out`,
the block's output (also where an MLP neuron writes and where a head reads,
one layer up), and `mid`, the residual after attention (where a neuron reads
and a head writes). Which one to trust is a question of depth: the S read is
the faithful one early in the model, the R read late, and both show, weighted,
in between. `drowse lens top` reads every layer of a prompt that way and labels
each read with the lens it used; `session.lens.read(vector, layer, site,
of=...)` does the same for any vector. The S read reports a footprint z, each
token's score as a z-score over the vocabulary. A residual (the stream itself)
carries a large component every position shares, so its S read subtracts the
model's neutral layer mean first (`of="residual"`); a direction such as a
steering vector or a neuron's write is read as it is (`of="direction"`).

[`stdlogits/rs-lenses`](https://huggingface.co/stdlogits/rs-lenses) publishes
both lenses at both places for Qwen3-8B and Gemma 4 12B (`lens fetch MODEL
rs-r` and `rs-s`). Drowse keeps one R-family and one S-family lens selected per
model. `jlens/<word>` steers toward saying a token and `slens/<word>` toward
having read it; as probes, `jlens/` reads the mean readout probability and
`slens/` the footprint z averaged over the layers where the S lens is the
faithful read (weighted by depth, early layers most), and either can gate a
term (`@when:slens/owl > 3`).

An S read ranks only tokens the model saw often enough in text: a token it
almost never saw has an untrained embedding that would top early reads. The
counts come from the neuron-lens catalog when it is bound (`unit fetch`), else
from `drowse lens counts MODEL`, which streams the S lens's corpus through the
tokenizer (no model load); without either, the tokenizer's own rules decide,
and `lens show` and `lens top` say so.

### Neurons and heads

Every MLP neuron (`neuron/L8n344`) and attention head (`head/L20h17`) is an
object Drowse reads, probes, gates on and steers. `drowse unit read` says what
one is about from its weights alone, through the same depth rule as the lens
reads: a neuron per pole (its activation takes both signs) *writes* (its
output direction, read where it lands, at the block output) and *fires on*
(its input direction, read at the MLP input); a head *copies* or
*suppresses* tokens (its OV circuit) and has a query, key and value that pick
up tokens at the layer below. Hub tokens that top many reads at once from the
lens's geometry can be hidden (`--hide-hubs`), which ranks each token against
the rest of the layer. `drowse unit top owl --by writes` finds the neurons that
write a token most specifically, and `drowse unit find deceptive` the neurons a
steering direction is made of.

For Qwen3-8B and Gemma 4 12B the
[neuron-lens catalog](https://huggingface.co/datasets/stdlogits/neuron-lens)
adds a label to every neuron pole and head, written by Claude Haiku 5.5, and
how often each fires; `drowse unit fetch MODEL` binds it (labels and small
tables only, pinned to a commit), and every unit view then links to the unit's
page on [neuron-lens.com](https://neuron-lens.com). A unit's readings then
divide by the catalog's typical high firing, so 1.0 means "as strong as this
unit usually fires in text"; without a catalog they divide by its typical
firing on Drowse's neutral statements, a smaller sample, so the same firing
reads several times larger, and every view says which.

### Sparse autoencoders

Drowse can either use a published SAELens release or train a local SAE.

## Installation

Drowse requires Python 3.11 or newer and PyTorch 2.13 or newer. CUDA or Apple
Silicon MPS is strongly recommended for interactive use; CPU is supported for
smaller models and non-GPU workflows.

```bash
pip install drowse.ai
```

The base package includes the HTTP server and the prebuilt Svelte WebUI.
Optional extras add specialized workflows:

| Extra | Adds |
|---|---|
| `flash` | FlashAttention 2 for supported NVIDIA CUDA models; stable and tested |
| `cuda` | `bitsandbytes` quantization and Hugging Face `kernels` acceleration |
| `sae` | SAELens 6.x, for published SAE releases (`sae fetch`); local SAEs need no extra |
| `hf` | `datasets` for streamed lens and SAE corpora |
| `gguf` | GGUF import/export support |
| `research` | `datasets`, NumPy, SciPy, scikit-learn, pandas, Matplotlib, and image helpers |
| `notebook` | Plotly, pandas, and Kaleido notebook helpers |
| `dev` | Test, lint, type-check, and build tooling |

Extras can be combined:

```bash
pip install "drowse.ai[cuda,flash]"     # full tested NVIDIA path
pip install "drowse.ai[sae,hf]"         # SAELens sources and streamed corpora
pip install "drowse.ai[hf,research]"    # dataset-backed research workflows
pip install "drowse.ai[notebook]"       # interactive figures
```

`cuda` and `flash` are Linux/NVIDIA CUDA extras. FlashAttention is selected
automatically when installed; there is no runtime flag to enable it.

From source:

```bash
uv pip install -e ".[dev]"
```

## Running the server

```bash
drowse serve MODEL [options]
```

Repository Python is disabled by default. For a trusted model that requires custom
code, use `DROWSE_TRUST_REMOTE_CODE=1 drowse serve MODEL`; Python callers can
pass `trust_remote_code=True` to `DrowseSession.from_pretrained`. This grants
the model repository permission to execute Python locally.

Common options:

| Option | Default | Purpose |
|---|---:|---|
| `-d`, `--device` | `auto` | `cuda`, `mps`, `cpu`, or automatic selection |
| `-q`, `--quantize` | none | `4bit` or `8bit` bitsandbytes quantization on CUDA |
| `--dtype` | device default | `float32`, `bfloat16` or `float16` weights (default bf16 on CUDA/MPS, fp32 on CPU); a fit records the dtype it ran at |
| `-p`, `--probes` | `all` | Bundled probe categories, `all`, or `none` |
| `-H`, `--host` | `127.0.0.1` | Bind address; non-loopback requires an API key |
| `-P`, `--port` | `8000` | Bind port |
| `-S`, `--steer` | none | Default steering expression |
| `-C`, `--cors` | none | Allowed CORS origin; repeatable |
| `--top-k-alts` | `0` | Alternative tokens captured at each decode step |
| `--compile` | off | Opt into `torch.compile` after Drowse probes the path |
| `--cuda-graphs` | off | Pair static cache and CUDA graph capture with `--compile` |
| `-k`, `--api-key` | none | Require bearer authentication; also reads `$DROWSE_API_KEY` |
| `--no-web` | off | Run the APIs without mounting the dashboard |
| `-c`, `--config` | none | Setup YAML; repeatable, later files win |

`serve` and every subcommand that accepts `-c/--config` read
`~/.drowse/config.yaml` first and then compose any explicit `-c PATH` files on top.
For example:

```yaml
model: google/gemma-3-4b-it
vectors: "-0.3 deceptive + 0.2 warm"
temperature: 0.8
top_p: 0.9
max_tokens: 512
return_top_k: 8
```

Command-line flags override YAML, and `vectors:` sets the server's default
steering only; the `experiment` verbs use exactly their own `-S`. Inspect the
resolved configuration with `drowse config show` and validate a file with
`drowse config validate path.yaml`.

## Command-line artifact workflows

The CLI has nine top-level verbs. The WebUI covers the interactive versions of
most workflows; the CLI is useful for reproducible preparation, distribution, and
batch work.

| Verb | Role |
|---|---|
| `serve` | Launch the WebUI and the three HTTP protocol surfaces |
| `manifold` | Extract, generate, derive from a template, fit, bake, merge, transfer, compare, or diagnose manifolds |
| `pack` | List, inspect, install, search, push, clear, refresh, remove, or export manifold packs |
| `experiment` | Run alpha fans, replay transcripts, and evaluate naturalness |
| `config` | Show or validate composed configuration |
| `template` | Create and score restricted-choice completion templates |
| `lens` | Fit, fetch, select, read, decompose, or remove Jacobian lenses |
| `sae` | Train, fetch, select, inspect, or remove SAE sources |
| `unit` | Read neurons and heads from their weights, find the neurons behind a token or a direction, and fetch, list, or remove the neuron-lens catalog |

Representative commands:

```bash
# Extract and fit a two-pole concept for one model
drowse manifold extract patient impatient -m google/gemma-3-4b-it

# Fit a bundled many-node manifold
drowse manifold fit personas -m google/gemma-3-4b-it

# Install or publish manifold packs through Hugging Face
drowse pack search creativity
drowse pack install OWNER/REPO
drowse pack push local/patient.impatient -a OWNER/REPO -m google/gemma-3-4b-it

# Move one manifold closure, plus its referenced template, as an archive
drowse pack export archive local/patient.impatient -o patient.drowse
drowse pack install patient.drowse

# Fetch a provider lens (Neuronpedia's J-lens by default), or fit a local one:
# --kind r (the default) is the RelP R-lens at local:r, --kind j the standard
# J-lens; --kind s and k are the S family's S- and K-lenses. Each fits the
# block output (out) and the MLP input (mid) unless --sites says otherwise
drowse lens fetch google/gemma-3-4b-it
drowse lens fit org/model
drowse lens fit org/model --kind s --prompts 100
drowse lens fit org/model --kind j --sites out --layers workspace
# Or on another commit-pinned Hub dataset (a shuffled one grows by nested samples)
drowse lens fit org/model --hf-corpus HuggingFaceFW/dclm_100BT-shuffled --prompts 1000
# A bilingual mix: 3 English documents per Chinese one, the Chinese side spread
# over 64 of its (unshuffled) data files
drowse lens fit org/model --prompts 1000 \
  --hf-corpus HuggingFaceFW/dclm_100BT-shuffled,weight=3 \
  --hf-corpus HuggingFaceFW/fineweb-2:cmn_Hani,files=64
# Pick which lens the session reads (one R-family and one S-family source)
drowse lens use org/model local:r
# rs-lenses' R- and S-lens, both sites, for Qwen3-8B and Gemma 4 12B
drowse lens fetch Qwen/Qwen3-8B rs-r
drowse lens fetch Qwen/Qwen3-8B rs-s
# Without a neuron-lens catalog, count the S lens's corpus for the S token mask
drowse lens counts org/model
# Read a prompt's residuals through both (--site mid: after each attention)
drowse lens top Qwen/Qwen3-8B "The capital of France is" --site out
# Which input tokens' footprints a direction resembles
drowse lens decompose deceptive -m Qwen/Qwen3-8B --dictionary tokens-in

# Neurons and heads: what one is about, who writes a token, what a direction is made of
drowse unit fetch Qwen/Qwen3-8B                      # the neuron-lens labels (Qwen3-8B, Gemma 4 12B)
drowse unit read neuron/L8n344 -m Qwen/Qwen3-8B --hide-hubs
drowse unit read head/L20h17 -m Qwen/Qwen3-8B
drowse unit top owl -m Qwen/Qwen3-8B --by fires --layers 0-12
drowse unit find deceptive -m Qwen/Qwen3-8B

# Fetch an SAE (needs the [sae] extra) or train a local source
drowse sae fetch google/gemma-3-4b-it saelens:gemma-scope-2-4b-it-res
drowse sae train org/model my-sae --layer 20 --tokens 1000000
```

A local fit averages 64 documents by default and prints its split-half
difference per site (how differently two halves of the corpus read: the
signal for whether more documents would help). Fits run on dense models
(llama, mistral, qwen2/3, gemma 2/3/4 text); mixture-of-experts and hybrid
stacks are refused, though their published lenses still load. Lens fits are
resumable: a fit checkpoints every 25 documents and every ten minutes, Ctrl-C
or SIGTERM finishes the document in flight and saves before exiting (a second
signal stops at once), rerunning the command continues, and an exact repeat
is recognized without loading the model. A fit never silently replaces a
local lens over more documents or more layers: a 5.4 lens fitted with the
old default of 100 documents needs `--prompts 100` (which adds the `mid`
site) or `-f` to replace it. Run `drowse <verb> -h` and
`drowse <verb> <subcommand> -h` for the complete flag surface.

## HTTP APIs

The same `drowse serve` process exposes four surfaces on one port:

- `/` — the Drowse WebUI;
- `/v1/*` — OpenAI-compatible models and chat completions;
- `/api/*` — Ollama-compatible generation and chat;
- `/drowse/v1/*` — native sessions, loom trees, probes, manifolds, templates,
  lens, SAE and unit lifecycles and reads, replay, SSE, and token-plus-measurement
  WebSockets.

Interactive OpenAPI documentation is available at
[http://localhost:8000/docs](http://localhost:8000/docs).

### OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Describe a rainy afternoon."}],
    extra_body={"steering": "0.3 warm + 0.2 concise"},
)
print(response.choices[0].message.content)
```

### Ollama client

```bash
curl -N http://localhost:8000/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Write a short haiku."}],
  "options": {"steer": "0.3 warm - 0.2 formal.casual"}
}'
```

Drowse targets a trusted local machine or lab network. It is not a hardened
multi-tenant inference service. If you bind it beyond a trusted host, read
[SECURITY.md](SECURITY.md), set an API key, and add TLS, rate limits, request
limits, and isolation outside Drowse.

## Python API

```python
from drowse import DrowseSession, SamplingConfig

with DrowseSession.from_pretrained(
    "google/gemma-3-4b-it",
    device="auto",
    return_top_k=8,
) as session:
    name, profile = session.extract("patient", baseline="impatient")
    session.add_probe(name)

    result = session.generate(
        "How should I learn a difficult skill?",
        steering=f"0.3 {name} + 0.2 concise",
        sampling=SamplingConfig(
            temperature=0.8,
            top_p=0.9,
            max_tokens=256,
            seed=42,
        ),
    ).first

    print(result.text)
    print(result.applied_steering)
    print(result.probe_readings[name].coords)
```

`generate` and `generate_stream` accept the same steering expression as the WebUI;
`session.steering(expr)` applies one to every generation inside a `with` block.
Generation returns a list-like `RunSet`; `.first` is convenient for a single
completion. `GenerationResult` carries text, token IDs, throughput and timing,
finish reason, the canonical applied expression, captured log probabilities,
aggregate probe readings, and the versioned measurement envelope every
instrument family writes into.

Batch and sweep helpers return the same result shape:

```python
batch = session.generate_batch(
    ["Describe a sunset.", "Describe a storm."],
    steering="0.3 warm",
)

sweep = session.generate_sweep(
    "Describe a forest.",
    sweep={"warm.clinical": [-0.4, 0.0, 0.4]},
)
```

Restricted-choice scoring reads the model's raw probability distribution over a
set of candidate completions, optionally under steering, so you can see whether
steering moved the distribution and not just the top choice:

```python
scores = session.score_choices(
    [{"role": "user", "content": "The first weekday is"}],
    ["Monday", "Tuesday", "Wednesday"],
    steering="0.3 confident",
)
```

The four read families are available as `session.geometry`, `session.lens`,
`session.sae` and `session.units`; `session.add_probe` attaches a concept, a
`jlens/<word>` or `slens/<word>` token probe, an `sae/<id>` feature or a
`neuron/L8n344` / `head/L20h17` unit to the right one.
`session.lens.read(vector, layer, site, of="residual" | "direction")` reads any
vector through both lens families at once.

`session.token_anatomy(node_id, raw_index)` replays one token and reads it
layer by layer: what the residual reads as through both lenses, what
attention and the MLP (and any steering) added, the neurons and heads that
pushed the token through the R lens, and where the top heads looked.
`session.unit_activity(node_id)` reads every neuron and head over a turn's
tokens: the strongest units with their peak token, per-token traces of named
units, and one token's whole population. Both read a prompt turn too (its
tokens replay through the turn's recorded capture). Both are also served at
`/drowse/v1/sessions/{id}/instruments/token-anatomy` and
`.../instruments/units/activity`.

Notebook helpers are available from `drowse.notebook` after installing
`drowse.ai[notebook]`: `plot_alpha_sweep`, `plot_probe_correlation`,
`plot_layer_norms`, `plot_trait_history`, and `to_dataframe`.

## Model support

Drowse has end-to-end tested paths for:

- Qwen 2, Qwen 3, and Qwen 3.5, including its text and MoE variants;
- Gemma 2, Gemma 3, and Gemma 4, including text-only extraction from supported
  multimodal checkpoints;
- Mistral 3 and Ministral 3;
- Llama, GLM, gpt-oss, and Talkie.

Other decoder-only models load when their transformer blocks sit at a standard
Transformers location, which covers most families (Mixtral, Phi, Cohere,
DeepSeek, OLMo, Granite, GPT-2, Falcon, MPT, OPT, and others). Drowse warns once
when an architecture has not been exercised end to end.

CUDA and Apple Silicon MPS both have real-model smoke coverage. Model-specific
features still depend on the checkpoint: role-augmented extraction and persona
regeneration need a chat template with a substitutable assistant label (Mistral's
does not have one), published SAE and lens sources cover only some models, lens
fits and unit reads refuse mixture-of-experts and hybrid stacks, and
FlashAttention depends on the model's Transformers attention implementation.

## State and distribution

Drowse keeps local state under `~/.drowse/`; set `$DROWSE_HOME` to move it. The
store contains authored manifolds, per-model fits and neutral statistics, local
lens and SAE artifacts, source and catalog bindings, and templates. It holds no
conversations: the dashboard keeps its saved chats in the browser and as
downloaded backup files, and Python saves a tree only to a `LoomTree.save()`
path you choose.

Manifold packs are folders with metadata, node corpora and optional fitted
tensors. They can be installed from a local path or distributed as
Hugging Face model repositories. A `.drowse` is the narrow ZIP transport for
exactly one manifold closure and its referenced template, when present; export it
with `drowse pack export archive` and install it with `drowse pack install`.
It never contains model weights, lenses, or SAEs. A fitted two-node PCA manifold
can instead be exported as a llama.cpp control-vector GGUF with
`drowse pack export gguf`.

A Hugging Face pack is a model repository tagged `drowse-manifold` that holds
exactly one `.drowse` at its root, a root `manifold.json` summary, and a model
card; `drowse pack push` writes that layout, and both drowse and the drowse.ai
browser edition install from it. An install pins the repository to an immutable
commit and records that commit as the manifold's source; the browser also checks
that the archive's declared repository and revision match it.

Treat model repositories and downloaded artifacts as executable or otherwise
untrusted input. Drowse bounds archive paths and sizes and checks tensor headers;
it does not establish who published a pack.

Drowse reads its current artifact formats, ignoring fields it doesn't use, and
does not migrate older format versions.
Releases through 5.3.0 were published as `saklas`, and drowse does not read a
`~/.saklas` directory. Lens artifacts fitted or fetched before 5.4 must be
converted once from a source checkout with
`python scripts/migrate_lens_v7.py --home ~/.drowse` (try `--dry-run` first).
6.0 recomputes its neutral statistics on first load and does not read 5.x
saved conversations or transcripts; manifolds, templates, packs and lenses
carry over unchanged. 6.1 changed the fitted frame (writes and reads are typed
separately), so fitted manifolds refit the next time they are used.

## Development

```bash
pip install -e ".[dev]"

ruff check .
pyright
pytest -q -m "not gpu"              # the CI suite
pytest -q -n auto -m "not perf"     # everything this machine can run
pytest -m perf -p no:xdist          # throughput gates, serially
python -m build
```

GPU integration tests download the public SmolLM2-360M-Instruct weights by
default. Set `DROWSE_TEST_MODEL` to exercise another compatible model:

```bash
pytest -q tests/gpu/test_smoke.py
DROWSE_TEST_MODEL=google/gemma-3-4b-it pytest -q tests/gpu/test_smoke.py
```

The WebUI is a Svelte 5 + Vite application in `webui/`. Its compiled bundle under
`drowse/web/dist/` is committed package data and ships in the wheel.

```bash
cd webui
npm ci
npm run check
npm run build
git diff --exit-code ../drowse/web/dist
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and adding a
model architecture.

## Research lineage and credits

Drowse builds on Representation Engineering
([Zou et al., 2023](https://arxiv.org/abs/2310.01405)).
[repeng](https://github.com/vgel/repeng) by Theia Vogel is the best-known compact
implementation of that approach; Drowse takes the workbench route, adding live
monitoring, manifold geometry, branching experiments, and server protocols.

Two-pole extraction uses difference-of-means following
[Im & Li, 2025](https://arxiv.org/abs/2502.02716). Manifold steering follows
[Goodfire's manifold work](https://arxiv.org/abs/2605.05115). The `personas`
source is derived from the framing in Anthropic's
[Assistant Axis paper](https://arxiv.org/abs/2601.10387). The Jacobian lens implements
the verbalizable-workspace method of
[Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html).
Unit reads follow [neuron-lens](https://neuron-lens.com)'s definitions (its
layer-contrastive reads, Elhage et al.'s copying score, Gurnee et al. 2024's
prediction neurons, Stolfo et al. 2024's entropy neurons); its catalog's labels
are by Claude Haiku 5.5.

Drowse is written by a9lim with Claude. The dashboard is drawn in
[Ledger](https://github.com/stdlogits/ledger), the Standard Logits design system.

If you use Drowse in published research, please cite the relevant upstream methods
alongside the Drowse version and exact model checkpoint you used.

## Contact, issues, and security

For questions, feedback, or research inquiries, email
[contact@drowse.ai](mailto:contact@drowse.ai) or use the
[contact form](https://drowse.ai/contact).

Please update to the latest Drowse release before filing a bug. Include the model
ID, device, dtype or quantization mode, Drowse version, and a minimal reproduction
in [GitHub Issues](https://github.com/stdlogits/drowse/issues).

Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## License

Drowse is licensed under AGPL-3.0-or-later. See [LICENSE](LICENSE).
