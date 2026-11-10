# model/

Everything drowse knows about a Hugging Face causal LM below steering: loading
and identity, where the residual blocks are and what they compute
(`BlockSpec`), unit directions, the logits path, layer selectors, token spans,
chat-template facts, role substitution, the scene stitcher, and
compile/StaticCache plumbing. Layer 3: imports only `errors`/`naming`/`events`,
`base` and `geometry`, holds no session state. Render policy lives in
`generation/decode.py::build_chat_input` and `generation/render.py`; this
package supplies the mechanisms.

| File | Owns |
|---|---|
| `load.py` | `load_model`: `LoadPlan` -> materialize -> finalize |
| `checkpoint.py` | safetensors reader; multimodal text-model extraction |
| `device.py` | device/dtype defaults, `memory_gb`, `patch_torch_for_mps` |
| `fingerprint.py` | source and loaded-model identities, `resolve_model_revision` |
| `arch.py` | `get_layers`, `TESTED_ARCHS`, `text_extractable`, `config_model_shape`, `workspace_layer_indices` |
| `readout.py` | final residual to logits (`LogitReadout`, `get_unembedding`, `get_final_norm`) |
| `blocks.py` | `BlockSpec`/`ModelSpec`: dense decoder blocks described for transport and unit reads; `model_spec`, `block_specs`, `verify_model_spec`, reference `block_forward` |
| `units.py` | neuron writes and pole reads, head write and read rows, from a `BlockSpec` (fp32, CPU) |
| `layers.py` | the one layer-selector parser, formatter and resolvers |
| `tokens.py` | `special_token_ids`, `last_content_index` |
| `chat.py` | `ChatTraits`: per-session chat facts and render cache |
| `role_templates.py` | `SEAT_HEADERS`, `resolve_seat_headers`, `validate_role`, the splice renderers |
| `scene.py` | template autopsy -> `TurnGrammar`; `render_scene`, `validate_turn_grammar` |
| `compile.py` | `compile_with_probe`: `torch.compile` plus warmup probes, eager fallback |
| `static_cache.py` | StaticCache support probes and the one factory |

`DrowseSession.from_pretrained` calls `load_model`; then
`CompiledRuntime.prepare` (`generation/compile_runtime.py`) calls
`compile_with_probe` and the StaticCache probes (CUDA/MPS only).
`DrowseSession.__init__` builds one `ChatTraits` and resolves `get_layers` once.

## Loading

`_resolve_load_plan` decides everything from config reads alone (no weight I/O,
so `LoadPlan` is unit-testable); `_materialize_model` reads weights;
`_finalize_model` stamps identity and freezes the model.

- **Revision pin.** The Hub commit is resolved once (`None` for a local
  directory; fails open to an unpinned read) and every config, tokenizer and
  weight read passes it, so a moving branch cannot pair config A with weights B.
  Finalize writes it to `config._commit_hash` (and the `text_config`'s), which
  lens bindings and capture fingerprints read.
- **Remote code** is off. `trust_remote_code=None` reads
  `DROWSE_TRUST_REMOTE_CODE` (`1`/`true`/`yes`/`on`; no CLI flag); explicit bools
  win. Even when trusted it is dropped for a `model_type` native to transformers
  (stale `auto_map` code breaks). Metadata-only reads never trust remote code.
- **dtype** defaults to bf16 on CUDA and MPS, fp32 on CPU; fp16 is avoided
  because some residual streams leave its range (Gemma-3-4b reaches ~1e5).
  Quantization (`4bit` nf4, `8bit`) is CUDA-only, dropped with a warning
  elsewhere.
- **Attention** `sdpa`; `flash_attention_2` on CUDA when `flash_attn` imports;
  `eager` on MPS for MLA models (`deepseek_v2`/`v3`: mismatched q/v head dims
  break MPS SDPA). Mistral tokenizers get `fix_mistral_regex=True` (the shipped
  regex mis-splits ~1% of tokens).
- **Text extraction** applies when `text_config.model_type` is a transformers
  causal-LM type (`arch.text_extractable`), even if the composite loads; quantized
  loads skip it.

Fallback ladder: unsupported attention -> `eager`; any other failure of an
unquantized load -> fp16 (CUDA) or fp32; an MPS weight-`CONVERSION` failure ->
load on CPU, then move. Retries narrow `plan.load_kwargs` in place. Not retried:
`_IncompleteCheckpointError` (no freshly initialized parameter ever ships) and a
quantized composite that only text extraction can load.

`load_text_from_multimodal` builds the text model with `from_config` and streams
shards tensor by tensor into it: ~1x peak memory on unified memory, where
`from_pretrained`'s stage-then-copy costs ~2x (>128 GB for gemma-4-31B). Reads
are chunked `pread`s on an `open_uncached` fd (`F_NOCACHE` on macOS, so the
checkpoint does not fill the page cache MPS shares). It strips the first
`language_model.` segment, keeps `lm_head.*`, dequantizes FP8 by
`<name>_scale_inv`, and raises `NoTextWeightsExtracted` on no match (falling back
to the full multimodal load). It is the one hand-written checkpoint parser and
validates duplicate keys, size caps, shard-name traversal and tensor extents.

Process-global effects: loads serialize on `_MODEL_LOAD_LOCK`; MPS loads set
`HF_DEACTIVATE_ASYNC_LOAD=1` (transformers' concurrent copies can segfault) and
install `patch_torch_for_mps` once (`histc` on integers via float for MoE
routing, `ldexp` via CPU).

## Identity (`fingerprint.py`)

- `model_source_fingerprint` is provable before any weight is read: sha256 of the
  canonical id, resolved commit, hashes of every regular file in a local snapshot
  (remote code can read anything), the config, CUDA-effective quantization and
  parameter dtype; `None` without a commit or local files, or on a read error.
  Load stamps it as `model._drowse_source_fingerprint`; weight-free preflights
  (`fit/preflight.py`, `fit/transfer.py`, the lens-fit no-op) recompute it, so the
  text-extraction load's in-place config writes (`text_config._name_or_path`, the
  dtype) are mirrored in its payload.
- `loaded_model_fingerprint` identifies the weights a live module holds; fitted
  artifacts bind to it. A clean drowse load hashes config, class, sorted
  parameter/buffer schema and the source stamp; anything else (in-memory, wrapped,
  or changed since load per `_model_state_signature`) also streams a hash of every
  tensor. Memoized per module, unwrapping `_orig_mod`; integrations that write
  through `Tensor.data` or swap adapters call `invalidate_loaded_model_fingerprint`.

## Architectures (`arch.py`, `readout.py`)

`get_layers` finds the block list structurally: the first non-empty
`nn.ModuleList` among `_LAYER_PATHS` (`model.layers`,
`model.language_model.layers`, `model.blocks`, `model.decoder.layers`,
`transformer.h`, `transformer.blocks`, `gpt_neox.layers`). There is no
per-family accessor table; a `model_type` outside `TESTED_ARCHS` warns once.

`TESTED_ARCHS`: `llama`, `mistral`, `ministral3`, `gemma2`, `gemma3`,
`gemma3_text`, `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text`,
`qwen2`, `qwen3`, `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `glm`, `gpt_oss`,
`talkie`. Mistral-Small-3 loads as `mistral` (its `mistral3` composite's text
sub-model); the gemma VLM composites are listed for callers who pass the full
model to `DrowseSession`.

`TESTED_ARCHS` and `SEAT_HEADERS` match `model_type` verbatim, so each variant a
checkpoint can load as needs its own entry (a text-extracted Qwen3.5-MoE loads as
`qwen3_5_moe_text`, in neither table). Adding an architecture: a new block
location is one `_LAYER_PATHS` entry; "tested" means cases in
`test_architecture_dispatch.py` and `test_model_numerics.py` plus the entry; role
and scene support need a `SEAT_HEADERS` entry plus a tokenizer in
`test_role_template_drift.py`. `model_type` branches stay out of capture and
hooks; the forward side's only per-family branches are in `readout.py`.

`config_model_shape` reads `(n_layers, hidden)` from the published config for
weight-free preflights and raises rather than guess. `workspace_layer_indices` is
the canonical workspace band, `0.40 <= l/(n_layers-1) <= 0.90`, falling back to
every candidate on shallow models.

`LogitReadout` is the logits path the R lens reads through without another
forward: final norm (a sibling of the block list), OPT `project_out`, head,
Cohere `logit_scale` or Falcon-H1 `lm_head_multiplier`, Granite
`logits_scaling`, then the `final_logit_softcapping` tanh cap. A model whose
forward ignores `past_key_values` is handled in `generation/decode.py`.

## Blocks and units (`blocks.py`, `units.py`)

A `BlockSpec` describes one dense decoder block semantically, so the lens
transport estimator and the unit reads share one description instead of each
family's attribute names. Every described block computes

```text
h_mid = h + attn_out_norm(o(attention(q(a), k(a), v(a)))),   a = attn_in_norm(h)
h_out = (h_mid + mlp_out_norm(down(act(gate(m)) * up(m)))) * scalar,   m = mlp_in_norm(h_mid)
```

with causal softmax attention over grouped KV heads, optional per-head q/k/v
RMSNorms before RoPE, scores `q.k * scaling` capped by
`logit_softcap*tanh(s/logit_softcap)` when set, and an optional sliding window
(position `i` sees `j` iff `i - window < j <= i`).

| Field | Holds |
|---|---|
| `attn_in_norm`, `mlp_in_norm` | pre-norms (`NormSpec`: module, `eps`, `offset` 0 for `w` or 1 for `1 + w`, `weighted`) |
| `attn_out_norm`, `mlp_out_norm` | Gemma's sandwich norms; `None` elsewhere |
| `attn` (`AttentionSpec`) | `module`, `q`/`k`/`v`/`o` (`v is None`: the value is `k`'s raw output, Gemma 4 full layers), `q_norm`/`k_norm`/`v_norm`, `n_heads`, `n_kv_heads`, `head_dim`, `scaling`, `sliding_window`, `logit_softcap`, `rope_table`, `qkv_bias`, `o_bias`, `kv_head(h)` |
| `mlp` (`MLPSpec`) | `gate`/`up`/`down`, `act` (`silu` or `gelu_tanh`), `d_ff`, `bias` |
| `scalar` | Gemma 4's `layer_scalar` (`scale()` is 1.0 without one) |

`ModelSpec` adds the embedding and `embed_scale` (sqrt(d) on Gemma), final norm,
`final_logit_softcap`, `unembed`, the rotary module, `n_layers`, `d_model`,
`d_ff` (widest block) and `vocab`; `activation`, `activation_factor` (RelP's
`act(g)/g`) and `activation_derivative` are the two activations' math.

- **Discovery is structural.** `model_spec`/`block_specs` find parts by module
  presence and config fields, not `model_type`. A block with
  `pre_feedforward_layernorm` is a sandwich block (its `post_attention_layernorm`
  normalizes the attention output). Norm conventions and the activation are
  identified by running the module on a probe, and each gated MLP is checked to
  compute `down(act(gate(x)) * up(x))`. Specs hold live module references.
- **Discovery is model work**: probes run on the model's device, so the first
  `model_spec` for a model happens inside the exclusive section. It is memoized
  per live model (a `WeakKeyDictionary` whose value pins the submodules and their
  classes); a replaced submodule or swapped class is rediscovered. Unit reads copy
  weight rows to the host, which also needs the guard.
- **Refusals** raise `UnsupportedBlockError` (422) naming the layer (`None` for
  model-level parts) and the finding: MoE MLP or block, fused gate-up or qkv,
  quantized (non-floating) projection weights, linear-attention layers (hybrids),
  attention sinks or other attention tensors, KV sharing, non-causal or NoPE
  attention, a norm that is not an RMSNorm with `w` or `1 + w`, an activation
  other than SiLU or tanh-GELU, unrecognized submodules (Gemma 4 per-layer
  inputs), config multipliers (`*_multiplier`, `logits_scaling`, `logit_scale`)
  other than 1, `llama_4_scaling_beta`.
- **Covered:** `llama`, `mistral`, `qwen2`, `qwen3`, `gemma2`, `gemma3_text` (and
  `gemma3`), `gemma4_text`/`gemma4_unified_text` without per-layer inputs, KV
  sharing or MoE. Refused: `glm`/`glm4`, `qwen3_5`, `qwen3_5_moe`, `gpt_oss`,
  `ministral3`.
- **Verification.** Forward terms the structure does not show are invisible to
  discovery (helium is llama-shaped but rotates RoPE pairs interleaved).
  `verify_model_spec` records each block's call on probe tokens and replays the
  block's own module at fp32 (TF32 off, bf16 rotary tables swapped for the fp32
  table); `block_forward` must reproduce the replay's attention update and
  `mlp_forward` its output within 1e-3 of the update. The replay, not the bf16
  forward, is the reference because bf16 rounding reads as a missing term
  (SmolLM2-360M missed by 19% at layer 4, where massive activations dwarf the
  update); the replay lands below 1e-5. A second replay scales scores until the
  largest reaches 256 and at least 4x a described cap, so a softcap shows
  whether or not the probe reaches it: a described cap the forward does not apply
  is refused, as is one it applies undescribed (Gemma 3 never passes
  `attn_logit_softcapping` to its attention). Verification briefly rescales
  attention modules, so the caller holds the model exclusively and unsteered.
  Transport fits call it; unit reads need only `model_spec`.

Unit directions (`units.py`) are fp32 CPU tensors with neuron-lens's
definitions. Neuron `n` is coordinate `n` of the down projection's input.
`neuron_write(spec, n)` is `s * gamma_out (.) W_down[:, n]` (`gamma_out` the MLP
output norm's gain, `s` the layer scalar), the `+` pole's write as it lands at
`out`; the `-` pole writes its negation. `neuron_read(spec, n, pole)` is `a^ +- b^`
for the unit rows `a = gamma_in (.) W_gate[n]`, `b = gamma_in (.) W_up[n]`, not
renormalized. `neuron_writes`/`neuron_reads`/`neuron_read_rows` are the
whole-layer matrices. `head_write_matrix(spec, h)` is
`gamma_attn_out (.) W_O[:, h*hd:(h+1)*hd]` (`[d, head_dim]`), landing at `mid`
before the scalar; `head_read_rows(spec, h)` gives the query rows and the KV
head's key and value rows (keys as values when `v` is absent), each times the
attention input norm's gain. An index outside the block raises `UnitIndexError`
(400); whether a `NeuronRef`/`HeadRef` names a block of the model is the
caller's check.

## Layer selectors and token spans

`layers.py` is the one parser for `--layers` / `layers=` (CLI, server, lens,
manifold fits) and the grammar's `{...}` clause; it is torch-free at import
because the CLI parses before a model loads. Selectors: `None`/`all`,
`workspace`, `sample` (`SAMPLE_COUNT = 9` evenly spaced, readout only) or an
index list of comma-separated indices and inclusive `a-b` ranges (`4-8,20`).
`parse_layer_selector` keeps written order and raises `LayerSelectorError` (400,
with the failing entry's `offset`) on an empty entry, negative index, reversed
range, repeated layer, or `sample` under `allow_sample=False`.
`format_layer_indices` is the inverse on index sets (sorted, runs as `a-b`), the
form the grammar renders. `resolve_model_layers` (fits) sorts, bounds-checks and
refuses `sample`; a manifold fit and its weight-free preflight both use it, so a
no-op proof covers exactly the fit's layers.

`tokens.last_content_index` is the single "last content token" definition every
single-state read pools from (fit capture, monitor reads,
`generation/read_plane.py`, naturalness). It trims template whitespace only
after a closing special (SmolLM2 renders `...content<|im_end|>\n`), then walks
back past `special_token_ids`, flooring at 0. Whitespace before the closing
marker stays content, since its state predicts the marker (the browser capture
pools the same token). `special_token_ids` includes every `added_tokens_encoder`
id, because some tokenizers (talkie's `<|user|>`/`<|end|>`) add turn markers
without marking them special.

## Chat traits (`chat.py`)

`ChatTraits` reads a tokenizer's chat behavior once per session: `eos_ids`
(generation-config eos, tokenizer eos and added end-of-turn tokens such as
`<end_of_turn>`, `<turn|>`, `<|im_end|>`, `<|eot_id|>`), `think`
(`ThinkDelimiters` token ids for the decode loop), `takes_enable_thinking`
(substring sniff of the template), `is_base`, `logits_keep_kwarg`, a token-text
table and a render cache (128 entries, 8 MiB; key built in
`generation/decode.py`). `adopt_eos` writes `eos_ids` into `generation_config` so
HF `generate` (batch paths) stops where the decode loop does; an extracted text
sub-config can under-specify end-of-turn (Gemma-4 lists `<eos>`, turns end
`<turn|>`). Delimiter detection covers switchable templates (round-trip marker
thinking), gpt-oss channels and Mistral-3's `[THINK]` pair. These token ids are
not the scene grammar's string pair: a family can stream thinking (gpt-oss) yet
carry no committed thinking.

## Role substitution (`role_templates.py`)

Templates branch on hardcoded role names, so a custom role is dropped, raises or
works depending on family. Drowse renders with the standard roles, splices the
label into the rendered string at the family's header bytes, and tokenizes with
`add_special_tokens=False`. `SEAT_HEADERS` maps `model_type` to both seats' `RoleHeader(before, after, label)`
(`role_templates.py` is the reference): the user label is always `user`; the
assistant label is `assistant` (`model` on gemma), framed by `<|im_start|>`...`\n`
(qwen), `<start_of_turn>`...`\n` (gemma2/3), `<|turn>`...`\n` (gemma4),
`<|start_header_id|>`...`<|end_header_id|>` (llama), `<|`...`|>` (glm, talkie),
`<|start|>`...`<|channel|>` (gpt-oss, the one family whose user seat differs:
`<|message|>`); mistral families have both seats `None` (positional `[INST]`).

`resolve_seat_headers` is the only resolver (splice paths, scene autopsy, server
capability flags). It first checks `_TEMPLATE_HEADER_OVERRIDES`, keyed by
`(model_type, sha256(chat_template))` (SmolLM2-360M-Instruct is a `llama`
checkpoint whose template renders ChatML). `None` means family absent;
`SeatHeaders(None, None)` means present but label-free.

`validate_role` (`^[a-z0-9._-]+$`) is the one slug gate for every label surface
(extraction `--role`, manifold node roles, loom cast labels, scene turns);
`render_role_label` de-slugs `_` to space.

- `apply_with_role` puts one label per side on every turn (both `None` is plain
  `apply_chat_template`). The assistant splice is strict: a missing header raises
  `RoleTemplateDriftError` (500), so the extraction baseline cannot silently
  diverge from the steering one; the user splice tolerates absence. Fit capture
  renders role-baselined rows through it.
- `apply_with_per_turn_roles` honors per-message `"label"` keys by occurrence
  index, best effort (gpt-oss's generation prompt lacks `<|channel|>`, so its
  `gen_role` never lands). Both sides splice in one scan, since sequential passes
  would let a user turn labelled `assistant` (GLM `<|assistant|>`) be claimed as
  an assistant occurrence. It is the fallback when the stitcher is unavailable.

Errors: `RoleSubstitutionUnsupportedError` (400), `RoleTemplateDriftError`
(500), `InvalidRoleError` (400).

## Scene grammar (`scene.py`)

The cast model needs renders no template produces: non-alternating seats, a label
per turn, a generation prompt on either seat (gemma and mistral templates raise on
non-alternation). **Template autopsy** (`extract_turn_grammar`) renders sentinel
conversations through the live template and slices a `TurnGrammar`: `prelude`,
per-seat `SeatWrapper`s (open around the label site, close), the system shape (a
real turn, a gemma-style fold into the first turn, or unsupported as on gemma-2),
`gen_extra` (bytes the generation prompt adds after the assistant open, probed
with `enable_thinking=False` where available, plus its `True` variant), trim
flags, `last_assistant_special` (qwen3's empty-think insert into a closing
assistant turn), and the think pair with its history policy.

`render_scene` is pure stitching, `open(label) + content + close` per
`SceneTurn` plus a trailing open for `gen_seat`; the label-to-turn binding is
positional. `validate_turn_grammar` stitches canonical alternating conversations
(plain, generation prompt, padded content, closed on assistant, system, thinking
prompt, as applicable) and byte-compares each with `apply_chat_template`, so a
passing family's alternating renders are bit-identical through the stitcher. That
is why generation can render through it while fit capture uses the template or
`apply_with_role` and extract baseline still equals steer baseline, and why a
template revision drops the family to the template paths rather than corrupting
renders.

`SceneGrammarError` (autopsy failure) loses scene mode: a label-free or
unregistered family, a raising probe render, a transformed or repeated sentinel,
first-turn-special headers, a generation prompt that does not extend the plain
render, misplaced system content, an asymmetric or non-plain trim (GLM-4.7-Flash
fails). Narrower defeats drop one feature (system support, `gen_extra_thinking`,
committed thinking).

Thinking takes a literal `(open, close)` pair: `PromptRenderer.scene_grammar`
passes `("<think>", "</think>")` when the template mentions `<think>`, else `None`
(channel formats like gpt-oss have none). If the template strips history thinking,
a turn's thinking renders only while it is the last before the generation header.
Thinking without a pair raises `SceneThinkingUnsupportedError` at commit time
(`loom/ops.py::check_thinking_commit`).

Routing (`build_chat_input`): with a template and a validated grammar the stitcher
serves an optional leading system turn plus user/assistant turns. A
mid-conversation system turn, no grammar, or an assistant-seat shape the grammar
declines goes to `apply_chat_template` or `apply_with_per_turn_roles`; a
non-assistant `gen_seat` has no fallback and raises `SceneRenderError`. A
template-less tokenizer uses `render_scene_raw` (`Label: text` blocks). The
session resolves the grammar lazily once (`DrowseSession.scene_grammar`). Errors:
`SceneRenderError` (400), `SceneGrammarError` (422), `SceneThinkingUnsupportedError`
(400).

## Compile and StaticCache

`compile_with_probe` compiles (`dynamic=None`) and immediately runs probe
forwards: DynamicCache prefills at lengths 2 and 16 (forcing dynamic-shape
promotion) and, under `reduce-overhead`, a StaticCache prefill plus a decode with
`attention_mask=None` (a separate dynamo call shape where inductor bugs have
surfaced). Failure warns and returns the eager model, so breakage shows at load.
Process-global knobs: `use_static_cuda_launcher = False` (the static launcher
segfaults uncatchably on codegen bugs; Triton's raises a catchable `TypeError`)
and, on CUDA, `float32_matmul_precision("high")`, restored only on probe failure.

`is_static_cache_supported` probes a 1-token StaticCache, cached per weakly-held
module (through `_orig_mod`), device and dtype. StaticCache enables compile on
every backend, MPS included (fixed K/V shapes let inductor reuse one trace);
`is_cuda_graphs_supported` is the CUDA gate on top and picks `reduce-overhead`
over `default`. `make_static_cache` is the one factory (callers size
`max_cache_len >= prompt_len + max_new_tokens + cache_position_offset`). It
initializes layer buffers outside the compiled graph (in-graph init blocks
CUDA-graph capture and recompiles per cache) and flags sliding layers that cannot
slide this generation; `_patch_static_sliding_mask` gives those a constant
`get_mask_sizes` (byte-identical in that regime), stopping the per-token recompile
storm on hybrid caches. Whether steering may take the static path is decided in
`steering/hooks.py` (`all_fast_path` / `static_steerable`).

## Tests

`tests/model/` runs on CPU without downloads (specs against each covered family's
forward, loader ladder on mocked transformers, tiny real architectures, splice and
stitcher against Jinja fakes in `tests/support/chat_templates.py`).
`test_role_template_drift.py` checks `SEAT_HEADERS` against real tokenizers in the
local HF cache (`DROWSE_TEST_LIVE_TOKENIZERS=1` permits downloads; run before a
release). `tests/support/tiny_models.py` supplies 4-layer models with real family
templates.
