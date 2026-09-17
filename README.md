# Ming Image 0.1 Design

Public Hugging Face inference code for two checkpoint families:

- text-to-image generation;
- layer decomposition with a requested number of output layers.

Both checkpoint families use the same inference entry point, but their runtime
semantics are declared by the checkpoint instead of being guessed at runtime.

## Models

- Text-to-image: [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)
- Layer decomposition: [inclusionAI/Ming-Image-0.1-Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer)

## Requirements

- Python 3.10 or newer;
- CUDA-capable PyTorch for full model inference;
- a local checkpoint directory or a Hugging Face Hub repository ID.

Install the runtime dependencies in a clean environment:

```bash
pip install -r requirements.txt
```

`flash_attention_2` is optional. The CLI defaults to `eager` attention because
the LLM only implements eager and FlashAttention 2 attention classes; the
diffusion transformer always uses PyTorch SDPA internally. Selecting
`--attn-implementation sdpa` fails closed at load time.

## Checkpoint contract

Every checkpoint must contain `inference_profile.json` in its root. Missing or
unknown fields are errors; the loader never infers padding behavior from a
directory name or silently falls back to another mode.

Text-to-image checkpoint:

```json
{
  "schema_version": 1,
  "inference_profile": "generation_edit",
  "alignment_padding_mode": "zero_masked",
  "multi_frame_output": false,
  "vae_input_channels": 4,
  "vae_sample_mode": "argmax"
}
```

Layer-decomposition checkpoint:

```json
{
  "schema_version": 1,
  "inference_profile": "layer_decompose",
  "alignment_padding_mode": "learned",
  "multi_frame_output": true,
  "vae_input_channels": 4,
  "vae_sample_mode": "argmax"
}
```

Copy the appropriate template from `examples/profiles/` into the checkpoint
root. Loading fails when the checkpoint does not match its declared profile.

## Inference

The `--model` argument accepts either a local directory or an HF Hub repo ID.
Hub models are resolved to one immutable local snapshot before any component is
loaded. Use `--revision` to pin a branch, tag or commit. The examples use the
published Hub IDs; replace them with local checkpoint directories for offline
inference.

Sampling defaults are profile-specific:

| Checkpoint family | Steps | CFG |
| --- | ---: | ---: |
| Text-to-image | 12 | 1.0 |
| Layer decomposition | 12 | 2.0 |

Pass `--steps` or `--cfg` to override either value explicitly.

The default and minimum validated deployment is **one GPU with at least 80 GiB
of memory**, running in BF16. GPU 0 hosts the image-generation modules and the
MLLM decoder, and this configuration passes both model families end-to-end.

```bash
export CUDA_VISIBLE_DEVICES=0
```

The CLI defaults to `--device-map balanced --num-gpus 1 --device cuda:0` and
BF16, so the examples below use one visible GPU. Select
`--attn-implementation flash_attention_2` when FlashAttention 2 is installed;
the portable CLI default remains `eager`.

Multi-GPU inference is optional. Expose the desired devices and pass
`--num-gpus N` to shard the MLLM decoder; the image-generation modules remain
on logical `cuda:0`. For example, the established 8-GPU mapping for a 32-layer
MLLM is `(1, 4, 4, 4, 4, 5, 5, 5)`.

Text-to-image:

```bash
python infer.py \
  --model inclusionAI/Ming-Image-0.1-Design \
  --task text-to-image \
  --prompt 'Create a square retro-futurist poster for a speculative ocean research festival. Build the poster as a simple table with five rows and three columns, using straight thin copper dividers and equal inner padding. Row 1, columns 1 through 3: center the headline "BEYOND THE BLUE" on a cream background. Row 2, columns 1 through 3: center the subtitle "OCEAN FUTURES" on a deep navy background. Row 3, column 1: place the event text "SEPTEMBER 18 IN SHANGHAI". Row 3, column 2: place one glass orbital greenhouse above calm midnight-blue water. Row 3, column 3: place the program text "LIVE LABS". Row 4, column 1: place one red research vessel in side view. Row 4, column 2: place exactly two translucent jellyfish against dark water. Row 4, column 3: place one coral specimen in front of a violet sunrise. Row 5, columns 1 through 3: center the call to action "RESERVE YOUR PASS". Keep every object fully inside its assigned cell. Do not overlap cells, repeat objects, add floating panels, or add decorative text. Render exactly the five quoted text strings once each and fully legible. The quotation marks only delimit the required copy and must not appear in the poster. Do not add any other letters, numbers, logos, or placeholder copy. Use deep navy, cyan, coral, cream, and metallic copper, with sharp vector edges and controlled halftone texture.' \
  --attn-implementation flash_attention_2 \
  --output-dir outputs/t2i
```

Layer decomposition:

```bash
python infer.py \
  --model inclusionAI/Ming-Image-0.1-Design-Layer \
  --task layer-decompose \
  --input-image composite.png \
  --prompt assets/layer_decompose_5layers.txt \
  --attn-implementation flash_attention_2 \
  --output-dir outputs/layers
```

`--prompt` accepts either raw prompt text or a path to a prompt file
(`assets/layer_decompose_5layers.txt` ships with the repository). The layer
count is parsed from the prompt — a "Decompose this image into N layers" or
"Number of layers: N" specification — and drives how many layer images are
produced. When `--prompt` is omitted, a default prompt is generated from
`--num-layers` (e.g. `Decompose this image into 5 layers.`), so `--num-layers`
only takes effect on that fallback path.

### Prompt enhancement (rewrite) for layer decomposition

Layer decomposition is driven by an explicit per-layer specification, not a
free-form caption. The reference pipeline first runs a prompt enhancer: an
instruction-following VLM rewrites the user's rough layer plan into a precise
decomposition — concrete colors/positions/shapes per layer, real text quoted
verbatim on its own front layer, a text-supporting card/panel/badge/banner as
its own layer directly behind the text, the main subject as its own layer, and
the background last absorbing the remaining supports and surfaces.

This rewrite is a pre-processing step *outside* `infer.py`: run the enhancer
first, then pass its output to `--prompt` (as raw text or via an asset file).
The enhancer's output is exactly the format this CLI parses — it regenerates the
"Decompose this image into N layers" / "Number of layers: N" spec, so the layer
count flows through the same `--prompt` parsing path described above.

Any instruction-following VLM can serve as the rewrite model; the reference
setup uses `qwen3.8-27B` for that role. The guided prompt below is part of the
released pipeline and is kept here for reproducibility:

```text
GUIDED_PROMPT = """You are a graphic-design layer-decomposition expert. You are given ONE flattened design image and a ROUGH layer plan from the user. Rewrite the rough plan into a precise layer decomposition that matches the image.

User's rough layer plan:
{spec}

Guidelines:
- Follow the user's plan EXACTLY: use the same number of layers and the same per-layer role/meaning, in the same order. Layer 1 is the FRONT-most (topmost); the last layer is the background/environment. Stacking the layers back-to-front must reproduce the image.
- For each layer, write a concrete one-or-two-sentence description grounded in the image: real colors, positions and shapes.
- TEXT goes in the front layer(s); quote any real text VERBATIM in double quotes and keep its original language (do not translate).
- A text-supporting CARD / PANEL / BADGE / BANNER is its OWN layer directly behind the text — do not merge it into the text layer or into the background.
- The MAIN SUBJECT (hero product/photo/illustration) is its own layer.
- The BACKGROUND/ENVIRONMENT is the LAST layer and ABSORBS supporting props and surfaces under the subject (tables, boards, plates, floors, shadows, gradients, patterns) — these are NOT separate layers.

Write the description DIRECTLY about the content; do NOT mention "image" or narrate your reasoning. Output EXACTLY in this format and NOTHING else (N = the number of layers in the user's plan):

Decompose this image into N layers with the following specifications:

Number of layers: N
Layer 1: <front-most layer>
Layer 2: <...>
Layer N: <background/environment layer>"""
```

### Prompt rewriting (enhancement) for text-to-image

Text-to-image uses the same rewrite pipeline as layer decomposition: an
instruction-following VLM turns the user's short caption into a precise,
layout-structured JSON description — Figma-style layers ordered back to
front, with exact coordinates, hierarchy, color specs, and every rendered
string quoted verbatim and owned exactly once. Unlike the layer-decomposition
rewriter, which builds an explicit per-layer decomposition, the
text-to-image rewriter builds a full-page layout description over the whole
1:1 canvas.

The rewrite is a pre-processing step *outside* `infer.py`, exactly like the
layer-decomposition enhancer above: run the enhancer first, then pass its
output to `--prompt` (as raw text or via an asset file). The same
instruction-following VLM can serve both roles; the reference setup uses
`qwen3.8-27B`.

The system prompt for this rewriter is kept verbatim in
`assets/t2i_rewriter_system_prompt.txt` (the reference release includes it
for reproducibility; the two rewriters therefore form one consistent prompt
system: layer decomposition plans layers, text-to-image plans the full page,
and both enforce the same quoting and layout rules).

Use `--validate-only` to check the checkpoint contract and task combination
without loading model weights:

```bash
python infer.py --model inclusionAI/Ming-Image-0.1-Design --task text-to-image \
  --prompt "A red circle on a white background" --validate-only
```

Layer decomposition returns the requested layers plus one leading
composite/full-canvas image; the CLI skips that first image and saves the
standalone layers as `layer_01.png`, `layer_02.png`, and so on.

## Verification

Fast contract tests do not require model weights:

```bash
python -m unittest -v tests.test_inference_profile
python -m unittest -v tests.test_padding
```

Full inference is a GPU smoke test and requires both checkpoint families. At a
minimum, validate text-to-image and two- and five-layer decomposition with
fixed seeds. Include a non-square resolution so alignment padding is exercised.
The smoke test defaults to the single-GPU placement and FlashAttention 2
configuration shown above:

```bash
export CUDA_VISIBLE_DEVICES=0
export MING_GENERATION_MODEL=inclusionAI/Ming-Image-0.1-Design
export MING_LAYER_MODEL=inclusionAI/Ming-Image-0.1-Design-Layer
export MING_TEST_IMAGE="$PWD/tests/assets/smoke_input.png"
export MING_SMOKE_OUTPUT_DIR=/path/to/persistent/results
python -m unittest -v tests.test_inference_smoke.InferenceSmokeTest.test_two_step_probes
```
