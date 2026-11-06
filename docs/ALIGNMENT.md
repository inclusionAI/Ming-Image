# Aligning tinyv3 results on third-party hardware

This branch ships a fixed smoke case and portable comparison tools so a
hardware-adaptation team can localise numerical drift without guessing.

## What is fixed

- Prompt bytes: `examples/smoke/t2i_fox_prompt.txt` (UTF-8, do not retype).
- Case parameters: 1024×1024, 30 steps, CFG 1.0, seed 42, bf16,
  `--device-map balanced` (single GPU). tinyv3 numerics depend on the
  loading path: balanced placement keeps the precision-sensitive
  `t_embedder`/`cap_embedder` in fp32 under bf16 autocast; a plain `.to(dtype)`
  cast makes the transformer all-bf16
  and shifts outputs. Always use the balanced placement for alignment.
- Reference outputs: `examples/smoke/reference/` (PNG + sha256 + pixel
  hash in `smoke_manifest.json`).

## Capture a comparable bundle on your hardware

```bash
python tools/capture_case.py \
  --model <path-to-complete-tinyv3-package> \
  --task text-to-image \
  --prompt examples/smoke/t2i_fox_prompt.txt \
  --width 1024 --height 1024 --steps 30 --seed 42 \
  --output-dir alignment/my-run
```

The bundle contains token ids/masks, both condition tensors, the initial
noise (pre-scheduler Gaussian, fp32), per-step latents, and the final
pixels, plus full identity metadata (config hashes, code SHA, environment).

The capture CLI draws noise with the target device's seeded generator; it
does not yet accept a saved-noise input. A different device RNG can therefore
fail the exact-noise check before any model comparison is meaningful. For a
hardware port, load `initial_noise` from the reference `arrays.npz` and pass
the resulting FP32 tensor through the model's `image_gen_latents` argument.
It is pre-scheduler noise; do not apply scheduler scaling twice. A capture
CLI replay option remains follow-up work.

## Compare against the reference bundle

```bash
python tools/compare_case.py \
  --reference examples/smoke/reference \
  --candidate alignment/my-run \
  --report alignment/my-report.json
```

Comparison order: identity (family, case parameters, prompt hash) ->
discrete inputs (exact) -> conditions -> initial noise (exact replay) ->
final-step latent -> final pixels. Discrete inputs, shapes and the noise
must match exactly; floating-point segments report max/mean abs, relative
L2 and cosine; the image reports pixel MAE/RMSE. The exit code covers the
available checked segments. Public references omit conditions and per-step
latents: these are marked `absent`, excluded from the verdict, and remain
unverified by a public smoke pass. When both bundles include per-step latents,
`step_latents_all` is reported as `needs_review`: per-step cross-backend
thresholds require joint measurement. Inspect those statuses separately from
the overall verdict before accepting a hardware port.

## Measured basis for the default thresholds (NVIDIA L20Y, 2026-11-06)

Same hardware, same code, repeated capture: every segment bitwise
identical. Against a separate reference pipeline (same
weights, same seed): final-image pixel MAE ≈ 1.2–2.0 at 30 steps (bf16
accumulation differences between the two pipeline implementations). These
numbers are a same-stack reference, not a cross-hardware guarantee.
Thresholds live at the top of `tools/compare_case.py`; override with
`--thresholds` after joint measurement on the target hardware.

For image editing, reference latents are bilinearly resampled onto the output
grid when necessary. This differs from implementations that patchify each
reference at its own resolution. The bundled edit output represents this
runtime and does not establish pixel equivalence with those implementations.

## Operator requirements third parties must port

- KDA linear-attention kernels (`fla-core`; no eager fallback — do not
  substitute plain attention and claim equivalence).
- Vision attention uses `flash-attn` ≥ 2.6 semantics (`softcap` argument).
- 3D M-RoPE positions, Qwen3-VL patch layout, RMSNorm/gating numerics.
- bf16 autocast around the sampling loop with fp32 t/cap embedders.
