# Public assets

This directory contains public inference inputs, prompt assets, and the source
images used by the two Hugging Face model cards. Test-only fixtures live under
`tests/assets/` and are not part of the public showcase.

## Directory layout

```text
assets/
├── ming-image-design-ui-ux-leaderboard.webp
├── model_cards/
│   ├── design_showcase.webp
│   ├── design_transparency_showcase.webp
│   └── layer_showcase.webp
├── t2i_samples/
│   ├── 01_fashion_aurel.jpg
│   ├── ...
│   ├── 15_child_plush_tiger.jpg
│   └── transparent_rgba/
│       ├── 01_silver_sedan.webp
│       ├── 02_tabby_cat.webp
│       └── 03_snowboarder_portrait.webp
├── layer_samples/
│   ├── card_making_input.png
│   ├── card_making_decomposition.png
│   └── card_making_prompt.txt
├── layer_decompose_5layers.txt
├── t2i_four_seasons_cabin_prompt.json
└── t2i_rewriter_system_prompt.txt
```

The top-level T2I sample naming convention is
`<two-digit order>_<case_id>.<ext>`. The order is the canonical display order
and the suffix is a stable descriptive case identifier.

All 15 top-level T2I samples are 2048x2048 RGB JPEGs encoded at quality 75
with 4:2:0 chroma subsampling.

The files under `transparent_rgba/` use the same ordered English naming style.
They are 2048x2048 WebP images encoded at color quality 90 and alpha quality
100. They retain the original 8-bit alpha channel and do not contain a
checkerboard background. The checkerboard is added only to the model-card
contact sheet; alpha values from 0 through 15 are treated as fully transparent
in that preview to suppress near-transparent background noise.

The Layer sample uses one case prefix for its input, prompt, and decomposition
strip. `card_making_decomposition.png` is the supplied source strip; it is not
an individual RGBA layer file.

## Repository README and model-card asset

`ming-image-design-ui-ux-leaderboard.webp` is the leaderboard image displayed
in the repository README and copied byte-for-byte to the Design model
repository as `assets/uiux_leaderboard.webp`. It is a 2160x1978 lossless WebP
conversion of the supplied PNG; its decoded RGBA pixels and dimensions are
unchanged. It is not used by the Layer model card.
SHA-256: `2a7260a832cdbea85bf5284c9df2f74502c486bb8166be24f8a9247c9297884f`.

## Model-card assets

The files below are ready to copy without rebuilding or re-encoding:

| Target model repository | Source file | Destination | Dimensions | SHA-256 |
| --- | --- | --- | --- | --- |
| `Ming-Image-0.1-Design` | `assets/model_cards/design_showcase.webp` | `assets/showcase.webp` | 2384x3960 | `9c08b25583d74295a2d34d519abd45544927b65d935581e40f0fcf1941c1da24` |
| `Ming-Image-0.1-Design` | `assets/model_cards/design_transparency_showcase.webp` | `assets/transparency_showcase.webp` | 2384x808 | `2b2a18df09e051ca6c95bada87c99edac2ae917ab9da54d7ed32b328707f55c8` |
| `Ming-Image-0.1-Design` | `assets/ming-image-design-ui-ux-leaderboard.webp` | `assets/uiux_leaderboard.webp` | 2160x1978 | `2a7260a832cdbea85bf5284c9df2f74502c486bb8166be24f8a9247c9297884f` |
| `Ming-Image-0.1-Design-Layer` | `assets/model_cards/layer_showcase.webp` | `assets/showcase.webp` | 4096x552 | `a5e243217323e4233e6c7fd427814794739e0d3de44b0dbdcb1c55ca00399c1b` |

The Design sheet contains all 15 T2I samples with no added labels or prompts.
The Design transparency sheet contains three generated RGBA subjects on a
preview-only checkerboard with no added labels or prompts.
The Layer sheet contains the input, six checkerboard-backed layer panels, and
the recomposed result; it does not contain the decomposition prompt.
