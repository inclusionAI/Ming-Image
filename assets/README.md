# Assets

This directory contains the examples and prompt files used by the project
README and runnable demos.

## Contents

- [`ming-image-design-ui-ux-leaderboard.webp`](./ming-image-design-ui-ux-leaderboard.webp)
  is the UI/UX Design leaderboard graphic shown in the main README.
- [`model_cards/`](./model_cards/) contains the image-only showcases used on
  the Design and Layer model pages.
- [`t2i_samples/`](./t2i_samples/) contains text-to-image examples. Its
  `transparent_rgba/` subdirectory contains examples with an alpha channel.
- [`layer_samples/`](./layer_samples/) contains the input, layer specification,
  and decomposition preview used by the layer-decomposition demo.
- [`t2i_four_seasons_cabin_prompt.json`](./t2i_four_seasons_cabin_prompt.json)
  is the structured prompt used by the text-to-image demo and smoke test.
- [`t2i_rewriter_system_prompt.txt`](./t2i_rewriter_system_prompt.txt) is the
  system prompt for optional text-to-image prompt rewriting.
- [`layer_decompose_5layers.txt`](./layer_decompose_5layers.txt) is a compact
  layer-decomposition prompt example.

Test fixtures used by the smoke suite live under [`tests/assets/`](../tests/assets/).
