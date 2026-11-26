# Assets

This directory contains the examples and prompt files used by the project
README and runnable demos.

## Contents

- [`ming-image-0.2-design-aa-t2i-leaderboard.webp`](./ming-image-0.2-design-aa-t2i-leaderboard.webp)
  is the AA-Image-T2I v2.1 overall open-weights leaderboard for 0.2 Design.
- [`ming-image-0.2-design-aa-t2i-leaderboard-preview.webp`](./ming-image-0.2-design-aa-t2i-leaderboard-preview.webp)
  is the preview used in the README, where it links to the full-resolution figure.
- [`ming-image-design-ui-ux-leaderboard.webp`](./ming-image-design-ui-ux-leaderboard.webp)
  is the historical 0.1 Design UI/UX leaderboard, displayed alongside the 0.2
  figure in the main README's image table.
- [`model_cards/`](./model_cards/) contains the image-only showcases used on
  the Design and Layer model pages, including the Layer gallery and
  performance result.
- [`model_cards/ming-image-0.2-design-ecommerce.webp`](./model_cards/ming-image-0.2-design-ecommerce.webp),
  [`model_cards/ming-image-0.2-design-web-design.webp`](./model_cards/ming-image-0.2-design-web-design.webp),
  and [`model_cards/ming-image-0.2-design-spatial-design.webp`](./model_cards/ming-image-0.2-design-spatial-design.webp)
  are animated slideshows of e-commerce, web, and spatial design examples
  in the 0.2 Design gallery.
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
