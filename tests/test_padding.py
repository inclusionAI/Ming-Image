import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from diffusion.padding import mask_out_alignment_padding


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ZeroPaddingTest(unittest.TestCase):
    def test_masks_alignment_padding_at_per_item_offsets(self):
        attention_mask = torch.ones((2, 8), dtype=torch.bool)
        pad_masks = [
            torch.tensor([False, False, True, True]),
            torch.tensor([False, True, False]),
        ]

        result = mask_out_alignment_padding(attention_mask, pad_masks, [0, 3])

        self.assertEqual(
            result[0].tolist(),
            [True, True, False, False, True, True, True, True],
        )
        self.assertEqual(
            result[1].tolist(),
            [True, True, True, True, False, True, True, True],
        )

    def test_rejects_non_boolean_attention_mask(self):
        attention_mask = torch.ones((1, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "2D boolean"):
            mask_out_alignment_padding(
                attention_mask, [torch.zeros(4, dtype=torch.bool)], [0]
            )


if __name__ == "__main__":
    unittest.main()
