"""Contract tests for the optional Transformer Engine import.

Transformer Engine 1.11 reads flash-attn package metadata at import time, so
flash-attn-free installs cannot import it at all. The vision tower only needs
``te.RMSNorm``; when Transformer Engine is unavailable, ``qwen2_5_vit`` falls
back to a native RMSNorm (``te is None``). These tests pin the contract that
``Qwen2RMSNorm`` always works, whichever backend is active. They need neither
GPU nor checkpoint files.
"""

import unittest

import torch

import qwen2_5_vit


class TeOptionalImportTest(unittest.TestCase):
    def test_qwen2rmsnorm_forward_works_with_active_backend(self):
        norm = qwen2_5_vit.Qwen2RMSNorm(32, eps=1e-6)
        hidden = torch.randn(3, 32) * 4
        if qwen2_5_vit.te is not None:
            # The Transformer Engine RMSNorm kernel only runs on CUDA.
            if not torch.cuda.is_available():
                self.skipTest("Transformer Engine backend needs CUDA")
            norm, hidden = norm.cuda(), hidden.cuda()
        output = norm(hidden)
        self.assertEqual(output.shape, hidden.shape)
        # RMSNorm output must carry the learned weight and unit RMS.
        with torch.no_grad():
            expected_rms = norm.weight.float().pow(2).mean().sqrt()
        self.assertTrue(
            torch.allclose(output.float().pow(2).mean(-1).sqrt(), expected_rms.expand(3), atol=1e-3)
        )

    def test_backend_selection_is_explicit(self):
        if qwen2_5_vit.te is None:
            self.assertTrue(issubclass(qwen2_5_vit._RMSNormImpl, torch.nn.Module))
            self.assertIs(qwen2_5_vit._RMSNormImpl.__module__, qwen2_5_vit.__name__)
        else:
            self.assertIs(qwen2_5_vit._RMSNormImpl, qwen2_5_vit.te.RMSNorm)

    def test_fallback_state_dict_matches_te_layout(self):
        # The fallback must load the same checkpoint entries as te.RMSNorm:
        # a single `weight` parameter and no buffers.
        if qwen2_5_vit.te is None:
            norm = qwen2_5_vit.Qwen2RMSNorm(32)
            self.assertEqual(list(norm.state_dict()), ["weight"])


if __name__ == "__main__":
    unittest.main()
