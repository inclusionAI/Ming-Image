"""Contract tests for CLI --attn-implementation propagation.

The checkpoint config.json never pins `_attn_implementation`, so the nested
vision/LLM configs fall back to their `flash_attention_2` code defaults. The
CLI value reaches `BailingMM2Config` only after construction (transformers
`from_pretrained` assigns it through the config's `_attn_implementation`
setter); the top-level config must forward explicit values to both nested
configs without letting the construction-time `None` erase their defaults.

These tests need neither GPU nor checkpoint files.
"""

import unittest

from configuration_bailingmm2 import BailingMM2Config


class AttnImplementationPropagationTest(unittest.TestCase):
    def _config(self):
        return BailingMM2Config(llm_config={}, vision_config={})

    def test_nested_defaults_survive_construction(self):
        config = self._config()
        self.assertIsNone(config._attn_implementation)
        self.assertEqual(config.vision_config._attn_implementation, "flash_attention_2")
        self.assertEqual(config.llm_config._attn_implementation, "flash_attention_2")

    def test_explicit_value_reaches_nested_configs(self):
        for value in ("eager", "flash_attention_2"):
            with self.subTest(value=value):
                config = self._config()
                # This assignment mirrors transformers from_pretrained.
                config._attn_implementation = value
                self.assertEqual(config._attn_implementation, value)
                self.assertEqual(config.vision_config._attn_implementation, value)
                self.assertEqual(config.llm_config._attn_implementation, value)

    def test_construction_kwarg_reaches_nested_configs(self):
        config = BailingMM2Config(
            attn_implementation="eager", llm_config={}, vision_config={}
        )
        self.assertEqual(config.vision_config._attn_implementation, "eager")
        self.assertEqual(config.llm_config._attn_implementation, "eager")

    def test_none_does_not_erase_nested_configs(self):
        config = self._config()
        config._attn_implementation = "eager"
        config._attn_implementation = None
        self.assertIsNone(config._attn_implementation)
        self.assertEqual(config.vision_config._attn_implementation, "eager")
        self.assertEqual(config.llm_config._attn_implementation, "eager")

    def test_serialization_round_trip_keeps_contract(self):
        # The checkpoint load path rebuilds the config from plain dicts.
        config = BailingMM2Config(**self._config().to_dict())
        self.assertEqual(config.vision_config._attn_implementation, "flash_attention_2")
        self.assertEqual(config.llm_config._attn_implementation, "flash_attention_2")
        config._attn_implementation = "eager"
        self.assertEqual(config.vision_config._attn_implementation, "eager")
        self.assertEqual(config.llm_config._attn_implementation, "eager")


if __name__ == "__main__":
    unittest.main()
