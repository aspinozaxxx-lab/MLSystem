from __future__ import annotations

import unittest

from mlsystem.src.pipeline.stages.registry import get_stage_entrypoint


class StageRegistryTests(unittest.TestCase):
    def test_known_stage_is_found(self) -> None:
        entrypoint = get_stage_entrypoint("inventory_scenes")
        self.assertTrue(callable(entrypoint))

    def test_unknown_stage_error_is_clear(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown MLSystem stage entrypoint"):
            get_stage_entrypoint("unknown_stage")


if __name__ == "__main__":
    unittest.main()
