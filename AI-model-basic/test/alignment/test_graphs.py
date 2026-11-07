"""Generate every graph for both devices without compiling or using hardware."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec("aie"), "requires MLIR-AIE Python bindings")
class GraphTest(unittest.TestCase):
    def test_explicit_target_and_runtime(self):
        environment = os.environ.copy()
        # Keep external batch/FIFO tuning from changing the regression graph.
        for name in ("AIE_TILE_SIZE", "AIE_TILES_PER_BATCH", "AIE_FIFO_DEPTH", "AIE_FIFO_DEPTHS"):
            environment.pop(name, None)
        for graph in sorted(ROOT.rglob("aie2_memtile_caching.py")):
            for argument, device in (("npu", "npu1"), ("npu2", "npu2")):
                with self.subTest(graph=graph.relative_to(ROOT), device=device):
                    result = subprocess.run(
                        [sys.executable, str(graph), argument], env=environment,
                        capture_output=True, text=True, check=True, timeout=30,
                    )
                    self.assertIn(f"aie.device({device})", result.stdout)
                    self.assertIn("aie.core(", result.stdout)
                    self.assertIn("aie.runtime_sequence", result.stdout)
                    if graph.parent == ROOT / "kernel":
                        self.assertEqual(result.stdout.count("aie.core("), 8)
                        self.assertIn("stack_size = 4096", result.stdout)
                        self.assertIn("center_frequency", result.stdout)


if __name__ == "__main__":
    unittest.main()
