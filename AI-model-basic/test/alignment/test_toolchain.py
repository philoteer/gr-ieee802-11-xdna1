"""Build command regression checks; these do not launch any AIE compiler."""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD_DIRS = ["kernel", *(str(path.parent.relative_to(ROOT))
                         for path in sorted((ROOT / "test").rglob("Makefile")))]


def dry_run(directory, *options):
    return subprocess.check_output(
        ["make", "--no-print-directory", "-B", "-n", "-C", str(ROOT / directory),
         "JOBS=1", *options, "all"], text=True,
    )


class ToolchainTest(unittest.TestCase):
    def test_device_and_compiler_selection(self):
        for directory in BUILD_DIRS:
            for device, architecture in (("npu", "aie2"), ("npu2", "aie2p")):
                with self.subTest(directory=directory, device=device):
                    output = dry_run(directory, f"devicename={device}", "AIECC=aiecc")
                    self.assertIn(f"xchesscc_wrapper {architecture} -aietools ", output)
                    self.assertIn(f"aie2_memtile_caching.py {device}", output)
                    self.assertIn("aiecc -j1 --aietools=", output)
                    self.assertIn("--get-xclbin --get-npu-insts --xchesscc", output)
                    self.assertNotIn("--aie-generate-xclbin", output)

    def test_legacy_driver_flags(self):
        for directory in BUILD_DIRS:
            with self.subTest(directory=directory):
                output = dry_run(directory, "NPU2=1", "AIECC=/legacy/bin/aiecc.py")
                self.assertIn("xchesscc_wrapper aie2p -aietools ", output)
                self.assertIn("aiecc.py -j1 --aie-generate-xclbin --no-compile-host", output)
                self.assertIn("--aie-generate-npu-insts", output)
                self.assertNotIn("--get-xclbin", output)

    def test_aie2p_preserves_model_selection(self):
        for directory in ("kernel", "test/frame_equalizer"):
            output = dry_run(directory, "NPU2=1", "AIECC=aiecc", "ENGINE=AI",
                             "MODEL=MLP_16", "EQUALIZER_ALGORITHM=LMS")
            self.assertIn("-DFRAME_EQUALIZER_ENGINE=FRAME_EQUALIZER_ENGINE_AI", output)
            self.assertIn("-DFRAME_EQUALIZER_MODEL=FRAME_EQUALIZER_MODEL_MLP_16", output)
            self.assertIn("-DFRAME_EQUALIZER_ALGORITHM=FRAME_EQUALIZER_ALGORITHM_LMS", output)
            self.assertIn(f"-I {ROOT}/kernel/common", output)

    def test_gnuradio_runtime_interpreter_is_selectable(self):
        output = subprocess.check_output(
            ["make", "--no-print-directory", "-B", "-n", "-C", str(ROOT / "test/frame_equalizer"),
             "JOBS=1", "NPU2=1", "AIECC=aiecc", "GR_PYTHON=/runtime/python3",
             "learned-ber"], text=True,
        )
        self.assertIn("/runtime/python3 build/test_learned_ber_grc.py", output)
        # Graph generation keeps the build Python, even with a different runtime.
        self.assertIn(f"python3 {ROOT}/test/frame_equalizer/aie2_memtile_caching.py", output)


if __name__ == "__main__":
    unittest.main()
