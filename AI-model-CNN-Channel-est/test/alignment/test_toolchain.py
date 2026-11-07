"""Build command checks that never launch an AIE compiler."""

from pathlib import Path
import subprocess
import tempfile
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
        for directory in ("kernel", "test/frame_equalizer",
                          "test/frame_equalizer_preproc/channel_est"):
            output = dry_run(directory, "NPU2=1", "AIECC=aiecc", "ENGINE=AI",
                             "MODEL=MLP_16", "EQUALIZER_ALGORITHM=LS", "CNN_DATA_UPDATER=MLP")
            for define in ("ENGINE_AI", "MODEL_MLP_16", "ALGORITHM_LS",
                           "CNN_DATA_UPDATER_MLP"):
                self.assertIn(f"=FRAME_EQUALIZER_{define}", output)
            self.assertIn(f"-I {ROOT}/kernel/common", output)

    def test_aie2p_preserves_weight_and_output_selection(self):
        output = dry_run("kernel", "NPU2=1", "AIECC=aiecc",
                         "EQUALIZER_ALGORITHM=CNN", "CNN_DATA_UPDATER=MLP",
                         "CNN_WEIGHTS=BUS_TO_ROOM", "DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1")
        self.assertIn("-DFRAME_EQUALIZER_ALGORITHM=FRAME_EQUALIZER_ALGORITHM_CNN", output)
        self.assertIn("-DFRAME_EQUALIZER_CNN_DATA_UPDATER=FRAME_EQUALIZER_CNN_DATA_UPDATER_MLP", output)
        self.assertIn("-DFRAME_EQUALIZER_CNN_WEIGHTS=FRAME_EQUALIZER_CNN_WEIGHTS_BUS_TO_ROOM", output)
        self.assertIn("-DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1", output)

    def test_build_marker_tracks_effective_device(self):
        # Only run the configuration recipe, never graph or object compilation.
        with tempfile.TemporaryDirectory(prefix="cnn-build-mode-") as directory:
            marker = Path(directory) / "build/.frame_equalizer_build_mode"
            command = ["make", "--no-print-directory", "-C", directory, "-f",
                       str(ROOT / "kernel/Makefile"), "check-build-mode", "NPU2=0"]
            subprocess.run([*command, "devicename=npu"], check=True, capture_output=True)
            self.assertIn("DEVICE=npu\n", marker.read_text())
            subprocess.run([*command, "devicename=npu2"], check=True, capture_output=True)
            self.assertIn("DEVICE=npu2\n", marker.read_text())
            unchanged_time = marker.stat().st_mtime_ns
            subprocess.run([*command, "devicename=npu2"], check=True, capture_output=True)
            self.assertEqual(marker.stat().st_mtime_ns, unchanged_time)


if __name__ == "__main__":
    unittest.main()
