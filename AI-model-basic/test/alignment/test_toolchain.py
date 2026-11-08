"""Build command regression checks; these do not launch any AIE compiler."""

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
                    for line in output.splitlines():
                        if "xchesscc_wrapper" in line:
                            self.assertRegex(line, r" -c \S+\.cc -o ")

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

    def test_build_configuration_tracks_changes(self):
        for directory in ("kernel", "test/frame_equalizer"):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as temporary:
                marker = Path(temporary) / "config"
                command = ["make", "--no-print-directory", "-C", str(ROOT / directory),
                           "check-build-mode", f"BUILD_CONFIG={marker}", "NPU2=0"]
                options = ["devicename=npu", "EQUALIZER_ALGORITHM=LS", "ENGINE=AI",
                           "MODEL=MLP_16X16", "AIE_TILES_PER_BATCH=16"]
                subprocess.run([*command, *options], check=True, capture_output=True)
                unchanged_time = marker.stat().st_mtime_ns
                subprocess.run([*command, *options], check=True, capture_output=True)
                self.assertEqual(marker.stat().st_mtime_ns, unchanged_time)
                for index, option in enumerate(("devicename=npu2", "EQUALIZER_ALGORITHM=CNN",
                                                "ENGINE=CLASSIC", "MODEL=MLP_16",
                                                "AIE_TILES_PER_BATCH=8")):
                    previous = marker.read_text()
                    options[index] = option
                    subprocess.run([*command, *options], check=True, capture_output=True)
                    self.assertNotEqual(marker.read_text(), previous)
                self.assertIn("DEVICE=npu2\n", marker.read_text())
                if directory == "test/frame_equalizer":
                    previous = marker.read_text()
                    subprocess.run([*command, *options, "DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1"],
                                   check=True, capture_output=True)
                    self.assertNotEqual(marker.read_text(), previous)
                    self.assertIn("NPU2=1\n", marker.read_text())

    def test_incremental_build_invalidates_objects(self):
        # Fake already-built artifacts in a private directory. Only configuration
        # recipes execute; object and graph rebuilds are inspected with make -n.
        for directory in ("kernel", "test/frame_equalizer"):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as temporary:
                build = Path(temporary) / "build"
                build.mkdir()
                command = ["make", "--no-print-directory", "-C", temporary,
                           "-f", str(ROOT / directory / "Makefile")]
                subprocess.run([*command, "check-build-mode"], check=True, capture_output=True)
                objects = ("autocorr_cf", "magsq_avg_cf", "threshold_cf", "sync_short_cf",
                           "sync_long_cf", "fft_cf", "frame_equalizer_preprocess",
                           "frame_equalizer_equalize") if directory == "kernel" else (
                               "frame_equalizer_preprocess", "frame_equalizer_equalize")
                for name in ("aie.mlir", "final.xclbin", "insts.bin",
                             "frame_equalizer_build_mode.txt", *(f"{name}.o" for name in objects)):
                    (build / name).touch()
                # The real configuration recipe has already run. In dry-run
                # mode make otherwise assumes its FORCE prerequisite changed
                # the stamp, since it cannot execute the content comparison.
                inspect = [*command, "-n", "-o", "FORCE", "all"]
                output = subprocess.check_output(inspect, text=True)
                self.assertNotIn("xchesscc_wrapper", output)
                subprocess.run([*command, "check-build-mode", "MODEL=MLP_16"],
                               check=True, capture_output=True)
                output = subprocess.check_output([*inspect, "MODEL=MLP_16"], text=True)
                self.assertIn("xchesscc_wrapper", output)
                self.assertIn("frame_equalizer_equalize.cc", output)


if __name__ == "__main__":
    unittest.main()
