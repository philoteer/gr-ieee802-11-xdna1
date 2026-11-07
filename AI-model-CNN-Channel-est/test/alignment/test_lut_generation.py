"""Host-only checks for the combined, target-aligned trig LUT header."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_generator(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "utils" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LutGenerationTest(unittest.TestCase):
    def test_combined_header_matches_generators(self):
        atan = load_generator("gen_atan_lut").generate_section(512, 8)
        sine = load_generator("gen_sin_linear_lut_q15").generate_section(256)
        expected = (
            "#ifndef AIE_TRIG_LUTS_H\n#define AIE_TRIG_LUTS_H\n\n"
            "#include <stdint.h>\n#include <aie_api/aie.hpp>\n\n"
            + atan + "\n" + sine + "\n#endif\n"
        )
        self.assertEqual(expected, (ROOT / "kernel/common/aie_trig_luts.h").read_text())

    def test_custom_tables_keep_target_alignment(self):
        atan = load_generator("gen_atan_lut").generate_section(16, 4)
        sine = load_generator("gen_sin_linear_lut_q15").generate_section(16)
        for section, array_count in ((atan, 2), (sine, 4)):
            self.assertEqual(section.count("alignas(aie::vector_decl_align)"), array_count)
            self.assertNotIn("alignas(32)", section)

    def test_sine_bank_repetition(self):
        generator = load_generator("gen_sin_linear_lut_q15")
        coefficients = [(index, 100 + index) for index in range(16)]
        for bank_bytes in (16, 32):
            with self.subTest(bank_bytes=bank_bytes):
                values = generator.duplicate_for_four_accesses(coefficients, bank_bytes)
                entries = bank_bytes // 4
                expected = []
                for base in range(0, len(coefficients), entries):
                    bank = [value for pair in coefficients[base:base + entries]
                            for value in pair]
                    expected.extend(bank * 2)
                self.assertEqual(values, expected)
                self.assertEqual(len(values), 4 * len(coefficients))

    def test_sine_rejects_incomplete_or_invalid_banks(self):
        generator = load_generator("gen_sin_linear_lut_q15")
        for count, bank_bytes in ((4, 32), (7, 16), (8, 24)):
            with self.subTest(count=count, bank_bytes=bank_bytes):
                with self.assertRaises(ValueError):
                    generator.duplicate_for_four_accesses([(0, 0)] * count, bank_bytes)


if __name__ == "__main__":
    unittest.main()
