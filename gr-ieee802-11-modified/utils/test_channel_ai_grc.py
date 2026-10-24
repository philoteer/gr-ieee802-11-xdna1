#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Exercise CPU equalizers with the existing GRC symbol-conformance corpus")
    parser.add_argument('--source', type=Path, default=root.parent / 'AI-model-CNN-Channel-est/test/frame_equalizer/test-symbols.grc')
    parser.add_argument('--build', type=Path, default=root / 'build')
    parser.add_argument('--debug-block', action='store_true')
    args = parser.parse_args()
    output = args.build.resolve() / 'channel_ai_grc'
    output.mkdir(parents=True, exist_ok=True)
    for active in (False, True):
        for algorithm in ('LS', 'LMS', 'STA', 'CNN_FRAME',
                          'CNN_CHANNEL', 'CNN_FRAME_MLP', 'CNN_CHANNEL_MLP',
                          'CNN_FRAME_INT8', 'CNN_CHANNEL_INT8',
                          'CNN_FRAME_MLP_INT8', 'CNN_CHANNEL_MLP_INT8',
                          'CNN_BUS_TO_ROOM', 'CNN_ROOM_TO_BUS',
                          'CNN_BUS_TO_ROOM_INT8', 'CNN_ROOM_TO_BUS_INT8',
                          'CNN_BUS_TO_ROOM_MLP', 'CNN_ROOM_TO_BUS_MLP',
                          'CNN_BUS_TO_ROOM_MLP_INT8', 'CNN_ROOM_TO_BUS_MLP_INT8'):
            graph = yaml.safe_load(args.source.read_text())
            removed = {'mlir_aie_frame_equalizer_npu', 'blocks_stream_to_vector_npu',
                       'blocks_npu_decisions_sink', 'blocks_vector_to_stream'}
            graph['blocks'] = [block for block in graph['blocks'] if block['name'] not in removed]
            graph['connections'] = [edge for edge in graph['connections']
                                    if edge[0] not in removed and edge[2] not in removed]
            for block in graph['blocks']:
                params = block['parameters']
                if block['name'] == 'algorithm':
                    params['value'] = repr(algorithm.lower())
                elif block['name'] == 'algorithm_id':
                    params['value'] = str({'LS': 0, 'LMS': 1, 'STA': 3,
                                           'CNN_FRAME': 6, 'CNN_CHANNEL': 7,
                                           'CNN_FRAME_MLP': 8, 'CNN_CHANNEL_MLP': 9,
                                           'CNN_FRAME_INT8': 10, 'CNN_CHANNEL_INT8': 11,
                                           'CNN_FRAME_MLP_INT8': 12, 'CNN_CHANNEL_MLP_INT8': 13,
                                           'CNN_BUS_TO_ROOM': 14, 'CNN_ROOM_TO_BUS': 15,
                                           'CNN_BUS_TO_ROOM_INT8': 16, 'CNN_ROOM_TO_BUS_INT8': 17,
                                           'CNN_BUS_TO_ROOM_MLP': 18, 'CNN_ROOM_TO_BUS_MLP': 19,
                                           'CNN_BUS_TO_ROOM_MLP_INT8': 20, 'CNN_ROOM_TO_BUS_MLP_INT8': 21}[algorithm])
                elif block['name'] == 'ieee802_11_frame_equalizer_reference' and args.debug_block:
                    block['id'] = 'ieee802_11_frame_equalizer_debug'
                elif block['name'] == 'zz_build_guard_and_impairment':
                    code = params['_source_code']
                    start, end = code.index('def _require_build'), code.index('def _profiles')
                    code = code[:start] + code[end:]
                    code = code.replace('_require_build(self.algorithm)', 'pass')
                    if active:
                        code = code.replace('return np.concatenate((flat, tracked, damaged))',
                                            'flat[0::4] *= 1.1\n    flat[1::4] *= 0.9\n'
                                            '    tracked[0] *= 1.1\n    tracked[1] *= 0.9\n'
                                            '    damaged[0] *= 1.1\n    damaged[1] *= 0.9\n'
                                            '    return np.concatenate((flat, tracked, damaged))')
                    params['_source_code'] = code
                elif block['name'] == 'verify':
                    params['code'] = """import numpy as np
import pmt
expected = np.asarray(self.payload_data, dtype=np.uint8).reshape(-1, 48)
actual = np.asarray(self.blocks_gnu_decisions_sink.data(), dtype=np.uint8).reshape(-1, 48)
np.testing.assert_array_equal(actual, expected)
tags = self.blocks_gnu_decisions_sink.tags()
encodings = [pmt.to_uint64(tag.value) for tag in tags if pmt.symbol_to_string(tag.key) == 'encoding']
assert encodings == list(range(8)) + [5, 0], encodings
csi = [tag for tag in tags if pmt.symbol_to_string(tag.key) == 'csi']
assert len(csi) == 10 and all(len(pmt.c32vector_elements(tag.value)) == 52 for tag in csi)
assert self.blocks_symbols_debug.num_messages() == 17
print('CPU ' + self.algorithm + ': all rates, tracked frame, SIGNAL decoding, CSI tags, and symbol PDUs passed')
"""
            path = output / 'cpu-symbols.grc'
            path.write_text(yaml.safe_dump(graph, sort_keys=False))
            subprocess.run(['grcc', '-o', str(output), str(path)], check=True,
                           stdout=subprocess.DEVNULL)
            wrapper = """import sys, runpy
from gnuradio import gr, digital
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, __import__('os').path.dirname(sys.argv[2]))
import ieee802_11_python
sys.modules['ieee802_11'] = ieee802_11_python
for factory in (ieee802_11_python.frame_equalizer, ieee802_11_python.frame_equalizer_debug):
    block = factory(ieee802_11_python.CNN_FRAME, 2.417e9, 20e6, False, False)
    for name in ('CNN_CHANNEL', 'CNN_FRAME_MLP', 'CNN_CHANNEL_MLP', 'CNN_FRAME',
                 'CNN_FRAME_INT8', 'CNN_CHANNEL_INT8', 'CNN_FRAME_MLP_INT8', 'CNN_CHANNEL_MLP_INT8',
                 'CNN_BUS_TO_ROOM', 'CNN_ROOM_TO_BUS', 'CNN_BUS_TO_ROOM_INT8', 'CNN_ROOM_TO_BUS_INT8',
                 'CNN_BUS_TO_ROOM_MLP', 'CNN_ROOM_TO_BUS_MLP',
                 'CNN_BUS_TO_ROOM_MLP_INT8', 'CNN_ROOM_TO_BUS_MLP_INT8'):
        block.set_algorithm(getattr(ieee802_11_python, name))
    del block
runpy.run_path(sys.argv[2], run_name='__main__')
"""
            print(f'{algorithm}: {"active" if active else "gated"} inference', flush=True)
            subprocess.run([sys.executable, '-c', wrapper,
                            str(args.build.resolve() / 'python/bindings'),
                            str(output / 'test_symbols_grc.py')], check=True, timeout=60)


if __name__ == '__main__':
    main()
