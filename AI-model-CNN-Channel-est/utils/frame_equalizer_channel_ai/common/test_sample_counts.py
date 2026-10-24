from pathlib import Path
import pytest

from utils.frame_equalizer_channel_ai.common import sample_counts
from utils.frame_equalizer_channel_ai.common.updater import frame_examples, load_cnn_header
from utils.frame_equalizer_channel_ai.dataset_generator.convert_sigmf import Frame
from utils.frame_equalizer_channel_ai.dataset_generator.test_convert_sigmf import _frame
from utils.frame_equalizer_channel_ai.dataset_generator.convert_sigmf import frame_examples as cnn_examples


@pytest.mark.parametrize('target_mode', ('averaged', 'single_symbol'))
def test_counts_match_prepared_examples(monkeypatch, target_mode):
    frames = []
    for recording, symbols in (('206', 6), ('206', 10), ('207', 20)):
        samples = _frame(symbols)
        frames.append(Frame(f'{recording}.sigmf-meta', 0, 1,
                            cnn_examples(samples, 8, 4), samples))
    monkeypatch.setattr(sample_counts, 'load_splits',
                        lambda *args, **kwargs: ({name: frames for name in ('train', 'val', 'test')},
                                                 {name: {} for name in ('train', 'val', 'test')}))
    header = Path(__file__).resolve().parents[3] / 'kernel/channel_estimator/model/frame_equalizer_channel_ai_weights_by_frame.h'
    cnn = load_cnn_header(header)
    selected = frames[:2]
    expected_cnn = sum(len(frame.corpus.inputs_q15) for frame in selected)
    expected_mlp = sum(len(frame_examples(frame, cnn, 8, 4, target_mode)[0]) for frame in selected)
    summary = sample_counts.count_samples(Path('unused'), target_mode=target_mode, recording='206')
    assert summary['samples']['cnn'] == dict.fromkeys(('train', 'validation', 'test'), expected_cnn)
    assert summary['samples']['mlp'] == dict.fromkeys(('train', 'validation', 'test'), expected_mlp)
    assert expected_mlp == (8 if target_mode == 'averaged' else 12)


def test_cli_report(tmp_path, monkeypatch, capsys):
    summary = {'samples': {'cnn': {'train': 9, 'validation': 2, 'test': 3}}}
    calls = []
    monkeypatch.setattr(sample_counts, 'count_samples', lambda *args: calls.append(args) or summary.copy())
    report = tmp_path / 'counts.json'
    monkeypatch.setattr('sys.argv', ['sample_counts.py', '--dataset', 'bus_207', '--model', 'cnn',
                                    '--report', str(report)])
    sample_counts.main()
    assert calls[0][0].name == '20260928_bus'
    assert calls[0][-1] == '207'
    assert calls[0][-2] == 'cnn'
    assert report.read_text() == capsys.readouterr().out
