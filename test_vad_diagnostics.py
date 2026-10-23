import json
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from vad_diagnostics import VadDiagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_capture_alignment_asr_and_retention(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'input.wav'
            with wave.open(str(source), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(np.zeros(1440, dtype=np.int16).tobytes())
            config = {'vad_diagnostics_dir': str(Path(root) / 'captures'), 'vad_diagnostics_max_captures': 1}
            for _ in range(2):
                diag = VadDiagnostics(config, 2)
                diag.observe(np.full(480, 100, dtype=np.int16), False, False)
                diag.observe(np.full(480, 1000, dtype=np.int16), True, False)
                diag.trigger()
                diag.observe(np.zeros(480, dtype=np.int16), False, True, True)
                diag.save(source, 48000, 'trailing_silence')
                diag.finish_asr(text='嗯')
            files = list((Path(root) / 'captures').glob('*.json'))
            self.assertEqual(len(files), 1)
            self.assertEqual(len(list(files[0].parent.glob('*.wav'))), 1)
            report = json.loads(files[0].read_text())
            self.assertEqual(report['duration_ms'], 90)
            self.assertEqual(report['vad_speech_ms'], 30)
            self.assertEqual(report['speech_above_noise_db'], 20)
            self.assertEqual(report['input_overflows_since_listen'], 1)
            self.assertEqual(report['asr_text'], '嗯')
            self.assertEqual(files[0].with_suffix('.wav').read_bytes(), source.read_bytes())
            diag.finish_asr(error='test failure')
            self.assertEqual(json.loads(files[0].read_text())['asr_status'], 'error')

    def test_disabled_and_unavailable_directory(self):
        diag = VadDiagnostics({'vad_diagnostics_enabled': False}, 0)
        diag.observe(np.zeros(480), False, False)
        diag.trigger()
        diag.save('missing.wav', 16000, 'test')
        diag.finish_asr(text='test')
        self.assertIsNone(diag.path)
        with tempfile.TemporaryDirectory() as root:
            blocked = Path(root) / 'file'
            blocked.write_text('not a directory')
            diag = VadDiagnostics({'vad_diagnostics_dir': str(blocked)}, 0)
            diag.save('missing.wav', 16000, 'test')
            diag.finish_asr(text='test')
            self.assertIsNone(diag.report)


if __name__ == '__main__':
    unittest.main()
