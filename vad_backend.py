"""Local streaming VAD; cache ONNX sessions, reset recurrent state per recording."""
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=2)
def _session(path):
    import onnxruntime as ort
    if not Path(path).is_file():
        raise FileNotFoundError(f'Silero VAD model missing: {path}. Run venv/bin/python setup_vad.py')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])


class SileroVad:
    frame_samples = 512
    frame_ms = 32

    def __init__(self, config):
        path = Path(config.get('silero_vad_model', 'models/silero-vad/silero_vad.onnx'))
        self.session = _session(str(path if path.is_absolute() else ROOT / path))
        self.threshold = float(config.get('silero_vad_threshold', 0.5))
        if not 0 < self.threshold < 1:
            raise ValueError('silero_vad_threshold must be between 0 and 1')
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.probability = None

    def is_speech(self, frame):
        frame = np.asarray(frame, dtype=np.int16).reshape(-1)
        if len(frame) != self.frame_samples:
            raise ValueError('Silero requires 512 samples at 16000 Hz')
        samples = frame.astype(np.float32)[None, :] / 32768
        samples = np.concatenate((self.context, samples), axis=1)
        output, self.state = self.session.run(None, {
            'input': samples, 'state': self.state, 'sr': np.array(16000, dtype=np.int64)})
        self.context = samples[:, -64:].copy()
        self.probability = float(output.reshape(-1)[0])
        return self.probability >= self.threshold


class WebRtcVad:
    frame_samples = 480
    frame_ms = 30
    probability = None

    def __init__(self, config):
        import webrtcvad
        self.vad = webrtcvad.Vad(int(config.get('vad_aggressiveness', 3)))

    def is_speech(self, frame):
        return self.vad.is_speech(np.asarray(frame, dtype=np.int16).tobytes(), 16000)


def create_vad(config):
    backend = config.get('vad_backend', 'silero')
    if backend == 'silero':
        return SileroVad(config)
    if backend == 'webrtc':
        return WebRtcVad(config)
    raise ValueError(f'Unknown VAD backend: {backend}')
