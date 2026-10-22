"""Persistent SenseVoice recognizer; Whisper fallback stays in agent.py."""
from pathlib import Path
import threading
import wave
import numpy as np
from config import CURRENT_CONFIG

_ROOT = Path(__file__).resolve().parent
_lock = threading.Lock()
_recognizer = None


def warmup():
    global _recognizer
    with _lock:
        if _recognizer is None:
            import sherpa_onnx
            directory = _ROOT / CURRENT_CONFIG.get('sensevoice_model_dir', 'sherpa-models/sensevoice-int8')
            _recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(directory / 'model.int8.onnx'), tokens=str(directory / 'tokens.txt'),
                num_threads=CURRENT_CONFIG.get('sensevoice_threads', 4),
                language='zh', use_itn=True,
            )
    return _recognizer


def transcribe(filename):
    recognizer = warmup()
    with wave.open(str(filename), 'rb') as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError('SenseVoice input must be mono 16-bit PCM WAV')
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768
    if not len(samples):
        return ''
    with _lock:
        stream = recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        recognizer.decode_stream(stream)
    return stream.result.text.strip()
