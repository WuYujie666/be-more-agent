"""Compare saved 16 kHz mono PCM with WebRTC and official Silero ONNX.
Usage: python experiments/compare_vad.py AUDIO --model MODEL --output REPORT
Silero state/context handling follows upstream OnnxWrapper (512 + 64 samples).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import wave
import numpy as np
import onnxruntime as ort
import webrtcvad


def summary(flags, frame_ms):
    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return {'speech_ms': sum(flags) * frame_ms,
            'longest_consecutive_speech_ms': longest * frame_ms,
            'triggers_150ms': longest >= math.ceil(150 / frame_ms)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with wave.open(args.audio, 'rb') as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
    results = {'audio': args.audio, 'audio_sha256': hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
               'duration_ms': len(pcm)/16, 'silero_model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
               'note': 'Fresh detector state at file start; live WebRTC state may differ. Silence-pad final partial frame.'}
    for mode in range(4):
        vad = webrtcvad.Vad(mode)
        start = time.perf_counter()
        flags = [vad.is_speech(np.pad(pcm[i:i+480], (0, max(0, 480-len(pcm[i:i+480])))).tobytes(), 16000)
                 for i in range(0, len(pcm), 480)]
        results[f'webrtc_{mode}'] = {**summary(flags, 30), 'elapsed_ms': (time.perf_counter()-start)*1000, 'frame_flags': flags}
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, sess_options=opts, providers=['CPUExecutionProvider'])
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros((1, 64), np.float32)
    samples = pcm.astype(np.float32)/32768
    probabilities = []
    start = time.perf_counter()
    for i in range(0, len(samples), 512):
        chunk = samples[i:i+512]
        chunk = np.pad(chunk, (0, 512-len(chunk)))[None, :]
        full = np.concatenate((context, chunk), axis=1)
        out, state = session.run(None, {'input': full, 'state': state, 'sr': np.array(16000, dtype=np.int64)})
        context = full[:, -64:]
        probabilities.append(float(out.reshape(-1)[0]))
    results['silero'] = {'elapsed_ms': (time.perf_counter()-start)*1000,
                         'max_probability': max(probabilities), 'frame_probabilities': probabilities,
                         'thresholds': {str(t): summary([p >= t for p in probabilities], 32) for t in (0.3, 0.5, 0.7)}}
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps({k: {a:b for a,b in v.items() if not a.startswith('frame_')} if isinstance(v, dict) else v for k,v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
