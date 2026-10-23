"""Install the tested official Silero ONNX model, verifying its content hash."""
import hashlib
from pathlib import Path
import urllib.request

MODEL_URL = 'https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx'
MODEL_SHA256 = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
TARGET = Path(__file__).resolve().parent / 'models/silero-vad/silero_vad.onnx'


def main():
    if TARGET.exists() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == MODEL_SHA256:
        print(f'Silero model ready: {TARGET}')
        return
    data = urllib.request.urlopen(MODEL_URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise RuntimeError('Silero download hash mismatch; existing model was not changed')
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix('.download')
    temporary.write_bytes(data)
    temporary.replace(TARGET)
    print(f'Silero model installed: {TARGET}')


if __name__ == '__main__':
    main()
