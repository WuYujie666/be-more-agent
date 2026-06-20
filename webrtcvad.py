"""webrtcvad mock for Windows GUI testing. Replaced by real module on Pi."""
import warnings
warnings.warn("Using mock webrtcvad — VAD will NOT work on Windows")


class Vad:
    def __init__(self, mode=2):
        self.mode = mode

    def is_speech(self, buf, sample_rate):
        return False
