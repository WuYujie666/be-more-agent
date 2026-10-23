import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np
from vad_backend import create_vad

ROOT = Path(__file__).resolve().parent
NEGATIVE = ROOT / 'diagnostics/vad/20261022T044724_881765Z_7dcc5e14.wav'
POSITIVE = ROOT / 'diagnostics/vad/positive_normal_soft_short.wav'
MODEL = ROOT / 'models/silero-vad/silero_vad.onnx'


def samples(path):
    with wave.open(str(path), 'rb') as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')


@unittest.skipUnless(MODEL.exists(), 'Local Silero model required')
class VadTests(unittest.TestCase):
    def test_reset_and_validation(self):
        first = create_vad({})
        second = create_vad({})
        self.assertIs(first.session, second.session)
        self.assertIsNot(first.state, second.state)
        for vad in (first, second):
            self.assertFalse(vad.is_speech(np.zeros(512, np.int16)))
        self.assertEqual(first.probability, second.probability)
        with self.assertRaises(ValueError):
            first.is_speech(np.zeros(480, np.int16))
        with self.assertRaises(ValueError):
            create_vad({'vad_backend': 'invalid'})

    @unittest.skipUnless(NEGATIVE.exists() and POSITIVE.exists(), 'Local captured fixtures required')
    def test_labeled_recordings(self):
        vad = create_vad({})
        negative = samples(NEGATIVE)
        self.assertFalse(any(vad.is_speech(negative[i:i+512]) for i in range(0, len(negative)-511, 512)))
        vad = create_vad({})
        positive = samples(POSITIVE)
        flags = [vad.is_speech(positive[i:i+512]) for i in range(0, len(positive)-511, 512)]
        for start, end in [(1.2,4.5),(9,13),(15,16.2),(17,18)]:
            longest = run = 0
            for i, flag in enumerate(flags):
                if start <= i*.032 < end:
                    run = run+1 if flag else 0
                    longest = max(longest,run)
            self.assertGreaterEqual(longest, 5)

    @unittest.skipUnless(NEGATIVE.exists() and POSITIVE.exists(), 'Local captured fixtures required')
    def test_production_recording_loop(self):
        import agent
        for path, should_capture in [(NEGATIVE,False),(POSITIVE,True)]:
            bot=agent.BotGUI.__new__(agent.BotGUI)
            bot.exiting=False
            audio=np.concatenate([np.zeros(4096,np.int16), samples(path), np.zeros(16000,np.int16)])
            # Simulate native USB rate to exercise the production resampler too.
            audio=agent.scipy.signal.resample_poly(audio.astype(np.float32),3,1).astype(np.int16)
            class Input:
                offset=0
                def __init__(self,**kwargs): pass
                def __enter__(self): return self
                def __exit__(self,*args): pass
                def read(self,n):
                    chunk=audio[self.offset:self.offset+n]
                    self.offset+=n
                    if self.offset>=len(audio): bot.exiting=True
                    return np.pad(chunk,(0,n-len(chunk)))[:,None],False
            with tempfile.TemporaryDirectory() as directory:
                settings={**agent.CURRENT_CONFIG,'vad_backend':'silero','vad_diagnostics_dir':directory}
                with patch.dict(agent.CURRENT_CONFIG,settings),patch.object(agent.sd,'InputStream',Input),patch.object(agent.sd,'stop'),patch.object(agent,'choose_input_samplerate',return_value=48000),patch.object(agent.time,'sleep'):
                    result=bot.record_voice_vad(directory+'/input.wav')
                self.assertEqual(bool(result),should_capture)
                if result:
                    report=json.loads(bot.vad_diagnostics.path.read_text())
                    with wave.open(result,'rb') as wav:
                        self.assertEqual(wav.getnframes()/16,report['duration_ms'])
                    self.assertEqual(report['frame_ms'],32)
                    self.assertTrue(all('speech_probability' in f for f in report['frames']))
                    self.assertEqual(report['stop_reason'],'trailing_silence')


if __name__=='__main__': unittest.main()
