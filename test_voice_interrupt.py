import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from voice_interrupt import VoiceInterrupt


class BargeInTests(unittest.TestCase):
    def test_speech_onset_keeps_preroll_and_cancels_once(self):
        hits=[]
        voice=VoiceInterrupt({'vad_diagnostics_enabled':False},lambda:hits.append(True),lambda:True)
        try:
            voice.vad.is_speech=lambda frame:True
            voice.vad.probability=.95
            for _ in range(7): voice._frame(np.ones(512,np.int16))
            self.assertEqual(hits,[])
            voice._frame(np.ones(512,np.int16))
            self.assertEqual(hits,[True])
            self.assertEqual(len(voice.buffer),8)
            voice._frame(np.ones(512,np.int16))
            self.assertEqual(hits,[True])
        finally: voice.close()

    @unittest.skipUnless(Path('diagnostics/vad/positive_normal_soft_short.wav').exists(), 'Local voice sample required')
    def test_near_end_voice_survives_aec(self):
        import wave
        import scipy.signal
        hits=[]
        voice=VoiceInterrupt({'vad_diagnostics_enabled':False},lambda:hits.append(True),lambda:True)
        try:
            with wave.open('diagnostics/vad/positive_normal_soft_short.wav','rb') as wav:
                audio=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
            audio=scipy.signal.resample_poly(audio,3,1)
            pending=np.empty(0,np.int16)
            # Real near-end sentence, no speaker echo: AEC must preserve it.
            for i in range(0, int(4.5*48000),480):
                clean=voice.echo.process(audio[i:i+480],np.zeros(480,np.float32))
                clean=np.clip(np.rint(scipy.signal.resample_poly(clean.astype(np.float32),1,3)),-32768,32767).astype(np.int16)
                pending=np.concatenate((pending,clean))
                while len(pending)>=512:
                    voice._frame(pending[:512]);pending=pending[512:]
            self.assertEqual(hits,[True])
        finally: voice.close()

    def test_cancelled_transition_does_not_start_sleep_audio(self):
        from chat_engine import ChatEngine
        from unittest.mock import Mock
        engine=ChatEngine.__new__(ChatEngine)
        engine.io=Mock()
        engine.io.is_interrupted.return_value=True
        engine.tools=None
        engine.audio_type='white_noise'
        engine.chat_and_respond=Mock()
        engine.start_session_memory=Mock()
        engine.handle_audio_answer('好')
        engine.io.play_relaxation_audio.assert_not_called()
        engine.start_session_memory.assert_not_called()

    def test_cancel_stops_output_without_closing_input(self):
        voice=VoiceInterrupt({},lambda:None,lambda:False)
        try:
            cancel=threading.Event()
            thread=threading.Thread(target=voice.play,args=(np.ones(48000,np.float32),48000,cancel.is_set))
            thread.start()
            deadline=time.monotonic()+1
            while voice.output is None and time.monotonic()<deadline: time.sleep(.001)
            cancel.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertIsNone(voice.output)
            self.assertFalse(voice.closed.is_set())
        finally: voice.close()

    def test_stale_synthesis_dropped_after_new_turn(self):
        import agent
        bot=agent.BotGUI.__new__(agent.BotGUI)
        bot.exiting=False
        bot.reply_id=0
        bot.interrupted=threading.Event()
        bot.tts_queue=[(0,'old reply')]
        bot.audio_queue=[]
        bot.audio_queue_max=2
        bot.tts_queue_lock=threading.Lock()
        bot.audio_queue_lock=threading.Lock()
        bot.synth_active=threading.Event()
        started=threading.Event(); release=threading.Event()
        def render(text):
            started.set(); release.wait(2)
            return np.ones(100),16000
        bot._render=render
        thread=threading.Thread(target=bot._synth_worker)
        thread.start()
        try:
            self.assertTrue(started.wait(1))
            bot.interrupted.set(); bot.reply_id+=1
            bot.interrupted.clear()  # new turn before old synthesis finishes
            release.set()
            deadline=time.monotonic()+1
            while bot.synth_active.is_set() and time.monotonic()<deadline: time.sleep(.005)
            self.assertEqual(bot.audio_queue,[])
        finally:
            bot.exiting=True; release.set(); thread.join(2)


if __name__=='__main__':unittest.main()
