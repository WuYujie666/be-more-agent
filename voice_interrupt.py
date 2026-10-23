"""Full-duplex voice capture and playback, isolated from GUI/dialogue code.

SpeexDSP AEC uses timestamp-aligned actual output as its reference. Silero and
segmentation run outside PortAudio's callback. No ASR/LLM work in either thread.
"""
from collections import deque
import ctypes as C
import ctypes.util
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import threading
import time
import uuid
import wave
import numpy as np
import scipy.signal
import sounddevice as sd
from vad_backend import create_vad
from vad_diagnostics import VadDiagnostics


class EchoCanceller:
    def __init__(self, rate=48000, frame=480):
        self.lib = C.CDLL(ctypes.util.find_library('speexdsp') or 'libspeexdsp.so.1')
        ptr = C.c_void_p
        pcm = np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS')
        for name, args, result in [
            ('speex_echo_state_init', [C.c_int, C.c_int], ptr),
            ('speex_echo_ctl', [ptr, C.c_int, ptr], C.c_int),
            ('speex_echo_cancellation', [ptr, pcm, pcm, pcm], None),
            ('speex_echo_state_destroy', [ptr], None),
            ('speex_preprocess_state_init', [C.c_int, C.c_int], ptr),
            ('speex_preprocess_ctl', [ptr, C.c_int, ptr], C.c_int),
            ('speex_preprocess_run', [ptr, pcm], C.c_int),
            ('speex_preprocess_state_destroy', [ptr], None),
        ]:
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.echo = self.lib.speex_echo_state_init(frame, rate // 5)
        self.pre = self.lib.speex_preprocess_state_init(frame, rate)
        if not self.echo or not self.pre:
            self.close()
            raise RuntimeError('Could not initialize SpeexDSP')
        self.lib.speex_echo_ctl(self.echo, 24, C.byref(C.c_int(rate)))
        self.lib.speex_preprocess_ctl(self.pre, 24, self.echo)
        self.lib.speex_preprocess_ctl(self.pre, 20, C.byref(C.c_int(-40)))
        self.lib.speex_preprocess_ctl(self.pre, 22, C.byref(C.c_int(-15)))
        self.frame = frame

    def process(self, mic, reference):
        mic = np.ascontiguousarray(np.clip(mic * 32768, -32768, 32767), dtype=np.int16)
        ref = np.ascontiguousarray(np.clip(reference * 32768, -32768, 32767), dtype=np.int16)
        if len(mic) != self.frame or len(ref) != self.frame:
            raise ValueError('Invalid AEC frame size')
        out = np.empty_like(mic)
        self.lib.speex_echo_cancellation(self.echo, mic, ref, out)
        self.lib.speex_preprocess_run(self.pre, out)
        return out

    def close(self):
        if getattr(self, 'pre', None):
            self.lib.speex_preprocess_state_destroy(self.pre)
            self.pre = None
        if getattr(self, 'echo', None):
            self.lib.speex_echo_state_destroy(self.echo)
            self.echo = None


@dataclass
class Utterance:
    path: str
    diagnostics: VadDiagnostics
    speech_end: float
    recording_end: float


class VoiceInterrupt:
    def __init__(self, config, on_interrupt, is_busy):
        self.config = config
        self.on_interrupt, self.is_busy = on_interrupt, is_busy
        self.rate = int(config.get('voice_duplex_rate', 48000))
        if self.rate not in (16000, 48000):
            raise ValueError('voice_duplex_rate must be 16000 or 48000')
        self.block = self.rate // 100
        self.closed = threading.Event()
        self.recording = threading.Event()
        self.incoming = queue.Queue(maxsize=200)
        self.utterances = queue.Queue(maxsize=4)
        self.error = None
        self.stream = None
        self.worker = None
        self.lock = threading.Lock()
        self.output = None
        self.output_pos = 0
        self.output_cancelled = lambda: False
        self.references = deque(maxlen=200)
        self.trace = deque(maxlen=3000)
        self.trace_lock = threading.Lock()
        self.trace_enabled = bool(config.get('debug_prompt'))
        self.meter_frames = 0
        self.meter_max = 0.0
        self.meter_rms = 0.0
        self.meter_overflows = 0
        self.echo = EchoCanceller(self.rate, self.block)
        self.vad = create_vad(config)
        self._reset_segment()

    def _reset_segment(self):
        self.recording.clear()
        self.run = self.silence = 0
        self.buffer = []
        self.preroll = deque(maxlen=max(1, math.ceil(self.config.get('vad_preroll_ms', 300)/self.vad.frame_ms), math.ceil(self.config.get('barge_in_start_ms',256)/self.vad.frame_ms)))
        self.diagnostics = VadDiagnostics(self.config, self.preroll.maxlen, self.vad.frame_ms)
        self.last_voice = None

    def start(self):
        try:
            self.stream = sd.Stream(samplerate=self.rate, blocksize=self.block,
                channels=(1, 1), dtype='float32', device=(self.config.get('input_device'), self.config.get('output_device')),
                callback=self._callback)
            self.worker = threading.Thread(target=self._run, daemon=True, name='voice-input')
            self.worker.start()
            self.stream.start()
        except Exception:
            self.close()
            raise
        print('[VOICE] Full duplex: Speex AEC + VAD; voice interruption enabled', flush=True)

    def _callback(self, indata, outdata, frames, clocks, status):
        outdata.fill(0)
        with self.lock:
            if self.output is not None:
                if self.output_cancelled():
                    self.output = None
                else:
                    n = min(frames, len(self.output)-self.output_pos)
                    outdata[:n, 0] = self.output[self.output_pos:self.output_pos+n]
                    self.output_pos += n
                    if self.output_pos >= len(self.output):
                        self.output = None
        # Capture and render times use PortAudio's common stream clock.
        self.references.append((clocks.outputBufferDacTime, outdata[:,0].copy()))
        target = clocks.inputBufferAdcTime - self.config.get('aec_reference_delay_ms', 10)/1000
        reference = np.zeros(frames, np.float32)
        for stamp, block in reversed(self.references):
            if stamp <= target:
                if target-stamp < .03:
                    reference = block
                break
        try:
            self.incoming.put_nowait((indata[:,0].copy(), reference, bool(status)))
        except queue.Full:
            self.error = RuntimeError('Voice input cannot keep up; stopping instead of using stale audio')
            self.closed.set()
            outdata.fill(0)
            raise sd.CallbackAbort

    def play(self, samples, rate, cancelled):
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != self.rate:
            audio = scipy.signal.resample_poly(audio, self.rate, rate).astype(np.float32)
        with self.lock:
            if cancelled() or self.closed.is_set():
                return
            self.output_cancelled = cancelled
            self.output_pos = 0
            self.output = audio
        while not self.closed.wait(.01):
            if cancelled():
                self.stop_playback()
                return
            with self.lock:
                if self.output is None:
                    break
        # Submitted output still needs to drain through the device.
        latency = self.stream.latency[1] if self.stream else 0
        deadline = time.monotonic() + latency
        while time.monotonic() < deadline and not self.closed.wait(.01):
            if cancelled():
                break

    def save_trace(self):
        if not self.trace_enabled:
            return
        with self.trace_lock:
            frames = list(self.trace)
        if not frames:
            return
        directory = Path('runtime/voice-traces')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.wav')
        # Channels: raw microphone, aligned speaker reference, processed mic.
        raw = np.concatenate([x[0] for x in frames])
        ref = np.concatenate([x[1] for x in frames])
        clean = np.concatenate([x[2] for x in frames])
        data = np.column_stack((np.clip(raw*32768,-32768,32767), np.clip(ref*32768,-32768,32767), clean)).astype('<i2')
        with wave.open(str(path),'wb') as wav:
            wav.setparams((3,2,self.rate,0,'NONE','not compressed'))
            wav.writeframes(data.tobytes())
        for old in sorted(directory.glob('*.wav'))[:-10]:
            old.unlink(missing_ok=True)
        print(f'[VOICE TRACE] raw/reference/clean: {path}', flush=True)

    def stop_playback(self):
        with self.lock:
            self.output = None

    def _run(self):
        pending = np.empty(0, np.int16)
        try:
            while not self.closed.is_set():
                try:
                    mic, reference, overflow = self.incoming.get(timeout=.1)
                except queue.Empty:
                    continue
                clean = self.echo.process(mic, reference)
                if self.trace_enabled:
                    with self.trace_lock:
                        self.trace.append((mic.copy(), reference.copy(), clean.copy()))
                self.meter_overflows += int(overflow)
                if self.rate != 16000:
                    clean = np.clip(np.rint(scipy.signal.resample_poly(clean.astype(np.float32), 16000, self.rate)), -32768,32767).astype(np.int16)
                pending = np.concatenate((pending, clean))
                while len(pending) >= self.vad.frame_samples:
                    frame, pending = pending[:self.vad.frame_samples], pending[self.vad.frame_samples:]
                    self._frame(frame, overflow)
        except Exception as exc:
            self.error = exc
            self.closed.set()
            self.stop_playback()
            print(f'[VOICE] Failed: {exc}', flush=True)

    def _frame(self, frame, overflow=False):
        speech = self.vad.is_speech(frame)
        busy = self.is_busy()
        if self.trace_enabled:
            self.meter_frames += 1
            self.meter_max = max(self.meter_max, self.vad.probability or 0)
            self.meter_rms = max(self.meter_rms, float(np.sqrt(np.mean(frame.astype(np.float64)**2))))
            if self.meter_frames >= 32:
                if busy:
                    db = 20*np.log10(max(self.meter_rms, .001)/32768)
                    print(f'[VOICE METER] p_max={self.meter_max:.3f} clean_peak_rms={db:.1f}dBFS run={self.run} recording={self.recording.is_set()} overflows={self.meter_overflows}', flush=True)
                self.meter_frames = 0
                self.meter_max = self.meter_rms = 0.0
                self.meter_overflows = 0
        if busy and not self.recording.is_set() and self.vad.probability is not None:
            speech = self.vad.probability >= float(self.config.get('barge_in_threshold', .7))
        self.diagnostics.observe(frame, speech, self.recording.is_set(), overflow, probability=self.vad.probability)
        if speech:
            self.last_voice = time.perf_counter()
        if not self.recording.is_set():
            self.preroll.append(frame.copy())
            self.run = self.run+1 if speech else 0
            start_ms = self.config.get('barge_in_start_ms',256) if busy else self.config.get('vad_start_ms',150)
            if self.run < math.ceil(start_ms/self.vad.frame_ms):
                return
            self.buffer = list(self.preroll)
            self.diagnostics.trigger()
            self.recording.set()
            if busy:
                print('[BARGE-IN] Confirmed user speech; cancelling reply', flush=True)
                self.on_interrupt()
        else:
            self.buffer.append(frame.copy())
            self.silence = 0 if speech else self.silence+1
            if self.silence*self.vad.frame_ms >= self.config.get('vad_silence_ms',900):
                self._finish('trailing_silence')
            elif len(self.buffer)*self.vad.frame_ms >= self.config.get('vad_max_record_ms',30000):
                self._finish('max_duration')

    def _finish(self, reason):
        directory = Path('runtime/voice-input')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (uuid.uuid4().hex+'.wav')
        with wave.open(str(path),'wb') as wav:
            wav.setparams((1,2,16000,0,'NONE','not compressed'))
            wav.writeframes(np.concatenate(self.buffer).astype('<i2').tobytes())
        ended = time.perf_counter()
        self.diagnostics.save(str(path), self.rate, reason)
        item = Utterance(str(path), self.diagnostics, self.last_voice, ended)
        try:
            self.utterances.put_nowait(item)
        except queue.Full:
            path.unlink(missing_ok=True)
            raise RuntimeError('Too many pending utterances')
        self._reset_segment()

    def receive(self):
        while not self.closed.is_set():
            try:
                return self.utterances.get(timeout=.1)
            except queue.Empty:
                pass
        if self.error:
            raise RuntimeError('Voice input failed') from self.error
        return None

    def close(self):
        self.closed.set()
        self.stop_playback()
        if self.stream:
            self.stream.abort()
            self.stream.close()
        if self.worker and self.worker is not threading.current_thread():
            self.worker.join(timeout=2)
        if not self.worker or not self.worker.is_alive():
            self.echo.close()
        while not self.utterances.empty():
            try:
                Path(self.utterances.get_nowait().path).unlink(missing_ok=True)
            except queue.Empty:
                break
