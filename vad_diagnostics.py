"""Local, bounded VAD captures. Metrics describe detector decisions, not ground truth."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import uuid
import numpy as np


def dbfs(rms):
    return round(float(20 * np.log10(max(float(rms), 1e-6) / 32768)), 2)


class VadDiagnostics:
    def __init__(self, config, preroll_frames, frame_ms=30):
        self.config = config
        self.frame_ms = frame_ms
        self.enabled = config.get('vad_diagnostics_enabled', True)
        self.preroll = deque(maxlen=preroll_frames)
        self.noise = deque(maxlen=max(1, 5000 // frame_ms))
        self.frames = []
        self.noise_at_trigger = None
        self.overflows = 0
        self.path = None
        self.report = None

    def observe(self, frame, speech, recording, overflow=False, probability=None):
        if not self.enabled:
            return
        samples = np.asarray(frame, dtype=np.float64)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        row = {'speech': bool(speech), 'rms_dbfs': dbfs(rms),
               'peak_dbfs': dbfs(np.max(np.abs(samples))), 'rms': rms}
        if probability is not None:
            row['speech_probability'] = float(probability)
        self.overflows += int(bool(overflow))
        if recording:
            self.frames.append(row)
        else:
            self.preroll.append(row)
            if not speech:
                self.noise.append(rms)

    def trigger(self):
        if not self.enabled:
            return
        self.frames.extend(self.preroll)
        self.preroll.clear()
        if self.noise:
            self.noise_at_trigger = dbfs(np.median(self.noise))

    def save(self, audio_file, input_rate, stop_reason):
        if not self.enabled:
            return
        try:
            directory = Path(self.config.get('vad_diagnostics_dir', 'diagnostics/vad'))
            directory.mkdir(parents=True, exist_ok=True)
            capture_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8]
            self.path = directory / (capture_id + '.json')
            wav = self.path.with_suffix('.wav')
            shutil.copyfile(audio_file, wav)
            count = len(self.frames)
            voiced = [f for f in self.frames if f['speech']]
            rms = np.sqrt(np.mean([f['rms'] ** 2 for f in self.frames])) if count else 0
            self.report = {
                'capture_id': capture_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                'audio_file': wav.name, 'sample_rate': 16000, 'input_sample_rate': input_rate,
                'stop_reason': stop_reason, 'frame_ms': self.frame_ms,
                'duration_ms': count * self.frame_ms, 'vad_speech_ms': len(voiced) * self.frame_ms,
                'vad_speech_ratio': round(len(voiced) / count, 4) if count else 0,
                'rms_dbfs': dbfs(rms), 'noise_floor_dbfs': self.noise_at_trigger,
                'speech_above_noise_db': round(float(np.median([f['rms_dbfs'] for f in voiced])) - self.noise_at_trigger, 2) if voiced and self.noise_at_trigger is not None else None,
                'input_overflows_since_listen': self.overflows,
                'vad_config': {k: self.config.get(k) for k in ('voice_interrupt_enabled', 'barge_in_threshold', 'barge_in_start_ms', 'aec_reference_delay_ms', 'vad_backend', 'silero_vad_model', 'silero_vad_threshold', 'vad_aggressiveness', 'vad_start_ms', 'vad_silence_ms', 'vad_max_record_ms', 'vad_preroll_ms')},
                'asr_backend': self.config.get('asr_backend', 'whisper'), 'asr_status': 'pending',
                'frames': [{k: v for k, v in f.items() if k != 'rms'} for f in self.frames],
            }
            self._write()
            keep = max(1, int(self.config.get('vad_diagnostics_max_captures', 100)))
            for old in sorted(directory.glob('????????T??????_*Z_*.json'))[:-keep]:
                old.with_suffix('.wav').unlink(missing_ok=True)
                old.unlink(missing_ok=True)
            print(f'[VAD DIAG] {self.path} speech={len(voiced)*self.frame_ms}ms rms={dbfs(rms)}dBFS noise={self.noise_at_trigger}', flush=True)
        except Exception as exc:
            print(f'[VAD DIAG] Save failed: {exc}', flush=True)

    def finish_asr(self, text=None, error=None):
        if self.report is None:
            return
        try:
            self.report.update(asr_status='error' if error is not None else 'ok', asr_text=text, asr_error=error)
            self._write()
        except Exception as exc:
            print(f'[VAD DIAG] ASR metadata update failed: {exc}', flush=True)

    def _write(self):
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(self.path)
