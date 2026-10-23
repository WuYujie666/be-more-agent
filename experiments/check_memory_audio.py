"""Live file-ASR/memory acceptance test across fresh processes; no mic/TTS/GUI.
Uses a disposable memory store and the production ChatEngine/ASR/LLM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TextIO:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def worker(phase, directory):
    from config import CURRENT_CONFIG
    CURRENT_CONFIG.update(long_term_memory=True, memory_dir=str(directory / 'memory'),
                          summaries_file=str(directory / 'summaries.json'))
    from chat_engine import ChatEngine
    out = {'phase': phase, 'checks': {}, 'seconds': {}}
    engine = None
    try:
        if phase == 'write':
            import asr_backend
            audio = ROOT / 'input_test_data.wav'
            with wave.open(str(audio)) as source:
                out['duration'] = source.getnframes() / source.getframerate()
            out['audio_sha256'] = hashlib.sha256(audio.read_bytes()).hexdigest()
            started = time.perf_counter()
            asr_backend.warmup()
            out['seconds']['asr_load'] = time.perf_counter() - started
            started = time.perf_counter()
            out['transcript'] = asr_backend.transcribe(audio)
            out['seconds']['asr'] = time.perf_counter() - started
            assert out['transcript'], 'Empty ASR transcript'
        engine = ChatEngine(TextIO())
        engine.memory.ready.result(60)
        if phase == 'write':
            started = time.perf_counter()
            engine.handle_turn(out['transcript'])
            out['seconds']['reply'] = time.perf_counter() - started
            out['reply'] = ''.join(engine.io.spoken)
            out['checks']['reply_present'] = bool(out['reply'])
            out['checks']['no_mid_session_write'] = not engine.memory.submit('list').result(20)['results']
            started = time.perf_counter()
            engine.finalize_session_memory()
            engine.memory.submit('flush').result(90)
            out['seconds']['finalize_and_flush'] = time.perf_counter() - started
            out['records'] = engine.memory.submit('list').result(20)['results']
            out['checks']['session_saved'] = bool(out['records'])
        elif phase == 'recall':
            out['checks']['fresh_session_empty'] = not engine.session_memory
            out['records'] = engine.memory.submit('list').result(20)['results']
            out['checks']['persisted_after_process_restart'] = bool(out['records'])
            question = '我之前那顿饭吃了哪些东西？后来为什么有食物没吃完，吃完回哪里了？'
            out['question'] = question
            out['retrieved'] = engine.memory.search(question)
            started = time.perf_counter()
            out['answer'] = engine.chat_and_respond(question)
            out['seconds']['recall_reply'] = time.perf_counter() - started
            out['checks']['food_details'] = all(x in out['answer'] for x in ['米线', '蛋羹', '西红柿', '西兰花'])
            out['checks']['destination'] = '寝室' in out['answer']
            out['checks']['reason'] = '饱' in out['answer'] or '多' in out['answer']
            engine.handle_turn('清空记忆')
            out['checks']['clear_records'] = not engine.memory.submit('list').result(20)['results']
            out['checks']['clear_summary'] = not (directory / 'summaries.json').exists()
        else:
            out['checks']['clear_survives_process_restart'] = not engine.memory.submit('list').result(20)['results']
            out['checks']['search_empty'] = not engine.memory.search('我之前吃了什么饭？')
    finally:
        if engine:
            engine.close()
        (directory / (phase + '.json')).write_text(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--phase', choices=['write', 'recall', 'verify'])
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.phase:
        worker(args.phase, args.directory)
        return
    result = {'scope': 'File ASR + production ChatEngine + session write + process restart + recall + clear; no GUI/VAD/TTS', 'phases': []}
    target = ROOT / 'experiments/memory_audio_results.json'
    with tempfile.TemporaryDirectory(prefix='memory-audio-', dir=ROOT / 'runtime') as temp:
        directory = Path(temp)
        for phase in ['write', 'recall', 'verify']:
            completed = subprocess.run([sys.executable, '-B', __file__, '--live', '--phase', phase,
                                        '--directory', str(directory)], cwd=ROOT, timeout=240)
            path = directory / (phase + '.json')
            if path.exists():
                result['phases'].append(json.loads(path.read_text()))
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            completed.check_returncode()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert all(ok for phase in result['phases'] for ok in phase['checks'].values()), 'Review failed checks'


if __name__ == '__main__':
    main()
