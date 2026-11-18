"""Expose completed listening experiments, never model/checkpoint files."""
import json
import wave
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TRAINING = PROJECT / 'training'
RUNS = TRAINING / 'runs' / 'voice03'
DATA = TRAINING / 'datasets' / 'voice03'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_training_catalog():
    catalog = {'voiceId': 'voice-03', 'voiceName': '목소리 03', 'items': [], 'models': []}
    media = {}
    if not (DATA / 'listening-test.json').is_file():
        return catalog, media

    def clip(path, media_id, allowed_root):
        path = path.resolve()
        if not path.is_relative_to(allowed_root.resolve()) or not path.is_file():
            return None
        try:
            with wave.open(str(path), 'rb') as audio:
                duration = audio.getnframes() / audio.getframerate()
        except (wave.Error, EOFError):
            return None
        media[media_id] = str(path.relative_to(PROJECT))
        return {'id': media_id, 'audio': f'/api/audio/{media_id}', 'duration': round(duration, 3)}

    rows = read_json(DATA / 'listening-test.json')
    for index, row in enumerate(rows, 1):
        # Use the exact prepared test clip. Do not take arbitrary paths from API requests.
        original = clip(DATA / 'audio' / f"{row['id']}.wav", f'tts03-{index:02}-original', DATA / 'audio')
        catalog['items'].append({'id': f'tts03-{index:02}', 'number': index, 'text': row['text'],
                                 'kind': '낭독' if row['kind'] == 'read_aloud' else '문답',
                                 'original': original, 'samples': {}})
    experiments = [
        (TRAINING / 'runs' / 'jeju_aihub_jss' / '20261118-voxcpm-sft', {
            'id': 'voxcpm-jss', 'name': 'VoxCPM2 · AI Hub + JSS', 'phases': ['after'],
            'scope': '새 학습 결과', 'featured': True,
            'description': 'AI Hub 47.9시간 + JSS 12.2시간 · 총 60.1시간 학습',
        }),
        (TRAINING / 'runs' / 'jeju_multispeaker' / '20261117-voxcpm-sft', {
            'id': 'voxcpm-jeju', 'name': 'VoxCPM2 · 제주어 다화자', 'phases': ['after'],
            'scope': 'AI Hub만 학습', 'wide': True,
            'description': '제주 화자 489명 · 약 47.9시간 학습',
        }),
        *[(RUNS / f'20261115-{model}-sft', {
            'id': model, 'name': name, 'phases': ['before', 'after'], 'scope': '목소리 03 학습',
        }) for model, name in [('qwen', 'Qwen3-TTS'), ('voxcpm', 'VoxCPM2')]],
    ]
    for run, model in experiments:
        status = run / 'status.json'
        if not status.is_file() or read_json(status).get('stage') != 'complete':
            continue
        catalog['models'].append(model)
        for phase in model['phases']:
            index_path = run / 'samples' / phase / 'index.json'
            indexed = {r['id']: r for r in read_json(index_path)} if index_path.is_file() else {}
            for item in catalog['items']:
                number = item['number']
                record = indexed.get(number)
                if not record or record.get('text') != item['text']:
                    continue
                media_id = f"{item['id']}-{model['id']}-{phase}"
                audio = clip(run / 'samples' / phase / f'{number:02}.wav', media_id, run / 'samples')
                if audio:
                    item['samples'][f"{model['id']}-{phase}"] = audio
    return catalog, media
