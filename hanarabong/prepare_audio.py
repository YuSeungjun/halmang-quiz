"""Render completed catalog sentences using the same voice as live listening."""
from pathlib import Path
from tempfile import NamedTemporaryFile

from catalog import AUDIO, audio_key, eligible_questions
from speech import render_wav


def main():
    sentences = dict.fromkeys(q['sentence'] for q in eligible_questions())
    AUDIO.mkdir(exist_ok=True)
    for sentence in sentences:
        target = AUDIO / f'{audio_key(sentence)}.wav'
        if target.is_file():
            continue
        data = render_wav(sentence)
        with NamedTemporaryFile(dir=AUDIO, suffix='.wav', delete=False) as tmp:
            tmp.write(data)
        Path(tmp.name).replace(target)
        print(sentence, target.name, flush=True)


if __name__ == '__main__':
    main()
