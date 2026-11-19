"""Compare generation of held-out JIT test turns independently from retrieval."""
import argparse
import json
import random
import sqlite3
from contextlib import closing
from pathlib import Path

from sacrebleu.metrics import CHRF
import translation as tr
from train_translation import LANGS, LESSON, usable, key


def test_pairs(count):
    folder = tr.PROJECT / 'data/jit'
    with (folder / 'ko.test').open() as ko, (folder / 'je.test').open() as je:
        pairs = [(tr.clean(a), tr.clean(b)) for a,b in zip(ko, je, strict=True)]
    with closing(sqlite3.connect(tr.INDEX.as_uri() + '?mode=ro', uri=True)) as db:
        seen = {key(t) for row in db.execute('SELECT ko,je FROM pair') for t in row}
    seen.update(key(t) for row in LESSON for t in row)
    random.Random(119).shuffle(pairs)
    selected = []
    for ko,je in pairs:
        if usable(ko,je) and not {key(ko), key(je)} & seen:
            selected.append((ko,je))
            seen.update((key(ko), key(je)))
            if len(selected) == count:
                return selected
    raise ValueError('Not enough unseen test pairs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=24)
    args = parser.parse_args()
    tests = test_pairs(args.count)
    results = {}
    for direction, language in LANGS.items():
        turns = []
        for ko,je in tests:
            source, target = (ko,je) if direction == 'ko-je' else (je,ko)
            prompt = f'{language} 번역한다. 뜻, 시제, 높임과 말투를 유지한다. translation 필드에 완성된 번역문만 쓴다.'
            reply = tr.ollama_json(prompt, {'입력 문장':source}, tr.SCHEMA)
            turns.append({'source':source, 'reference':target, 'output':reply.get('translation','')})
        score = CHRF().corpus_score([t['output'] for t in turns], [[t['reference'] for t in turns]])
        results[direction] = {'chrF':round(score.score, 2), 'turns':turns}
        print(tr.DRAFT_MODEL, direction, results[direction]['chrF'], flush=True)
    args.output.write_text(json.dumps({'model':tr.DRAFT_MODEL,'evaluation':results},ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
