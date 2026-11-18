"""Read the same source material and versioned choice reviews as the research site."""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'test'))
from server import CATALOG, QuestionLab, read_json, read_state  # noqa: E402
from choice_review import meanings  # noqa: E402

AUDIO = HERE / 'audio'
RUN = PROJECT / 'training/runs/jeju_aihub_jss/20261118-voxcpm-sft'
# The packaged quiz recordings use this checkpoint even on machines without a training run.
SELECTION = RUN / 'best.json'
CHECKPOINT = Path(read_json(SELECTION)['path']).name if SELECTION.is_file() else 'step_0002362'
REFERENCE = PROJECT / 'training/datasets/voice03/reference.json'


def audio_key(sentence):
    identity = f'{RUN.name}/{CHECKPOINT}/voice03\n{sentence}'
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def eligible_questions():
    lab = QuestionLab(read_json(CATALOG / 'sources.json'), read_json(CATALOG / 'questions.json'))
    statuses = read_state().get('questions', {})
    rows = []
    for q in lab.catalog_questions():
        if q['choiceReview']['state'] != 'unique' or statuses.get(q['id'], {}).get('status') == 'rejected':
            continue
        glosses = {word: option_meaning(q, word, lab.corpus) for word in q['options']}
        rows.append({**q, 'optionMeanings': glosses})
    return rows


def option_meaning(question, word, corpus):
    if word == question['answer']:
        return question['meaning']
    for definition in meanings(corpus, word):
        # Dictionary entries can start with a chain of equivalent dialect spellings.
        gloss = re.sub(r'^(?:=[^\s.]+\.?\s*)+', '', definition.strip())
        if gloss and not gloss.startswith('⇒'):
            return gloss
    return question['choiceReview']['choices'][word]['이유']


def learner_question(q):
    row = {k: q[k] for k in ('id', 'prompt', 'sentence', 'standard', 'meaning', 'answer',
                             'hint', 'category', 'options', 'optionMeanings')}
    row['audio'] = f"/audio/{audio_key(q['sentence'])}.wav"
    return row
