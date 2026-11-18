"""Retrieve paired quotations, compare local LLM generations and publish validated drafts."""
import hashlib
import json
import os
import random
import re
import threading
import uuid
from datetime import datetime, timezone
from choice_review import fingerprint, review_choices
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'storage' / 'generation'
MODEL = 'qwen3:32b'
ENDPOINT = 'http://127.0.0.1:18765'
FIELDS = ('sentence', 'standard', 'answer', 'meaning', 'distractors', 'hint', 'explanation')
SCHEMA = {'type': 'object', 'required': list(FIELDS), 'additionalProperties': False,
          'properties': {k: {'type': 'array', 'items': {'type': 'string'}} if k == 'distractors'
                         else {'type': 'string'} for k in FIELDS}}
PROMPT = """제주어 학습자를 위한 5지선다 빈칸 문제 한 개를 JSON으로 작성한다. 제주특별자치도 생활방언에 실제로 나오는 짧은 제주어 문장(sentence)과 같은 줄의 표준어 풀이(standard)를 원문 그대로 쓴다. answer는 문장 안에 한 번 나오는 제주어 단어 하나다. 조사나 어미를 포함해 가릴 부분을 정확히 정한다. 표준어와 같은 단어는 피한다. meaning은 해당 문맥의 표준어 뜻이다. distractors에는 정답과 품사·용법이 비슷하지만 표준어 풀이로 배제되는 실제 제주어 단어 네 개를 쓴다. 뜻이 비슷한 말이나 같은 단어의 표기 변형은 보기에서 제외한다. hint는 정답을 그대로 알려주지 않는 짧은 한국어 단서, explanation은 정답을 넣어 문맥을 풀어주는 한 문장이다. 자료가 주어지면 제주어와 표준어를 글자·띄어쓰기·문장부호까지 복사하고, 어휘 후보 중에서 보기를 고른다. 별표나 따옴표를 덧붙이지 않는다. 힌트는 뜻을 반복하지 말고 문맥을 짚어 준다. 단어를 정의하는 힌트라면 오답도 같은 정의에 해당하는지 살펴 구별되는 단서를 쓴다. 자료가 없으면 알고 있는 표현으로 같은 형식의 문항을 시도한다. 근거가 부족하면 해당 필드는 비워 둔다."""
TOPICS = {
    '바다': '바다 바당 해녀 전복 소라 파도 물고기',
    '가족': '가족 할머니 할아버지 어머니 아버지 손자 부부',
    '음식': '음식 먹 식사 마시 달걀 전복죽 마늘 밥 감자 고기',
    '바람': '바람 보롬 보름 샛바람 갈바람',
    '인사': '인사 안녕 편안 반갑 어서 오십시오 오셨',
    '여행': '여행 가느냐 어디 길 한라산 서귀포 한림공원',
    '자연': '자연 바다 바람 산 나무 꽃 새 구름',
}
# Small teaching vocabulary grouped by use; definitions and attestations come from the local files.
CHOICE_GROUPS = [
    '하르방 할망 아방 어멍 아시 손지 비바리 새서방',
    '바당 질 고팡 빌레 낭 보롬',
    '마농 괴기 지슬 독새기 도새기 구젱기',
    '누게 무사 어드레 어느제 무싱거 어떵',
    '혼저 촌촌이 하영 호꼼 몬딱 솔째기 안트레',
    '구덕 허벅 바농 낭 독새기',
]
GROUP = {word: words.split() for words in CHOICE_GROUPS for word in words.split()}
PARTICLES = r'(?:이|가|은|는|을|를|도|만|에|이영|영|이랑|랑|들|은게|이나|나|과|와|으로|로|디|신디|보다|엔|이우|이주|이우다|이꽈|이여|이네)*'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def word_present(word, sentence):
    return re.search(r'(?<![가-힣])' + re.escape(word) + PARTICLES + r'(?![가-힣])', sentence) is not None


def pairs(row):
    a, b = ([line.strip() for line in row[k].splitlines() if line.strip()] for k in ('contents', 'solution'))
    if len(a) != len(b):
        return
    for line, translation in zip(a, b):
        dialect, standard = (re.split(r'(?<=[.!?])\s+', value) for value in (line, translation))
        if len(dialect) == len(standard):
            yield from zip(dialect, standard)
        else:
            yield line, translation


class Corpus:
    def __init__(self, sources, seeds):
        self.rows = {row['seq']: row for row in sources['life']}
        self.utterances = '\n'.join(row['contents'] for row in self.rows.values())
        self.dictionary = {}
        for row in sources['dictionary']:
            self.dictionary.setdefault(row['siteName'], []).append(row)
        self.gloss = {w: rows[0]['contents'] for w, rows in self.dictionary.items()
                      if re.fullmatch('[가-힣]{1,10}', w)}
        self.lines = [(sid, s, t) for sid, row in self.rows.items() for s, t in pairs(row)
                      if s and t]
        self.fixed = {word: meaning for _, _, _, word, meaning, _, _ in seeds}
        self.gloss.update(self.fixed)
        self.attested = set(self.dictionary) | {w for w in self.fixed if any(word_present(w, s) for _, s, _ in self.lines)}
        self.choices = []
        for sid, s, t in self.lines:
            matches = []
            for token in re.findall('[가-힣]+', s):
                for n in range(1, len(token) + 1):
                    w = token[:n]
                    if w not in self.gloss or w in t or s.count(w) != 1 or not word_present(w, s):
                        continue
                    glosses = [r['contents'] for r in self.dictionary.get(w, [])]
                    if w in self.fixed:
                        glosses.insert(0, self.fixed[w])
                    for gloss in glosses:
                        meanings = re.findall('[가-힣]+', gloss)
                        if any(re.search(r'(?<![가-힣])' + re.escape(m), t) for m in meanings[:3] if len(m) >= 2 or m == gloss):
                            matches.append((w, gloss))
                            break
            if matches:
                self.choices.append((sid, s, t, matches))

    def retrieve(self, topic, used, used_words, used_sources):
        terms = TOPICS.get(topic, topic).split()
        ranked = []
        for sid, s, t, matches in self.choices:
            if any(s in previous or previous in s for previous in used) or not 5 <= len(s) <= 85 or not s.endswith(('.', '!', '?')):
                continue
            title = self.rows[sid]['siteName']
            in_line = sum(bool(re.search(r'(?<![가-힣])' + re.escape(term), s + ' ' + t)) for term in terms)
            in_title = sum(term in title for term in terms)
            if not in_line and not in_title:
                continue
            viable = [(w, g) for w, g in matches if w in GROUP and w not in used_words]
            if viable:
                matched, meaning = max(viable, key=lambda p: (len(p[0]) >= 2, p[0] in self.fixed, len(p[0])))
                topical_word = any(term in matched or term in meaning for term in terms)
                score = (bool(in_line), topical_word, sid not in used_sources, in_line + in_title, len(matched) >= 2, -len(s))
                ranked.append((score, sid, s, t, matched, meaning))
        if not ranked:
            return None
        _, sid, s, t, word, meaning = max(ranked)
        vocab = {w: self.gloss[w] for w in GROUP[word] if w in self.attested and w in self.gloss
                 and (w == word or w not in s)}
        vocab[word] = meaning
        if len(vocab) < 5:
            return None
        return {'source': sid, 'sentence': s, 'standard': t, 'target': word, 'vocabulary': vocab}

    def assess(self, item):
        texts = [item.get(k) for k in FIELDS if k != 'distractors']
        filled = all(isinstance(v, str) and 0 < len(v.strip()) <= 500 for v in texts)
        sentence, standard, answer = (item.get(k, '') for k in ('sentence', 'standard', 'answer'))
        choices = item.get('distractors')
        options = [answer, *choices] if isinstance(choices, list) and isinstance(answer, str) else []
        shape = (filled and len(options) == 5 and all(isinstance(w, str) and re.fullmatch('[가-힣ㆍᆞ]+', w)
                                                     for w in options) and len(set(options)) == 5
                 and sentence.count(answer) == 1 and '____' not in sentence and word_present(answer, sentence))
        exact = filled and any(sentence in s and standard in t for _, s, t in self.lines)
        words = len(options) == 5 and all(isinstance(w, str) and 0 < len(w) <= 25 and
                                           (w in self.dictionary or word_present(w, self.utterances)) for w in options)
        return {'format': bool(shape), 'quotation': bool(exact), 'vocabulary': words,
                'passed': bool(shape and exact and words)}

    def publishable(self, trial):
        item = dict(trial['rag']['output'])
        evidence = trial['evidence']
        word = evidence['target']
        changes = []
        sentence = item.get('sentence')
        if isinstance(sentence, str) and sentence != evidence['sentence']:
            plain = sentence.replace('**' + word + '**', word, 1)
            if plain == evidence['sentence']:
                item['sentence'] = plain
                changes.append('정답을 감싼 Markdown 강조 표시 제거')
        others = item.get('distractors')
        if (isinstance(others, list) and len(others) > 4 and all(isinstance(w, str) and w != word
                and w in evidence['vocabulary'] for w in others) and len(set(others)) == len(others)):
            item['distractors'] = others[:4]
            changes.append('출처가 확인된 오답 중 앞 네 개 선택')
        if (self.assess(item)['passed'] and item['sentence'] == evidence['sentence']
                and item['standard'] == evidence['standard'] and item['answer'] == word
                and all(w in evidence['vocabulary'] for w in item['distractors'])):
            q = self.draft(item, evidence, trial['topic'], trial['seed'])
            check = review_choices(q, self, model_json)
            trial['choiceReviews'] = [check]
            if check['state'] != 'unique':
                # Replace only competing/uncertain distractors, keeping the question and answer fixed.
                verdicts = check['choices']
                if verdicts.get(word, {}).get('판정') != '맞음':
                    return None
                competing = [w for w in item['distractors'] if verdicts[w]['판정'] != '아님']
                replacements = [w for w in sorted(evidence['vocabulary']) if w not in q['options']]
                if not competing or len(replacements) < len(competing):
                    return None
                swaps = dict(zip(competing, replacements))
                item['distractors'] = [swaps.get(w, w) for w in item['distractors']]
                if not self.assess(item)['passed']:
                    return None
                q = self.draft(item, evidence, trial['topic'], trial['seed'])
                check = review_choices(q, self, model_json)
                trial['choiceReviews'].append(check)
                if check['state'] != 'unique':
                    return None
                changes.extend(f'복수정답 가능 보기 교체: {w} → {new}' for w, new in swaps.items())
            q['choiceReview'] = check
            trial.update(questionId=q['id'], publicationEdits=changes)
            return q
        return None

    def draft(self, item, evidence, topic, seed):
        sid, answer = evidence['source'], item['answer']
        row = self.rows[sid]
        key = hashlib.sha256((sid + '\n' + item['sentence'] + '\n' + answer).encode()).hexdigest()[:12]
        options = [answer, *item['distractors']]
        random.Random(seed).shuffle(options)
        return {**{k: item[k] for k in FIELDS if k != 'distractors'}, 'id': 'RAG-' + key,
                'category': topic, 'difficulty': '검토 전', 'options': options,
                'prompt': item['sentence'].replace(answer, '____', 1),
                'source': {'id': int(sid), 'title': row['siteName'], 'dataset': '제주특별자치도 생활방언',
                           'url': 'https://www.jeju.go.kr/culture/dialect/lifeDialect.htm?act=view&seq=' + sid,
                           'original': row['contents'], 'translation': row['solution'], 'audio': '/api/audio/life-' + sid},
                'dictionaryMatches': [{'word': answer, 'definition': r['contents'],
                                       'url': 'https://www.jeju.go.kr/culture/dialect/dictionary.htm?act=view&seq=' + r['seq']}
                                      for r in self.dictionary.get(answer, [])],
                'generation': {'author': MODEL, 'method': '생활방언·사전 검색 후 LLM 생성', 'review': '제주어 화자 검토 전',
                               'checks': ['원문·대응 풀이 일치', '빈칸 1개', '중복 없는 보기 5개', '보기 어휘 출처 확인']}}


def generate(topic, focus, evidence, seed):
    content = f'주제 {topic} · 출제하려는 단어의 표준어 뜻 {focus}\n'
    if evidence:
        content += json.dumps(evidence, ensure_ascii=False) + '\n가릴 단어는 target을 사용한다.'
    return model_json(PROMPT, content, SCHEMA, seed)


def model_json(system, content, schema, seed):
    body = {'model': MODEL, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            'think': False, 'stream': False, 'format': schema, 'keep_alive': '1m',
            'options': {'temperature': 0, 'seed': seed, 'num_ctx': 4096, 'num_predict': 600}}
    request = Request(ENDPOINT + '/api/chat', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=180) as response:
        raw = json.load(response)['message']['content']
    try:
        item = json.loads(raw)
        return item if isinstance(item, dict) else {'invalidOutput': raw}
    except ValueError:
        return {'invalidOutput': raw}


def metrics(trials):
    return {arm: {key: sum(r[arm]['checks'][key] for r in trials) for key in ('format', 'quotation', 'vocabulary', 'passed')}
            for arm in ('rag', 'baseline')}


class QuestionLab:
    def __init__(self, sources, originals):
        RUNS.mkdir(parents=True, exist_ok=True)
        self.corpus = Corpus(sources, read(ROOT / 'question_seeds.json'))
        self.originals = originals
        review_path = ROOT / 'storage' / 'choice-reviews.json'
        self.existing_reviews = read(review_path) if review_path.is_file() else {}
        self.lock = threading.Lock()
        self.reports = [read(p) for p in sorted(RUNS.glob('*.json'))]
        self.active = None

    def drafts(self):
        return [q for r in self.reports for q in r['questions']]

    def current_review(self, q):
        wanted = fingerprint(q)
        for review in (self.existing_reviews.get(q['id']), q.get('choiceReview')):
            if review and review.get('fingerprint') == wanted:
                return review
        return {'state': 'unchecked', 'choices': {}}

    def catalog_questions(self):
        return [{**q, 'choiceReview': self.current_review(q)} for q in self.originals + self.drafts()]

    def status(self):
        with self.lock:
            latest = self.reports[-1] if self.reports else None
            summary = {k: latest[k] for k in ('id', 'topics', 'requested', 'tested', 'scores', 'added')} if latest else None
            return {'active': dict(self.active) if self.active else None, 'latest': summary}

    def start(self, topics):
        if not isinstance(topics, str) or len(topics) > 120:
            raise ValueError('주제를 120자 이내로 입력해 주세요.')
        names = list(dict.fromkeys(t.strip() for t in topics.split(',') if t.strip()))
        if not 1 <= len(names) <= 10 or any(not re.fullmatch('[가-힣A-Za-z0-9 ]{1,20}', t) for t in names):
            raise ValueError('주제를 한글·영문·숫자로 입력하고 여러 주제는 쉼표로 구분해 주세요.')
        with self.lock:
            if self.active and self.active['status'] == 'running':
                raise RuntimeError('이미 문제 생성이 진행 중입니다.')
            taken = {q['sentence'] for q in self.originals + self.drafts()}
            jobs, words, sources = [], set(), set()
            while len(jobs) < 10:
                before = len(jobs)
                for topic in names:
                    evidence = self.corpus.retrieve(topic, taken, words, sources)
                    if evidence:
                        jobs.append((topic, evidence))
                        taken.add(evidence['sentence'])
                        words.add(evidence['target'])
                        sources.add(evidence['source'])
                    if len(jobs) == 10:
                        break
                if len(jobs) == before:
                    break
            if not jobs:
                raise ValueError('이 주제에서 새 문장과 대응 어휘를 찾지 못했어요. 다른 주제를 입력해 주세요.')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            run_id = stamp + '-' + uuid.uuid4().hex[:8]
            self.active = {'id': run_id, 'status': 'running', 'completed': 0, 'total': len(jobs), 'topics': names}
            threading.Thread(target=self.run, args=(run_id, names, jobs), daemon=True).start()
            return dict(self.active)

    def run(self, run_id, topics, jobs):
        trials, questions = [], []
        try:
            for i, (topic, evidence) in enumerate(jobs):
                focus = evidence['vocabulary'][evidence['target']]
                trial = {'topic': topic, 'focus': focus, 'seed': 2026 + i, 'evidence': evidence}
                for arm in (('rag', 'baseline') if i % 2 == 0 else ('baseline', 'rag')):
                    item = generate(topic, focus, evidence if arm == 'rag' else None, trial['seed'])
                    checks = self.corpus.assess(item)
                    trial[arm] = {'output': item, 'checks': checks}
                q = self.corpus.publishable(trial)
                if q:
                    questions.append(q)
                trials.append(trial)
                with self.lock:
                    self.active['completed'] = len(trials)
            report = {'id': run_id, 'model': MODEL, 'topics': topics, 'requested': 10, 'tested': len(trials),
                      'added': len(questions), 'scores': metrics(trials), 'trials': trials, 'questions': questions,
                      'systemPrompt': PROMPT, 'temperature': 0, 'humanReview': 'pending',
                      'scoring': '두 조건 모두 모델 원출력으로 채점. 게시용 형식 보정은 publicationEdits에 기록.'}
            tmp = RUNS / (run_id + '.tmp')
            tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            with self.lock:
                os.replace(tmp, RUNS / (run_id + '.json'))
                self.reports.append(report)
                self.active = None
        except (OSError, URLError, TimeoutError, ValueError, KeyError, TypeError) as error:
            with self.lock:
                self.active.update(status='error', error='생성이 중단되었습니다. 로컬 모델 서버를 확인하고 다시 시도해 주세요.')
            print(f'문제 생성 {run_id}: {error}', flush=True)
