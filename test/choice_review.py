"""Blindly evaluate every completion against the displayed translation."""
import hashlib
import json
import re

VERSION = 1
JUDGE_PROMPT = """제주어 빈칸 문제의 독립 검토자다. 제시된 표준어 풀이와 빈칸 주변 문맥이 평가 기준이다. 각 보기를 넣은 문장이 이 표준어 문장과 같은 내용을 전달하는지 각각 판정하라. 정답 수는 0개일 수도 여러 개일 수도 있다. 딱 하나를 고르는 문제가 아니다. 단지 문법적으로 자연스러워도 번역의 뜻이 달라지면 '아님'이다. 동의어·다른 표기·높임말이라도 같은 뜻으로 읽힐 수 있다면 모두 '맞음'이다. 다의어는 문맥에 들어맞는 뜻이 하나라도 있는지 본다. 지역 어휘의 근거가 불충분해서 배제할 수 없으면 '불확실'이다. 보기별로 짧은 한국어 이유와 판정을 반환한다."""
VERDICTS = ['맞음', '아님', '불확실']


def fingerprint(q):
    fields = [VERSION, q['prompt'], q['standard'], q['answer'], sorted(q['options'])]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def references(definition):
    return [w.replace('-', '') for w in re.findall(r'[⇒=]\s*([가-힣-]+)', definition)]


def meanings(corpus, word):
    glosses = [r['contents'] for r in corpus.dictionary.get(word, [])]
    if word in corpus.fixed:
        glosses.insert(0, corpus.fixed[word])
    for gloss in glosses.copy():
        for alias in references(gloss):
            glosses.extend(r['contents'] for r in corpus.dictionary.get(alias, []))
    return list(dict.fromkeys(glosses))


def review_choices(q, corpus, model_json):
    words = sorted(q['options'])
    result = {'state': 'uncertain', 'fingerprint': fingerprint(q), 'choices': {}}
    if len(words) != 5 or len(set(words)) != 5 or q['answer'] not in words or q['prompt'].count('____') != 1:
        return result
    glosses = {word: meanings(corpus, word) for word in words}
    # Send no answer key, completed source quotation, authored hint or explanation.
    task = {'빈칸': q['prompt'], '표준어': q['standard'], '검토할 보기': [
        {'보기': word, '대입한 문장': q['prompt'].replace('____', word), '자료의 뜻': glosses[word]}
        for word in words]}
    entry = {'type': 'object', 'required': ['판정', '이유'], 'additionalProperties': False,
             'properties': {'판정': {'type': 'string', 'enum': VERDICTS}, '이유': {'type': 'string'}}}
    schema = {'type': 'object', 'required': words, 'properties': {w: entry for w in words}, 'additionalProperties': False}
    judged = model_json(JUDGE_PROMPT, json.dumps(task, ensure_ascii=False), schema, 2026)
    if not isinstance(judged, dict) or set(judged) != set(words):
        return result
    for word in words:
        row = judged[word]
        if (not isinstance(row, dict) or row.get('판정') not in VERDICTS
                or not isinstance(row.get('이유'), str) or not row['이유'].strip()):
            return result
    # A blind model can still overlook an explicitly shared dictionary sense.
    # Limit this veto to short, identical senses visible in the displayed translation.
    answer = q['answer']
    target_senses = {g.strip(' .') for g in glosses[answer] if g.strip(' .') in q['standard']}
    for word in words:
        shared = sorted(target_senses & {g.strip(' .') for g in glosses[word]} - {''})
        if word != answer and shared and judged[word]['판정'] == '아님':
            judged[word] = {'판정': '불확실', '이유': '두 보기의 자료 풀이가 겹칩니다: ' + ', '.join(shared)}
    result['choices'] = judged
    fits = [w for w in words if judged[w]['판정'] == '맞음']
    if len(fits) > 1:
        result['state'] = 'ambiguous'
    elif any(r['판정'] == '불확실' for r in judged.values()):
        result['state'] = 'uncertain'
    elif fits == [answer]:
        result['state'] = 'unique'
    else:
        result['state'] = 'mismatch'
    return result
