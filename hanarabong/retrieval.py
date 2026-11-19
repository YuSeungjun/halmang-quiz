"""Index standard Korean predicates and conversational context alongside the parallel texts."""
import fcntl
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

# The side index is disposable. The authoritative translations stay in pair.
VERSION = '2-kiwi-0.22.2'
MAX_CLAUSES = 8
PUNCT = '.!?。！？'
TOPICS = {'아빠': '부모', '아버지': '부모', '엄마': '부모', '어머니': '부모',
          '같이': '동행', '함께': '동행', '먹': '식사', '드시': '식사', '식사': '식사'}
LESS_SPECIFIC = {'하', '되', '있', '없', '보', '것', '거', '수', '저', '나', '당신'}


@lru_cache(maxsize=1)
def analyzer():
    from kiwipiepy import Kiwi
    return Kiwi(num_workers=1)


def clauses(text):
    """Split unpunctuated turns at final endings, preserving direct quotations."""
    if any(c in text for c in '"“”‘’「」'):
        return [text]
    breaks = set()
    for token in analyzer().tokenize(text):
        end = token.start + token.len
        if token.tag == 'SF' or (token.tag == 'EF' and end < len(text) and text[end].isspace()):
            breaks.add(end)
    parts, start = [], 0
    for end in sorted(breaks | {len(text)}):
        piece = text[start:end].strip()
        if piece and piece.strip(PUNCT):
            parts.append(piece)
        start = end
    return parts


def features(text, tokens):
    content, grammar = set(), set()
    last_ending = ''
    for i, token in enumerate(tokens):
        form, tag = token.form, token.tag.partition('-')[0]
        if tag in ('VV', 'VA'):
            content.add('v' + form)
        elif tag in ('NNG', 'NNP', 'SL', 'MAG', 'NP'):
            content.add('n' + form)
        if tag == 'XSV' and i and tokens[i-1].tag in ('NNG', 'NNP'):
            content.add('v' + tokens[i-1].form + form)
        if tag == 'VX' and form == '하' and i >= 2 and tokens[i-2].form in ('좋', '싫'):
            stem = tokens[i-2].form
            content.discard('v' + stem)
            content.add('v' + ('좋아하' if stem == '좋' else '싫어하'))
        if form in TOPICS:
            content.add('t' + TOPICS[form])
        if tag == 'EP' and ('었' in form or '았' in form or '였' in form):
            grammar.add('past')
        if tag == 'VX' and form == '있':
            grammar.add('ongoing')
        if (tag == 'MAG' and form in ('안', '못')) or (tag == 'VX' and form == '않'):
            grammar.add('negative')
        if tag == 'EF':
            last_ending = form
    if last_ending:
        polite = re.search(r'요$|니다$|니까$|시오$|십시오$', last_ending) or text.rstrip(PUNCT).endswith('요')
        grammar.add('polite' if polite else 'plain')
    if re.search(r'(?:[는간한된온란]대|[느다냐]고)(?:요)?[?？]$', text):
        grammar.add('reported')
    if re.search(r'(?:으러|[가보사먹으]러|으래)\s+가', text) or any(t.tag == 'EC' and t.form in ('러','으러') for t in tokens):
        grammar.add('purpose')
    if re.search(r'주\s*(?:시겠|실래|실 수|세요|십시오|겠니|겠어|겠냐|시렵)', text):
        grammar.add('request')
    if re.search(r'(?:을래|[갈볼할실을릴줄낼올]래|까요|을까|갈까|볼까)(?:요)?\s*[?？.!]?$', text):
        if last_ending and ('래' in last_ending or '까' in last_ending):
            grammar.add('volition')
    if ('t동행' in content and ('volition' in grammar or 'v가' in content or 'v오' in content)):
        grammar.add('together')
    if '?' in text or '？' in text or '습니까' in last_ending or 'request' in grammar:
        grammar.add('question')
    return content, grammar


def index_path(base):
    return base.with_name('translation-context.sqlite3')


def signature(base):
    st = base.stat()
    return f'{VERSION}:{st.st_ino}:{st.st_mtime_ns}:{st.st_size}'


def build(base):
    path = index_path(base)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        version = signature(base)
        if path.exists():
            with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
                if db.execute('SELECT version FROM meta').fetchone() == (version,):
                    return
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temp = Path(file.name)
        try:
            with closing(sqlite3.connect(base.as_uri() + '?mode=ro', uri=True)) as corpus, closing(sqlite3.connect(temp)) as db:
                db.execute('CREATE VIRTUAL TABLE context USING fts5(content, grammar)')
                db.execute('CREATE TABLE meta(version TEXT)')
                rows = corpus.execute('SELECT id,ko FROM pair WHERE je_key != ko_key AND length(ko)<=100 ORDER BY id')
                while batch := rows.fetchmany(1000):
                    documents = []
                    for (rowid, text), tokens in zip(batch, analyzer().tokenize([r[1] for r in batch])):
                        content, grammar = features(text, tokens)
                        documents.append((rowid, ' '.join(sorted(content)), ' '.join(sorted(grammar))))
                    db.executemany('INSERT INTO context(rowid,content,grammar) VALUES(?,?,?)', documents)
                db.execute("INSERT INTO context(context) VALUES('optimize')")
                db.execute('INSERT INTO meta VALUES(?)', (version,))
                db.commit()
            if signature(base) != version:
                raise RuntimeError('번역 색인이 갱신됐습니다. 문맥 색인 생성을 다시 실행하세요.')
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def find_examples(db, base, text):
    """Search each clause, then compare its predicate, aspect, speech act and register."""
    db.execute('ATTACH DATABASE ? AS ctx', (index_path(base).as_uri() + '?mode=ro',))
    if db.execute('SELECT version FROM ctx.meta').fetchone() != (signature(base),):
        return []
    result = []
    parts = clauses(text)
    for part in (parts if len(parts) <= MAX_CLAUSES else parts[:4] + parts[-4:]):
        query, context = features(part, analyzer().tokenize(part))
        terms = {t for t in query if t[1:] not in LESS_SPECIFIC} or query
        if not terms:
            continue
        def disjunction(tokens):
            return '(' + ' OR '.join('"' + t.replace('"', '""') + '"' for t in sorted(tokens)) + ')'
        expression = 'content:' + disjunction(terms)
        queries = [expression]
        if context:
            queries.insert(0, expression + ' AND grammar:' + disjunction(context))
        verbs = {v for v in terms if v.startswith('v')}
        acts = context & {'request', 'reported', 'volition', 'purpose'}
        if acts and verbs:
            queries.insert(0, 'content:' + disjunction(verbs) + ' AND grammar:' + disjunction(acts))
        candidates = {}
        for search in queries:
            rows = db.execute('''SELECT pair.id,source,ko,je,context.content,context.grammar
                FROM ctx.context JOIN pair ON pair.id=context.rowid WHERE context MATCH ?
                ORDER BY bm25(context, 3, 1) LIMIT 60''', (search,))
            for rowid, source, ko, je, words, tags in rows:
                overlap = set(words.split()) & terms
                flags = set(tags.split())
                weight = sum(4 if t.startswith('v') else 2 for t in overlap)
                if verbs and not overlap & verbs:
                    weight -= 6
                conflicts = {'negative', 'past', 'ongoing', 'request', 'together', 'volition', 'purpose', 'reported'}
                weight += 2 * len(flags & context) - 3 * len((flags ^ context) & conflicts)
                weight += 4 * len(acts & flags)
                weight -= len(ko) // 25
                if {'plain', 'polite'} <= flags | context and not {'plain', 'polite'} & flags & context:
                    weight -= 3
                candidates[rowid] = (weight, len(overlap), -len(ko), source, ko, je)
        chosen = []
        ranked = sorted(candidates.values(), reverse=True)
        for score, _, _, source, ko, je in ranked:
            if score <= 0 or score < ranked[0][0] - 5:
                break
            if any(SequenceMatcher(None, ko, seen[1]).ratio() >= .83 for seen in chosen):
                continue
            chosen.append((source, ko, je))
            if len(chosen) == (4 if len(parts) == 1 else 2):
                break
        result.extend(chosen)
    return list(dict.fromkeys(result))


if __name__ == '__main__':
    from catalog import HERE
    build(HERE / 'runtime/translations.sqlite3')
