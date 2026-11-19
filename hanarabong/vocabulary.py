"""Bidirectional learner vocabulary with a source for each sense."""
import re
from collections import defaultdict
from functools import lru_cache

from catalog import CATALOG, PROJECT, read_json

# These teaching equivalents occur in the numbered 생활방언 dialogues.
# The quiz seeds provide the same fields for the rest of the core vocabulary.
EXTRA_WORDS = [('하르방', '할아버지', 67), ('할망', '할머니', 42),
               ('손지', '손자', 7), ('비바리', '처녀', 32)]
FAMILY_ENTRY = 'https://www.jeju.go.kr/jejueo/dictionary/search.htm?id=1-24344-0'
PARTICLE = re.compile(r'(?:들)?(?:에게서|께서는|으로는|에서는|에게|에서|한테|으로|이랑|이영|하고|께서|안티|까지|부터|처럼|드렌|이|가|은|는|을|를|에|의|와|과|도|만|로|랑|영|디|께)$')


@lru_cache(maxsize=1)
def vocabulary():
    words = []
    claimed = {'ko': set(), 'je': set()}
    seeds = [(row[3], row[4], row[0]) for row in read_json(PROJECT / 'test/question_seeds.json')]
    for je, ko, seq in seeds + EXTRA_WORDS:
        if je in claimed['je'] or not re.fullmatch(r'[가-힣]+', ko):
            continue
        entry = {'je': je, 'ko': ko, 'source': '제주특별자치도 생활방언',
                 'url': f'https://www.jeju.go.kr/culture/dialect/lifeDialect.htm?act=view&seq={seq}'}
        if je == '하르방':
            entry.update(source='제주어대사전', url=FAMILY_ENTRY, variants=['하르버지', '하르바지', '하라바지'])
        words.append(entry)
        claimed['ko'].add(ko)
        claimed['je'].update([je, *entry.get('variants', [])])

    # A one-word gloss supports direct lookup only when both sides are unambiguous
    # in this index. Longer definitions and cross-references need context.
    senses = defaultdict(set)
    equivalents = defaultdict(set)
    rows = read_json(CATALOG / 'sources.json')['dictionary']
    for row in rows:
        je, ko = row['siteName'], row['contents'].strip().rstrip('.')
        senses[je].add(ko)
        equivalents[ko].add(je)
    for row in rows:
        je, ko = row['siteName'], row['contents'].strip().rstrip('.')
        if (len(senses[je]) != 1 or len(equivalents[ko]) != 1 or je == ko
                or not re.fullmatch('[가-힣ㆍᄀ-ᇿ]+', je) or not re.fullmatch('[가-힣]+', ko)
                or je in claimed['je'] or ko in claimed['ko']):
            continue
        words.append({'je': je, 'ko': ko, 'source': '제주특별자치도 방언사전',
                      'url': 'https://www.jeju.go.kr/culture/dialect/dictionary.htm?act=view&seq=' + str(row['seq'])})
        claimed['je'].add(je)
        claimed['ko'].add(ko)
    return words


@lru_cache(maxsize=2)
def word_index(language):
    index = {}
    for entry in vocabulary():
        index[entry[language]] = entry
        if language == 'je':
            for name in entry.get('variants', []):
                index[name] = entry
    return index


def sentence_terms(text, language):
    index = word_index(language)
    matches = {}
    for token in re.findall(r'[^\W_]+', text):
        entry = index.get(token) or index.get(PARTICLE.sub('', token))
        if entry:
            matches[entry['je']] = entry
    return matches
