"""Regression cases for dictionary routing and the generated-sentence review."""
import unittest
from unittest.mock import patch

import translation as tr


class TranslationTest(unittest.TestCase):
    def test_unpunctuated_conversation_keeps_requests_and_invitations_separate(self):
        clauses = ['사랑해요', '저는 지금부터 당신에게 고백하려고 합니다', '제 고백을 받아주시겠어요',
                   '고백을 받고 고기국수라도 먹으러 가실래요?']
        guides = tr.clause_guides(' '.join(clauses))
        self.assertEqual([g['원문'] for g in guides], clauses)
        opening = '소랑합니다. 지금부터 고백허젠 헴수다. '
        request = '내 고백 받아줄 거꽈? '
        invitation = '받고 나민 고기국수라도 먹으레 갈 거꽈?'
        self.assertEqual(tr.check_clauses(guides, opening + request + invitation), '')
        self.assertIn('부탁', tr.check_clauses(guides, opening + '내 고백을 받아카마씨? ' + invitation))
        self.assertIn('이동', tr.check_clauses(guides, opening + request + '고기국수라도 먹으카마씨?'))
        self.assertIn('낮춰', tr.check_clauses(guides, opening.replace('지금부터', '지금부터 니당게') + request + invitation))

    def test_guides_use_the_action_from_each_request(self):
        guides = tr.clause_guides('제 사진을 봐주시겠어요?같이 영화 보러 갈래요?')
        self.assertEqual(len(guides), 2)
        self.assertIn('봐 주시쿠과?', guides[0]['기능'])
        self.assertIn('이동', tr.check_clauses(guides, '제 사진 봐 주시쿠과? 고치 영화 보카마씨?'))
        self.assertEqual(tr.check_clauses(guides, '제 사진 봐 주시쿠과? 고치 영화 보레 가카마씨?'), '')
        self.assertFalse(tr.clause_guides('동생이 “같이 영화 보러 갈래요?” 하고 물었어요.'))
        self.assertFalse(tr.clause_guides('동문시장에서 만나서 고기국수라도 먹을래요?'))

    def test_pronouns_are_restored_only_for_the_matching_noun_in_its_clause(self):
        guides = tr.clause_guides('지금부터 당신에게 부탁하려고 합니다. 제 사진을 봐주시겠어요?')
        self.assertEqual(tr.align_pronouns(guides, '지금부터 당신게 부탁허젠 헴수다. 저게 사진 봐 주시쿠과?'),
                         '지금부터 당신에게 부탁허젠 헴수다. 제 사진 봐 주시쿠과?')
        omitted = '지금부터 부탁허젠 헴수다. 사진 봐 주시쿠과?'
        self.assertEqual(tr.align_pronouns(guides, omitted), omitted)
        self.assertEqual(tr.align_pronouns(guides, '저게 뭐우꽈?'), '저게 뭐우꽈?')
        direct = tr.clause_guides('저는 지금부터 당신에게 부탁하려고 합니다. 제 사진을 봐주시겠어요?')
        self.assertEqual(tr.align_pronouns(direct, '지금부터 저가 당신게 부탁허젠 헴수다. 사진 봐 주시쿠과?'), omitted)

    def test_contradictory_review_triggers_a_retry(self):
        context = {'입력 문장': '도와주세요.', '어휘 풀이': {}, '참고 용례': []}
        report = {key: True for key in tr.CHECKS}
        report['feedback'] = '청자에게 부탁하는 어미로 바꾸세요.'
        with patch.object(tr, 'ollama_json', return_value=report):
            self.assertEqual(tr.check_sentence(context, '도와줍서.', 'ko-je'), report['feedback'])

    def test_introduction_and_invitation_need_grounded_korean_endings(self):
        original = '안녕하세요 저는 한지민입니다. 시장에서 만나서 점심 먹을래요?'
        notes = tr.conversation_patterns(original)
        context = {'입력 문장': original, '회화 문형': notes, '참고 용례': [], '어휘 풀이': {}}
        self.assertEqual(len(notes), 3)
        with patch.object(tr, 'ollama_json', side_effect=AssertionError):
            self.assertIn('말조각', tr.check_sentence(context, '안녕하세요 저 죠 한지민이우다. 점심 먹을래요?', 'ko-je'))
            self.assertIn('어미', tr.check_sentence(context, '안녕허우꽈? 한지민이우다. 점심 먹을래요?', 'ko-je'))
        self.assertFalse(tr.conversation_patterns('그는 “밥 먹을래요?”라고 물었다.'))
        self.assertFalse(tr.conversation_patterns('저는 오늘 집에서 먹을래요.'))

    def test_menu_name_and_offered_alternative_survive_review(self):
        original = '남문시장에서 비빔국수라도 먹을래요?'
        anchors = [place or dish for place, dish in tr.OUTING_NAMES.findall(original)]
        self.assertEqual(anchors, ['남문시장', '비빔국수라도'])
        context = {'입력 문장': original, '회화 문형': tr.conversation_patterns(original), '보존할 명칭': anchors}
        with patch.object(tr, 'ollama_json', side_effect=AssertionError):
            self.assertIn('비빔국수라도', tr.check_sentence(context, '남문시장에서 비빔국수 먹으카마씨?', 'ko-je'))

    def test_greeting_may_be_a_question_before_a_declaration(self):
        original = '안녕하세요. 저는 김민준입니다.'
        context = {'입력 문장': original, '회화 문형': tr.conversation_patterns(original), '참고 용례': [], '어휘 풀이': {}}
        answers = [{'translation': '안녕허우꽈. 김민준이우다.'}, {key: True for key in tr.CHECKS}]
        with patch.object(tr, 'ollama_json', side_effect=answers):
            result = tr.checked_draft('번역', context, 'ko-je', lambda *_: None)
        self.assertEqual(result, '안녕허우꽈? 김민준이우다.')

    def test_dictionary_works_without_database_or_model(self):
        with patch.object(tr, 'INDEX', None), patch.object(tr, 'ollama_json', side_effect=AssertionError):
            for word in ('하르방', '하르버지', '하라바지', '하르바지'):
                self.assertEqual(tr.translate(word, 'je-ko')['text'], '할아버지')
            result = tr.translate('할아버지.', 'ko-je')
            self.assertEqual((result['kind'], result['text']), ('word', '하르방'))
            self.assertIn('하르버지', result['evidence'][0]['variants'])
            self.assertEqual(tr.translate('어서 오십시오.', 'ko-je')['kind'], 'corpus')

    def test_retry_corrects_question_ending_and_preserves_progress(self):
        context = {'입력 문장': '할아버지가 집에 오셨어요.', '어휘 풀이': {}, '참고 용례': []}
        answers = [{'translation': '하르방이 집에 와수과.'}, {'translation': '하르방이 집에 와수다.'},
                   {key: True for key in tr.CHECKS}]
        with patch.object(tr, 'ollama_json', side_effect=answers) as model:
            events = []
            result = tr.checked_draft('번역', context, 'ko-je', lambda n, _: events.append(n))
            self.assertEqual(result, '하르방이 집에 와수다.')
            self.assertEqual(events, sorted(events))
            self.assertIn('고칠 점', ' '.join(model.call_args_list[1].args[1]))

    def test_repeated_failure_withholds_draft(self):
        context = {'입력 문장': '할아버지가 왔다.', '어휘 풀이': {}, '참고 용례': []}
        failed = {key: key != 'meaning' for key in tr.CHECKS}
        answers = [{'translation': '할망이 왓수다.'}, failed] * 2
        with patch.object(tr, 'ollama_json', side_effect=answers) as model:
            self.assertIsNone(tr.checked_draft('번역', context, 'ko-je', lambda *_: None))
            self.assertEqual(model.call_count, 4)


if __name__ == '__main__':
    unittest.main()
