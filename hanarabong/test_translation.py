"""Regression cases for dictionary routing and the generated-sentence review."""
import unittest
from unittest.mock import patch

import translation as tr


class TranslationTest(unittest.TestCase):
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
