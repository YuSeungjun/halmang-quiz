"""Learner site with local translation and on-demand sentence audio."""
import argparse
import json
import re
import threading
import traceback
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from catalog import AUDIO, HERE, PROJECT, eligible_questions, learner_question
from server import Handler as ResearchHandler
from speech import sentence_audio
from translation import LIMIT, clean, translate

PUBLIC = HERE / 'public'
FONTS = PROJECT / 'test/public/fonts'
SLOTS = {'/api/translate': threading.BoundedSemaphore(2),
         '/api/speech': threading.BoundedSemaphore(1)}


class LearnerHandler(ResearchHandler):
    def stream_translation(self, text, direction):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-store, no-transform')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True

        def event(payload):
            self.wfile.write(('data: ' + json.dumps(payload, ensure_ascii=False) + '\n\n').encode())
            self.wfile.flush()

        try:
            result = translate(text, direction, lambda value, stage: event({'progress': value, 'stage': stage}))
            event({'result': result})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            event({'error': '번역을 완료하지 못했어요. 잠시 후 다시 시도해 주세요.'})

    def send_header(self, keyword, value):
        # Generated WAVs are received by POST and played from a browser-local Blob URL.
        if keyword == 'Content-Security-Policy':
            value = value.replace("media-src 'self';", "media-src 'self' blob:;")
        super().send_header(keyword, value)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/questions':
            rows = [learner_question(q) for q in eligible_questions()]
            return self.json_response([q for q in rows if (AUDIO / Path(q['audio']).name).is_file()])
        if re.fullmatch(r'/audio/[0-9a-f]{24}\.wav', path):
            return self.serve_file(AUDIO / Path(path).name, audio=True)
        if path in ('/fonts/Pretendard-Regular.woff2', '/fonts/Pretendard-SemiBold.woff2', '/fonts/LICENSE.txt'):
            return self.serve_file(FONTS / Path(path).name)
        file = (PUBLIC / (path.lstrip('/') or 'index.html')).resolve()
        if not file.is_relative_to(PUBLIC.resolve()) or not file.is_file():
            return self.json_response({'error': '파일을 찾을 수 없어요.'}, 404)
        return self.serve_file(file)

    def do_POST(self):
        slot = SLOTS.get(self.path)
        if slot is None:
            return self.json_response({'error': '요청 경로를 찾을 수 없어요.'}, 404)
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.json_response({'error': '하나라봉 페이지에서 다시 시도해 주세요.'}, 403)
        if self.headers.get_content_type() != 'application/json':
            return self.json_response({'error': 'JSON 요청이 필요해요.'}, 415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192:
                raise ValueError('입력 크기를 확인해 주세요.')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or not isinstance(data.get('text'), str):
                raise ValueError('문장을 입력해 주세요.')
            text = clean(data['text'])
            if not 1 <= len(text) <= LIMIT or not any(char.isalnum() for char in text):
                raise ValueError(f'글자나 숫자가 있는 문장을 {LIMIT}자 이내로 입력해 주세요.')
            if self.path == '/api/translate' and data.get('direction') not in ('je-ko', 'ko-je'):
                raise ValueError('번역 방향을 선택해 주세요.')
        except (ValueError, UnicodeError) as error:
            return self.json_response({'error': str(error)}, 400)
        if not slot.acquire(blocking=False):
            return self.json_response({'error': '모델이 다른 문장을 처리하고 있어요. 잠시 후 다시 눌러 주세요.'}, 429)
        try:
            if self.path == '/api/translate':
                if 'text/event-stream' in self.headers.get('Accept', ''):
                    return self.stream_translation(text, data['direction'])
                return self.json_response(translate(text, data['direction']))
            wav = sentence_audio(text)
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Content-Length', str(len(wav)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(wav)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            action = '번역' if self.path == '/api/translate' else '음성 합성'
            self.json_response({'error': f'{action}을 완료하지 못했어요. 잠시 후 다시 시도해 주세요.'}, 503)
        finally:
            slot.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    with ThreadingHTTPServer((args.host, args.port), LearnerHandler) as server:
        print(f'하나라봉 · http://{args.host}:{args.port}', flush=True)
        server.serve_forever()
