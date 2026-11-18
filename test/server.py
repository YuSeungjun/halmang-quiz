#!/usr/bin/env python3
"""Local listening/review workspace. No third-party dependencies or API keys."""
import argparse
import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from training_catalog import load_training_catalog
from quiz_generation import QuestionLab

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / 'catalog'
PUBLIC = ROOT / 'public'
STATE = ROOT / 'storage' / 'reviews.json'
LOCK = threading.Lock()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_state():
    if STATE.exists():
        return read_json(STATE)
    return {'selectedVoice': None, 'voices': {}, 'questions': {}}


class Handler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        super().end_headers()

    def json_response(self, body, status=200):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def valid_host(self):
        host = self.headers.get('Host', '').split(':')[0]
        allowed = {'127.0.0.1', 'localhost', self.server.server_address[0]}
        if host not in allowed:
            self.json_response({'error': 'localhost 포트 포워딩으로 접속해 주세요.'}, 403)
            return False
        return True

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.valid_host():
            return
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        if path == '/api/catalog':
            with LOCK:
                state = read_state()
            lab = self.server.lab
            with lab.lock:
                questions = lab.catalog_questions()
                latest_ids = [q['id'] for q in lab.reports[-1]['questions']] if lab.reports else []
            summary = {**self.server.catalog['summary'], 'questions': len(questions),
                       'generation': '생활방언·방언사전 근거로 작성 후 검증한 LLM 초안'}
            return self.json_response({**self.server.catalog, 'questions': questions, 'summary': summary, 'latestQuestions': latest_ids, 'state': state})
        if path == '/api/generation':
            return self.json_response(self.server.lab.status())
        if path.startswith('/api/generation/'):
            run_id = path.rsplit('/', 1)[-1]
            with self.server.lab.lock:
                report = next((r for r in self.server.lab.reports if r['id'] == run_id), None)
            return self.json_response(report or {'error': '실험 기록이 없습니다.'}, 200 if report else 404)
        if path == '/api/state':
            with LOCK:
                return self.json_response(read_state())
        if path == '/api/health':
            return self.json_response({'ok': True})
        if path == '/api/sources':
            query = parse_qs(parsed.query).get('q', [''])[0].strip()[:100]
            if not query:
                return self.json_response({'results': []})
            terms = query.split()
            results = []
            for kind in ('life', 'dictionary'):
                for item in self.server.sources[kind]:
                    text = item['siteName'] + ' ' + item['contents'] + ' ' + item.get('solution', '')
                    score = sum(text.count(t) for t in terms) + 10 * (item['siteName'] == query)
                    if score:
                        results.append({'kind': kind, **item, 'score': score})
            results.sort(key=lambda x: -x['score'])
            return self.json_response({'results': results[:12], 'total': len(results)})
        if path.startswith('/api/audio/'):
            media_id = path.removeprefix('/api/audio/')
            source = self.server.media.get(media_id)
            if source is None:
                return self.json_response({'error': '음성을 찾을 수 없습니다.'}, 404)
            return self.serve_file(ROOT.parent / source, audio=True)
        if path.startswith('/api/'):
            return self.json_response({'error': '존재하지 않는 API입니다.'}, 404)
        file = (PUBLIC / (path.lstrip('/') or 'index.html')).resolve()
        if not file.is_relative_to(PUBLIC.resolve()) or not file.is_file():
            return self.json_response({'error': '파일을 찾을 수 없습니다.'}, 404)
        self.serve_file(file)

    def serve_file(self, path, audio=False):
        if not path.is_file():
            return self.json_response({'error': '원본 파일이 없습니다. 데이터 경로를 확인해 주세요.'}, 404)
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get('Range') if audio else None
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            try:
                if not match or not any(match.groups()):
                    raise ValueError()
                a, b = match.groups()
                if a:
                    start = int(a)
                    end = min(int(b), size-1) if b else size-1
                else:
                    suffix = int(b)
                    if suffix <= 0:
                        raise ValueError()
                    start = max(0, size - suffix)
                if start > end or start >= size:
                    raise ValueError()
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        self.send_response(status)
        self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'private, max-age=3600' if audio else 'no-cache')
        if audio:
            self.send_header('Accept-Ranges', 'bytes')
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD':
            return
        try:
            with path.open('rb') as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser may cancel a playing clip or a range request.

    def do_POST(self):
        if not self.valid_host():
            return
        if self.path not in ('/api/state', '/api/generation'):
            return self.json_response({'error': '존재하지 않는 API입니다.'}, 404)
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{self.headers.get("Host")}':
            return self.json_response({'error': '다른 출처의 저장 요청은 허용하지 않습니다.'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 20000:
                raise ValueError('저장 요청 크기가 올바르지 않습니다.')
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('JSON 형식이 필요합니다.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or len(payload) != 1:
                raise ValueError('저장할 항목을 하나씩 보내 주세요.')
            field = next(iter(payload))
            value = payload[field]
            if self.path == '/api/generation':
                if field != 'topics':
                    raise ValueError('주제를 입력해 주세요.')
                try:
                    return self.json_response(self.server.lab.start(value), 202)
                except RuntimeError as error:
                    return self.json_response({'error': str(error)}, 409)
            with LOCK:
                state = read_state()
                if field == 'selectedVoice':
                    if value is not None and value not in self.server.voice_ids:
                        raise ValueError('올바르지 않은 목소리입니다.')
                    state[field] = value
                elif field in ('voice', 'question'):
                    valid_ids = self.server.voice_ids if field == 'voice' else self.server.question_ids
                    if field == 'question':
                        with self.server.lab.lock:
                            valid_ids = valid_ids | {q['id'] for q in self.server.lab.drafts()}
                    if not isinstance(value, dict) or value.get('id') not in valid_ids:
                        raise ValueError('대상 항목이 없습니다.')
                    allowed = {'id', 'note', 'favorite'} if field == 'voice' else {'id', 'note', 'status'}
                    if set(value) - allowed:
                        raise ValueError('알 수 없는 항목입니다.')
                    if 'note' in value and (not isinstance(value['note'], str) or len(value['note']) > 3000):
                        raise ValueError('메모는 3,000자 이내로 입력해 주세요.')
                    if 'favorite' in value and not isinstance(value['favorite'], bool):
                        raise ValueError('즐겨찾기 값이 올바르지 않습니다.')
                    if 'status' in value and value['status'] not in ('draft', 'approved', 'rejected'):
                        raise ValueError('검토 상태가 올바르지 않습니다.')
                    collection = 'voices' if field == 'voice' else 'questions'
                    state[collection].setdefault(value['id'], {}).update({k: v for k, v in value.items() if k != 'id'})
                else:
                    raise ValueError('저장할 항목이 없습니다.')
                STATE.parent.mkdir(exist_ok=True)
                temporary = STATE.with_suffix('.tmp')
                temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
                os.replace(temporary, STATE)
            self.json_response(state)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self.json_response({'error': str(e) or '잘못된 요청입니다.'}, 400)
        except OSError:
            self.json_response({'error': '저장하지 못했습니다. 서버의 저장 공간과 권한을 확인해 주세요.'}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    if not (CATALOG / 'questions.json').exists():
        raise SystemExit('먼저 python3 test/build_catalog.py 를 실행해 주세요.')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.catalog = {k: read_json(CATALOG / f'{k}.json') for k in ('voices', 'questions', 'summary')}
    server.sources = read_json(CATALOG / 'sources.json')
    server.lab = QuestionLab(server.sources, server.catalog['questions'])
    server.media = read_json(CATALOG / 'media.json')
    for row in server.sources['life']:
        seq = int(row['seq'])
        audio_path = f'data/audio/life/life-{seq}.mp3'
        if (ROOT.parent / audio_path).is_file():
            server.media[f'life-{seq}'] = audio_path
    try:
        training, training_media = load_training_catalog()
        server.catalog['training'] = training
        server.media.update(training_media)
    except (OSError, ValueError, KeyError, EOFError) as error:
        print(f'학습 음성 카탈로그를 읽지 못했습니다: {error}', flush=True)
        server.catalog['training'] = {'voiceId': 'voice-03', 'voiceName': '목소리 03', 'items': [], 'models': []}
    server.voice_ids = {v['id'] for v in server.catalog['voices']}
    server.question_ids = {q['id'] for q in server.catalog['questions']}
    print(f'소랑 연구실 · http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
