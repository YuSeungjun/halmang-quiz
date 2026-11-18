"""Package the current read-only catalog, page and audio for opening as a local file."""
import base64
import json
import mimetypes
import re
from pathlib import Path

from catalog import AUDIO, HERE, PROJECT, eligible_questions, learner_question

PUBLIC = HERE / 'public'
OUTPUT = HERE / '하나라봉.html'


def embed(path):
    return [mimetypes.guess_type(path.name)[0] or 'application/octet-stream',
            base64.b64encode(path.read_bytes()).decode('ascii')]


def main():
    questions = [learner_question(q) for q in eligible_questions()]
    files = {'fruit.svg': PUBLIC / 'fruit.svg'}
    files.update({f'assets/{p.name}': p for p in (PUBLIC / 'assets').iterdir() if p.suffix != '.txt'})
    files.pop('assets/halmang_happy.webp', None)
    fonts = PROJECT / 'test/public/fonts'
    files.update({f'/fonts/{p.name}': p for p in fonts.glob('*.woff2')})
    files.update({q['audio']: AUDIO / Path(q['audio']).name for q in questions})

    html = (PUBLIC / 'index.html').read_text()
    html = re.sub(r'<link rel="preload"[^>]+>\n', '', html)
    html = html.replace('<script src="app.js" defer></script>', '')
    html = html.replace('href="fruit.svg"', 'data-file-href="fruit.svg"')
    html = re.sub(r'src="(fruit\.svg|assets/[^"]+)"', r'data-file-src="\1"', html)
    html = re.sub(r'srcset="(assets/[^"]+)"', r'data-file-srcset="\1"', html)
    bundle = {'questions': questions, 'files': {url: embed(path) for url, path in files.items()},
              'css': (PUBLIC / 'style.css').read_text()}
    data = json.dumps(bundle, ensure_ascii=False, separators=(',', ':')).replace('<', r'\u003c')
    loader = """<script>
const packed = JSON.parse(document.getElementById('book-data').textContent);
const media = {};
for (const [path, [type, bytes]] of Object.entries(packed.files)) {
  const data = Uint8Array.from(atob(bytes), c => c.charCodeAt(0));
  media[path] = URL.createObjectURL(new Blob([data], {type}));
}
window.hanarabongPocket = {questions: packed.questions, media};
const styles = document.createElement('style');
styles.textContent = packed.css.replace(/url\\('([^']+)'\\)/g, (_, path) => 'url('+media[path]+')');
document.head.append(styles);
document.getElementById('book-data').remove();
delete packed.files;
</script>"""
    html = html.replace('<link rel="stylesheet" href="style.css">',
                        '<script type="application/json" id="book-data">' + data + '</script>' + loader)
    boot = """document.querySelectorAll('[data-file-srcset]').forEach(el => { el.srcset = media[el.dataset.fileSrcset]; });
document.querySelectorAll('[data-file-src]').forEach(el => { el.src = media[el.dataset.fileSrc]; });
document.querySelectorAll('[data-file-href]').forEach(el => { el.href = media[el.dataset.fileHref]; });"""
    licenses = '\n\n'.join(p.read_text() for p in [fonts / 'LICENSE.txt', *sorted((PUBLIC / 'assets').glob('*LICENSE.txt'))])
    credits = '낭독: VoxCPM2, AI Hub 71558 + Kakao Brain JSS, 목소리 03. 문장: 제주특별자치도 생활방언.'
    html = html.replace('</body>', f'<script type="text/plain" id="credits">{credits}\n{licenses}</script>\n'
                        '<script>' + boot + '\n' + (PUBLIC / 'app.js').read_text() + '</script></body>')
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(html, encoding='utf-8')
    temporary.replace(OUTPUT)
    print(OUTPUT.name, f'{OUTPUT.stat().st_size / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
