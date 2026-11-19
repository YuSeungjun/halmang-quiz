"""Run against the started learner site with a Python environment that has Playwright."""
import argparse
import json
import tempfile
import wave
from pathlib import Path
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

from catalog import AUDIO, eligible_questions, learner_question

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8766')
parser.add_argument('--screenshots', type=Path, default=Path(tempfile.gettempdir()) / 'hanarabong-check')
args = parser.parse_args()
args.screenshots.mkdir(parents=True, exist_ok=True)
rows = json.load(urlopen(args.url + '/api/questions'))
catalog = {q['id']: q for q in rows}
assert rows == [learner_question(q) for q in eligible_questions()]
assert len({q['answer'] for q in rows}) >= 10
for q in rows:
    assert set(q['optionMeanings']) == set(q['options']) and all(q['optionMeanings'].values())
    assert q['optionMeanings'][q['answer']] == q['meaning']
for name in {q['audio'] for q in rows}:
    with wave.open(str(AUDIO / Path(name).name), 'rb') as wav:
        assert wav.getnframes() > wav.getframerate() / 4
    with urlopen(Request(args.url + name, headers={'Range':'bytes=0-43'})) as response:
        assert response.status == 206 and response.read()[:4] == b'RIFF'


def playing(page, q):
    page.wait_for_function("""url => {
        const a = document.querySelector('audio');
        return a.src.endsWith(url) && a.currentTime > 0 && a.duration > 0;
    }""", arg=q['audio'])


def positions(page):
    return page.locator('.option, #prompt').evaluate_all(
        '(els) => els.map(e => [e.getBoundingClientRect().x, e.getBoundingClientRect().y])')


def check_round(browser, width, height, reduced=False, perfect=False):
    mode = 'reduce' if reduced else 'no-preference'
    context = browser.new_context(viewport={'width':width,'height':height}, reduced_motion=mode)
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('response', lambda r: errors.append(f'{r.status} {r.url}') if r.status >= 400 else None)
    page.goto(args.url)
    page.evaluate('document.fonts.ready')
    start = page.get_by_role('button', name='10문제 시작하기')
    assert page.get_by_role('heading',name='혼저 옵서.').is_visible()
    assert page.locator('.welcome-halmang').evaluate('async e => { await e.decode(); return e.currentSrc; }').endswith(
        'happy_still.png' if reduced else 'happy.gif')
    assert start.bounding_box()['y'] + start.bounding_box()['height'] <= height
    page.screenshot(path=args.screenshots / f'home-{width}.png', full_page=True)
    assert page.locator('.welcome-copy .translation').inner_text() == '어서 오십시오.'
    assert page.locator('.welcome-copy').get_by_role('button').count() == 1
    assert page.locator('audio').evaluate('a => a.paused && !a.hasAttribute("src")')
    page.get_by_role('button', name='놀이 방법').click()
    assert page.get_by_role('dialog').is_visible()
    page.keyboard.press('Escape')
    for card in page.locator('[data-word]').all():
        q = catalog[card.get_attribute('data-word')]
        card.click()
        assert page.locator('dialog .sample').inner_text() == q['sentence']
        page.locator('dialog [data-listen]').click()
        playing(page, q)
        page.get_by_role('button',name='닫기', exact=True).click()
    start.click()
    missed, words = [], set()
    for i in range(10):
        page.wait_for_selector('.question-sheet')
        q = catalog[page.locator('.question-sheet').get_attribute('data-question')]
        assert q['answer'] not in words and set(q['options']) == set(page.locator('.option-word').all_inner_texts())
        words.add(q['answer'])
        assert page.locator('#prompt').inner_text() == q['prompt'].replace('____','?')
        assert page.locator('.standard-line').inner_text().endswith(q['standard'])
        assert page.locator('.page-count b').inner_text() == str(i + 1)
        assert page.locator('.portrait img').evaluate('async e => {await e.decode(); return e.currentSrc}').endswith('halmang_idle.png')
        if i in (0,1):
            page.get_by_role('button',name='할망의 힌트').click()
            assert page.locator('#hint-text').inner_text() == q['hint']
            src = page.locator('.portrait img').evaluate('async el => { await el.decode(); return el.currentSrc; }')
            assert src.endswith('hint_still.png' if reduced else 'hint.webp')
            if i == 1:
                page.get_by_role('button',name='할망의 힌트').click()
                assert not page.locator('#hint-text').is_visible()
        right = perfect or i % 3 != 1
        choice = q['answer'] if right else next(w for w in q['options'] if w != q['answer'])
        choice_button = page.locator('.option').filter(has=page.get_by_text(choice,exact=True))
        choice_button.scroll_into_view_if_needed()
        before = positions(page)
        choice_button.click()
        assert before == positions(page), (width, q['id'], before, positions(page))
        assert page.locator('#prompt').inner_text() == q['sentence']
        mood = 'happy' if right else 'sad'
        assert page.locator('.portrait img').evaluate('async el => { await el.decode(); return el.currentSrc; }').endswith(
            mood + ('_still.png' if reduced else '.gif' if right else '.webp'))
        assert page.locator('.correct .option-state').inner_text() == '정답'
        if not right:
            missed.append(q)
            assert page.locator('.wrong .option-state').inner_text() == '내가 고른 답'
            assert page.locator('.wrong .option-word').inner_text() == choice
            assert page.locator('.answer-meanings dd').all_inner_texts() == [
                f"{choice} ({q['optionMeanings'][choice]})", f"{q['answer']} ({q['meaning']})"]
            assert page.locator('audio').evaluate('a => a.paused')
            page.locator('.actions [data-listen]').click()
        playing(page, q)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        if i == 1: page.screenshot(path=args.screenshots / f'answer-{width}.png',full_page=True)
        page.locator('[data-action=next]').click()
    assert page.locator('.score').inner_text().startswith(str(10-len(missed)))
    assert page.locator('.review-item h3').all_inner_texts() == [q['sentence'] for q in missed]
    assert page.locator('.review-item p').all_inner_texts() == [q['standard'] for q in missed]
    if missed:
        page.locator('.review-item [data-listen]').first.click()
        playing(page, missed[0])
    else:
        assert page.locator('.perfect').is_visible()
    page.screenshot(path=args.screenshots / f'results-{width}.png',full_page=True)
    page.get_by_role('button',name='다시 풀기').click()
    assert page.locator('.page-count b').inner_text() == '1'
    page.locator('#sound-toggle').click()
    q = catalog[page.locator('.question-sheet').get_attribute('data-question')]
    page.locator('.option').filter(has=page.get_by_text(q['answer'],exact=True)).click()
    assert page.locator('audio').evaluate('a => a.paused && !a.hasAttribute("src")')
    page.locator('#home-link').click()
    page.get_by_role('button',name='첫 장으로',exact=True).click()
    assert start.is_visible() and not errors, errors
    context.close()
    print(f'{width}×{height}, motion={mode}, score={10-len(missed)}/10: passed')


def check_timer(browser):
    page = browser.new_page(viewport={'width':390,'height':844})
    page.clock.install()
    page.add_init_script("""window.playedNotes = [];
        const start = OscillatorNode.prototype.start;
        OscillatorNode.prototype.start = function(...args) {
            playedNotes.push(this.frequency.value);
            return start.apply(this, args);
        };""")
    page.goto(args.url)
    page.locator('.start').click()
    page.wait_for_selector('.option')
    label = page.get_by_role('timer')
    assert label.inner_text() == '30초'
    page.clock.fast_forward(10_000)
    assert label.inner_text() == '20초'
    scale = page.locator('.timer-fill').evaluate("el => parseFloat(el.style.transform.slice('scaleX('.length))")
    assert .62 < scale < .69
    page.get_by_role('button', name='놀이 방법').click()
    paused = label.inner_text()
    page.clock.fast_forward(40_000)
    assert label.inner_text() == paused and not page.locator('.feedback').count()
    page.keyboard.press('Escape')
    page.locator('dialog').wait_for(state='hidden')
    page.clock.run_for(120)
    page.clock.fast_forward(15_000)
    assert label.inner_text() == '5초'
    warning = page.evaluate('playedNotes.slice()')
    assert len(warning) == 1
    page.get_by_role('button', name='놀이 방법').click()
    page.clock.fast_forward(3000)
    assert label.inner_text() == '5초' and page.evaluate('playedNotes') == warning
    page.keyboard.press('Escape')
    page.wait_for_function('() => timer?.deadline != null && !document.querySelector("dialog").open')
    for seconds in (4, 3, 2, 1):
        page.clock.fast_forward(1000)
        assert label.inner_text() == f'{seconds}초'
        notes = page.evaluate('playedNotes.slice()')
        count = 2 if seconds <= 2 else 1
        assert len(notes) == len(warning) + count
        assert all(note > warning[0] if seconds <= 2 else note == warning[0] for note in notes[-count:])
        warning = notes
    page.clock.fast_forward(1050)
    assert label.inner_text() == '0초'
    assert page.locator('.feedback-title').inner_text() == '시간이 다 됐어요.'
    assert page.locator('.correct').count() == 1 and not page.locator('.wrong').count()
    wrong = page.evaluate('playedNotes.slice()')[len(warning):]
    assert wrong and wrong != warning
    expired = page.locator('.question-sheet').get_attribute('data-question')
    page.locator('.option').first.evaluate('el => el.click()')
    assert page.locator('.feedback-title').inner_text() == '시간이 다 됐어요.'
    for i in range(1,10):
        page.locator('[data-action=next]').click()
        assert label.inner_text() == '30초'
        if i == 1:
            page.clock.fast_forward(28_050)
            assert label.inner_text() == '2초'
            before_answer = page.evaluate('playedNotes.slice()')
        elif i == 2:
            page.locator('#sound-toggle').click()
            page.clock.fast_forward(25_050)
            page.clock.run_for(4000)
        q = catalog[page.locator('.question-sheet').get_attribute('data-question')]
        page.locator('.option').filter(has=page.get_by_text(q['answer'],exact=True)).click()
        frozen = label.inner_text()
        page.clock.fast_forward(31_000)
        assert label.inner_text() == frozen
        if i == 1:
            correct = page.evaluate('playedNotes.slice()')[len(before_answer):]
            assert correct == [784, 1046.5]
    assert page.evaluate('playedNotes') == before_answer + correct
    page.locator('[data-action=next]').click()
    assert page.locator('.score').inner_text().startswith('9')
    assert page.locator('.timeout-tag').inner_text() == '시간 초과'
    assert page.locator('.review-item h3').all_inner_texts() == [catalog[expired]['sentence']]
    page.close()
    print('Countdown, all five warning ticks, final double beeps, pause, mute and answer cutoff: passed')


def check_translation(browser, width):
    page = browser.new_page(viewport={'width':width,'height':844})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(args.url + '/#translate')
    cards = page.locator('.suggestion')
    submit = page.locator('.translation-submit')
    source = page.locator('#source-example')
    result = page.locator('#target-example')
    listen = page.get_by_role('button', name='제주어 듣기', exact=True)
    reset = page.get_by_role('button', name='초기화', exact=True)
    assert cards.count() == 3 and submit.is_enabled()
    assert source.input_value() == '어서 오십시오.' and result.inner_text() == '혼저 옵서.'
    assert page.locator('#source-language').inner_text() == '표준어'
    assert page.locator('.sentence-listen').count() == 1
    source.hover()
    assert not source.input_value() and not result.inner_text() and reset.is_disabled()
    pairs = {q['standard']: q['sentence'] for q in rows}
    for card in cards.all():
        card.click()
        sentence = page.locator('#source-example').input_value()
        source.click()
        assert source.input_value() == sentence
        assert page.locator('#target-example').inner_text() == pairs[sentence]
        assert page.locator('.target-page #jeju-listen').is_visible()
        assert submit.inner_text() == '번역하기'
        page.locator('#swap-languages').click()
        assert page.locator('#source-example').input_value() == pairs[sentence]
        assert page.locator('#target-example').inner_text() == sentence
        assert page.locator('.source-page #jeju-listen').is_visible()
        assert submit.inner_text() == '번역하기'
        page.locator('#swap-languages').click()
    reset.click()
    assert not source.input_value() and reset.is_disabled() and source.evaluate('e => document.activeElement === e')
    assert submit.is_disabled() and listen.is_disabled() and not result.inner_text()
    check_word_lookup(page)
    for original, translated, corpus in [
            ('학교 동산에 동백꽃 많이 피었더라', '학교 동산에 동백고장 하영 피어서라', 'AI Hub'),
            ('건 어떵 허는 거꽈?', '그건 어떻게 하는 겁니까?', 'JIT')]:
        source.fill(original)
        with page.expect_response('**/api/translate') as response:
            submit.click()
        assert response.value.headers['content-type'].startswith('text/event-stream')
        page.wait_for_function("s => document.querySelector('#target-example').textContent === s", arg=translated)
        assert corpus in page.locator('#translation-status').inner_text()
        page.locator('#swap-languages').click()
        assert source.input_value() == translated and result.inner_text() == original
    novel = '할망이영 바당에 강 저녁 먹을 거우다.' if width == 390 else '내일은 어머니와 함께 바다에 가고 싶어요.'
    if width == 390:
        page.locator('#swap-languages').click()
    source.fill(novel)
    page.evaluate(r"""() => {
      window.readings = [];
      const label = document.querySelector('.submit-percent');
      new MutationObserver(records => {
        for (const record of records) for (const node of record.addedNodes) {
          const number = node.textContent.match(/(\d+)%/);
          if (number) readings.push(Number(number[1]));
        }
      }).observe(label, {childList:true});
    }""")
    with page.expect_response('**/api/translate', timeout=210_000) as response:
        submit.click()
    page.wait_for_function("() => document.querySelector('#target-example').textContent.length > 0", timeout=210_000)
    readings = page.evaluate('readings')
    assert readings == sorted(readings) and readings[0] == 0 and readings[-1] == 100 and any(40 < r < 96 for r in readings), readings
    assert 'AI 번역' in page.locator('#translation-status').inner_text()
    with page.expect_response('**/api/speech', timeout=180_000) as response:
        listen.click()
    expected_jeju = novel if width == 390 else result.inner_text()
    assert response.value.request.post_data_json == {'text':expected_jeju}
    assert response.value.ok and response.value.headers['content-type'] == 'audio/wav'
    page.wait_for_function("() => {const a=document.querySelector('audio');return a.src.startsWith('blob:') && a.currentTime>0 && a.duration>0}")
    assert page.get_by_role('button', name='표준어 듣기').count() == 0
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=args.screenshots / f'translation-{width}.png', full_page=True)
    language = page.locator('#source-language').inner_text()
    reset.click()
    assert page.locator('#source-language').inner_text() == language
    assert not source.input_value() and not result.inner_text()
    assert page.locator('audio').evaluate('a => a.paused && !a.hasAttribute("src")')
    source.fill('새 입력')
    page.locator('#translation-title').hover()
    source.click()
    assert source.input_value() == '새 입력'
    assert not result.inner_text() and not page.locator('audio').get_attribute('src')
    page.get_by_role('link', name='낱말 퀴즈', exact=True).click()
    page.locator('.start').click()
    page.locator('.option').first.wait_for()
    question_id = page.locator('.question-sheet').get_attribute('data-question')
    page.get_by_role('link', name='문장 번역', exact=True).click()
    page.locator('#translation-page').wait_for()
    assert page.evaluate('timer.deadline === null')
    page.go_back()
    page.locator('.question-sheet').wait_for()
    assert page.locator('.question-sheet').get_attribute('data-question') == question_id
    assert page.evaluate('timer.deadline !== null')
    assert not errors, errors
    page.close()
    print(f'{width}px · Standard Korean first, JIT / AI Hub, Qwen3 translation, Jeju-only playback, quiz resume: passed')


def check_word_lookup(page):
    source = page.locator('#source-example')
    result = page.locator('#target-example')
    evidence = page.locator('#word-evidence')
    source.fill('할아버지')
    page.locator('.translation-submit').click()
    page.get_by_role('link', name='제주어대사전', exact=True).wait_for()
    assert result.inner_text() == '하르방' and '하르버지' in evidence.inner_text()
    page.locator('#swap-languages').click()
    assert evidence.is_hidden()
    source.fill('하르버지')
    page.locator('.translation-submit').click()
    page.get_by_role('link', name='제주어대사전', exact=True).wait_for()
    assert result.inner_text() == '할아버지'
    page.locator('#swap-languages').click()
    page.locator('#translation-reset').click()
    assert not result.inner_text() and evidence.is_hidden()


def check_offline_translation(browser):
    context = browser.new_context(offline=True, viewport={'width':360,'height':800})
    page = context.new_page()
    page.goto(Path(__file__).with_name('하나라봉.html').resolve().as_uri() + '#translate')
    for _ in range(2):
        page.get_by_role('button', name='제주어 듣기', exact=True).click()
        page.wait_for_function("() => document.querySelector('audio').currentTime > 0")
        page.locator('#swap-languages').click()
    page.locator('#source-example').fill('어서 오십시오')
    page.locator('.translation-submit').click()
    assert page.locator('#target-example').inner_text() == '혼저 옵서.'
    page.locator('#swap-languages').click()
    page.locator('#source-example').fill('오늘 새로 쓴 문장이에요.')
    page.locator('.translation-submit').click()
    page.get_by_role('button', name='제주어 듣기', exact=True).click()
    assert all('서버판' in page.locator(selector).inner_text() for selector in ('#translation-status','#speech-status'))
    page.locator('#swap-languages').click()
    check_word_lookup(page)
    context.close()
    print('Offline exact pair lookup, Jeju-only recordings before and after swap, server-only guidance: passed')


with sync_playwright() as pw:
    browser = pw.chromium.launch(args=['--no-sandbox'])
    check_translation(browser, 390)
    check_translation(browser, 1440)
    check_offline_translation(browser)
    check_timer(browser)
    check_round(browser, 1440, 900)
    check_round(browser, 390, 844, reduced=True)
    check_round(browser, 320, 568, reduced=True, perfect=True)
    browser.close()
print(f"{len(rows)} eligible questions, {len({q['audio'] for q in rows})} sentence recordings: passed")
