'use strict';

const $ = (selector, parent = document) => parent.querySelector(selector);
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#${name}"/></svg>`;
const escapeHTML = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
const pocket = window.hanarabongPocket;
const mediaURL = path => pocket?.media[path] || path;
const audio = $('#voice');
const dialog = $('#dialog');
const game = $('#game');
const notice = $('#notice');
const welcome = $('#welcome');
const translationPage = $('#translation-page');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
let catalog = [];
let round = null;
let pending = false;
let soundOn = true;
let activeVoice = null;
let noticeTimer;
const questionMs = 30_000;
let timer = null;
let soundContext;
let cueOutput;
const cues = {
  correct: [[784, 0, .10], [1046.5, .075, .15]],
  wrong: [[392, 0, .14], [293.7, .12, .21]],
  warning: [[880, 0, .09]],
  urgent: [[988, 0, .075], [988, .15, .075]]
};
try { soundOn = localStorage.getItem('hanarabong.sound') !== 'off'; } catch {}

function announce(message) {
  clearTimeout(noticeTimer);
  notice.textContent = message;
  notice.hidden = false;
  noticeTimer = setTimeout(() => { notice.hidden = true; }, 3800);
}

async function loadCatalog() {
  if (pocket) {
    catalog = pocket.questions;
  } else {
    const response = await fetch('/api/questions', {cache: 'no-store'});
    if (!response.ok) throw new Error('catalog');
    catalog = await response.json();
  }
  document.querySelectorAll('[data-word]').forEach(card => {
    card.hidden = !catalog.some(q => q.id === card.dataset.word);
  });
  $('.vocabulary').hidden = !$('.word-card:not([hidden])');
  return catalog;
}

// The original quiz groups equivalent answers before drawing, then shuffles each question's options.
function shuffled(items) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function drawQuestions(pool) {
  const byAnswer = new Map();
  for (const question of pool) {
    const answer = question.answer.trim().normalize('NFC');
    if (!byAnswer.has(answer)) byAnswer.set(answer, []);
    byAnswer.get(answer).push(question);
  }
  if (byAnswer.size < 10) return [];
  return shuffled(byAnswer.values()).slice(0, 10).map(variants => {
    const question = variants[Math.floor(Math.random() * variants.length)];
    return {...question, options: shuffled(question.options)};
  });
}

function syncSound() {
  const button = $('#sound-toggle');
  button.setAttribute('aria-pressed', String(soundOn));
  button.setAttribute('aria-label', `효과음과 자동 낭독 ${soundOn ? '켜짐' : '꺼짐'}`);
  button.innerHTML = icon(soundOn ? 'sound' : 'mute') + `<span>소리 ${soundOn ? '켜짐' : '꺼짐'}</span>`;
}

// Open the audio device from a tap, so the five-second cue can play without another tap.
function enableCues() {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!soundOn || !Context) return;
  try {
    soundContext ||= new Context();
    if (soundContext.state !== 'running') soundContext.resume().catch(() => {});
  } catch { /* The visible timer and answer feedback also work without an audio device. */ }
}

function stopCue() {
  cueOutput?.disconnect();
  cueOutput = null;
}

function playCue(kind) {
  stopCue();
  if (!soundOn || !soundContext || soundContext.state === 'closed') return;
  cueOutput = soundContext.createGain();
  cueOutput.connect(soundContext.destination);
  const start = soundContext.currentTime;
  for (const [frequency, offset, duration] of cues[kind]) {
    const tone = soundContext.createOscillator();
    const envelope = soundContext.createGain();
    const at = start + offset;
    tone.type = kind === 'wrong' ? 'triangle' : 'sine';
    tone.frequency.value = frequency;
    envelope.gain.setValueAtTime(.0001, at);
    envelope.gain.exponentialRampToValueAtTime(kind === 'correct' ? .065 : .10, at + .012);
    envelope.gain.exponentialRampToValueAtTime(.0001, at + duration);
    tone.connect(envelope).connect(cueOutput);
    tone.onended = () => { tone.disconnect(); envelope.disconnect(); };
    tone.start(at);
    tone.stop(at + duration);
  }
}

function syncPlayback() {
  document.querySelectorAll('[data-listen]').forEach(button => {
    const playing = activeVoice === button.dataset.listen && !audio.paused;
    $('use', button)?.setAttribute('href', playing ? '#pause' : '#sound');
    button.setAttribute('aria-pressed', String(Boolean(playing)));
    button.dataset.listenLabel ||= button.getAttribute('aria-label');
    button.setAttribute('aria-label', playing ? '문장 낭독 멈추기' : button.dataset.listenLabel);
    const label = $('.audio-label', button);
    if (label) label.textContent = playing ? '멈추기' : button.dataset.label;
  });
}

function stopVoice() {
  activeVoice = null;
  audio.pause();
  audio.removeAttribute('src');
  audio.load();
  syncPlayback();
}

async function playVoice(q) {
  if (!q) return;
  if (activeVoice === q.id && !audio.paused) { stopVoice(); return; }
  audio.pause();
  activeVoice = q.id;
  audio.src = mediaURL(q.audio);
  try { await audio.play(); }
  catch (error) {
    if (error.name !== 'AbortError' && activeVoice === q.id) {
      activeVoice = null;
      announce('듣기 버튼을 눌러 문장을 다시 들어보세요.');
    }
  }
  syncPlayback();
}

audio.addEventListener('play', syncPlayback);
audio.addEventListener('pause', syncPlayback);
audio.addEventListener('ended', () => { activeVoice = null; syncPlayback(); });

function listenButton(q, label = '할망 목소리 듣기') {
  return `<button class="small-button" data-listen="${escapeHTML(q.id)}" data-label="${label}" aria-label="${escapeHTML(q.sentence)} 듣기" aria-pressed="false">${icon('sound')}<span class="audio-label">${label}</span></button>`;
}

function showDialog(content) {
  pauseTimer();
  stopVoice();
  $('#dialog-content').innerHTML = content;
  if (!dialog.open) dialog.showModal();
}
dialog.addEventListener('close', () => {
  stopVoice();
  if (!dialog.open) resumeTimer();
});
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});

function showHelp() {
  showDialog(`<h2 id="dialog-title">놀이 방법</h2><ol><li>제주어 문장 아래의 표준어 뜻을 읽어요.</li><li>30초 안에 빈칸에 어울리는 낱말을 골라요. 할망의 힌트를 펼칠 수 있고, 마지막 5초 동안 매초 알림음이 울려요.</li><li>정답을 맞히면 할망이 문장을 읽어줘요. 틀리거나 시간을 넘긴 표현은 마지막에 다시 만나요.</li></ol><button class="sub-button" data-action="close">알겠어요</button>`);
}

function showWord(q) {
  const art = {'JEJU-008':'wave', 'JEJU-010':'garlic', 'JEJU-011':'tree'}[q.id];
  showDialog(`<span class="word-art ${art}" aria-hidden="true"></span><h2 id="dialog-title">${escapeHTML(q.answer)}</h2><span>${escapeHTML(q.meaning)}</span><p class="sample">${escapeHTML(q.sentence)}</p><p>${escapeHTML(q.standard)}</p>${listenButton(q)}`);
}

async function startRound() {
  if (pending) return;
  enableCues();
  pending = true;
  const button = $('[data-action="start"]:not([hidden])', welcome.hidden ? game : welcome);
  button?.setAttribute('aria-busy', 'true');
  try {
    const questions = drawQuestions(await loadCatalog());
    if (!translationPage.hidden || location.hash === '#translate') return;
    if (!questions.length) { announce('지금은 열 낱말이 준비되지 않았어요. 잠시 뒤 다시 시작해 주세요.'); return; }
    stopVoice();
    round = {questions, index: 0, answers: [], hint: false};
    welcome.hidden = true;
    game.hidden = false;
    renderQuestion();
  } catch { announce('문제를 불러오지 못했어요. 연결을 확인하고 다시 눌러주세요.'); }
  finally { pending = false; button?.removeAttribute('aria-busy'); }
}

function goHome() {
  stopTimer();
  stopCue();
  stopVoice();
  if (dialog.open) dialog.close();
  round = null;
  if (!translationPage.hidden) location.hash = 'quiz';
  game.hidden = true;
  game.replaceChildren();
  welcome.hidden = false;
  document.title = '하나라봉 · 할망의 제주어 낱말책';
  window.scrollTo(0, 0);
  $('.start').focus({preventScroll:true});
}

function companion(mood = 'idle') {
  const alt = {idle:'문제 곁에서 기다리는 제주 할망', happy:'정답을 맞혀 기뻐하는 할망', hint:'힌트를 건네는 할망', sad:'아쉬운 표정을 짓는 할망'}[mood];
  const filename = {idle:'halmang_idle.png', happy:'halmang_happy.gif', hint:'halmang_hint.webp', sad:'halmang_sad.webp'}[mood];
  const still = mood === 'idle' ? '' : `<source media="(prefers-reduced-motion: reduce)" srcset="${mediaURL(`assets/halmang_${mood}_still.png`)}">`;
  return `<div class="portrait"><picture>${still}<img src="${mediaURL('assets/' + filename)}" alt="${alt}" width="512" height="640"></picture></div><figcaption>제주 할망</figcaption>`;
}

function displayMood(mood) {
  const figure = $('.companion', game);
  figure.innerHTML = companion(mood);
  if (mood !== 'happy') return;
  const seal = document.createElement('span');
  seal.className = 'answer-seal';
  seal.setAttribute('aria-hidden', 'true');
  seal.textContent = '정답';
  figure.append(seal);
  if (reduced.matches) return;
  const burst = document.createElement('span');
  burst.className = 'confetti';
  burst.setAttribute('aria-hidden','true');
  const colors = ['#e9a238','#83945f','#d77b49','#f1c95c'];
  for (let i = 0; i < 16; i++) {
    const part = document.createElement('i');
    const angle = Math.PI + Math.PI * i / 15;
    const radius = figure.offsetWidth * (.42 + (i % 4) * .08);
    part.style.setProperty('--x', Math.cos(angle) * radius + 'px');
    part.style.setProperty('--y', Math.sin(angle) * radius - 22 + 'px');
    part.style.setProperty('--turn', (i % 2 ? -1 : 1) * (75 + 10*i) + 'deg');
    part.style.setProperty('--color', colors[i % 4]);
    burst.append(part);
  }
  figure.append(burst);
  setTimeout(() => burst.remove(), 950);
}

function renderTrail() {
  $('.trail', game).innerHTML = round.questions.map((q, i) => {
    const picked = round.answers[i];
    const state = picked !== undefined ? (picked === q.answer ? 'done' : 'miss') : (i === round.index ? 'current' : '');
    const earned = picked === q.answer && i === round.index;
    return `<i class="${state}${earned ? ' collected' : ''}"></i>`;
  }).join('');
}

function sentenceMarkup(q, filled = false) {
  const [before, after] = q.prompt.split('____').map(escapeHTML);
  const word = filled ? `<span class="filled">${escapeHTML(q.answer)}</span>` : '<span class="blank" aria-label="빈칸">?</span>';
  return before + word + after;
}

function focusPage(title) {
  title.tabIndex = -1;
  title.focus({preventScroll:true});
  window.scrollTo(0,0);
}

function stopTimer() {
  if (timer) clearInterval(timer.interval);
  timer = null;
}

function updateTimer() {
  if (!timer || timer.deadline === null) return;
  const left = Math.max(0, timer.deadline - performance.now());
  const seconds = Math.ceil(left / 1000);
  if (timer.seconds !== seconds) {
    timer.seconds = seconds;
    timer.label.textContent = seconds + '초';
    if (seconds > 0 && seconds <= 5) {
      timer.element.classList.add('urgent');
      playCue(seconds <= 2 ? 'urgent' : 'warning');
    }
  }
  const fraction = reduced.matches ? seconds / (questionMs / 1000) : left / questionMs;
  timer.fill.style.transform = `scaleX(${fraction})`;
  if (left === 0) finishAnswer(null);
}

function pauseTimer() {
  if (!timer || timer.deadline === null) return;
  updateTimer();
  if (!timer) return;
  timer.remaining = Math.max(0, timer.deadline - performance.now());
  timer.deadline = null;
  clearInterval(timer.interval);
  stopCue();
}

function resumeTimer() {
  if (!timer || timer.deadline !== null || !translationPage.hidden || dialog.open) return;
  timer.deadline = performance.now() + timer.remaining;
  timer.interval = setInterval(updateTimer, 100);
  updateTimer();
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) updateTimer();
});

function renderQuestion() {
  const q = round.questions[round.index];
  document.title = `${round.index + 1}번째 문제 · 하나라봉`;
  const prompt = sentenceMarkup(q);
  game.innerHTML = `<div class="game-head"><span class="page-count">문제 <b>${round.index + 1}</b> <small>/ 10</small></span><span class="trail" aria-hidden="true"></span></div>
  <article class="book-sheet question-sheet" data-question="${escapeHTML(q.id)}">
  <div class="question-top"><div class="question-copy"><p class="category">${icon('stone-grandfather')}${escapeHTML(q.category)}</p>
  <div class="prompt-stack"><span class="sizer" aria-hidden="true">${prompt}</span><span class="sizer" aria-hidden="true">${sentenceMarkup(q,true)}</span><h1 id="prompt">${prompt}</h1></div>
  <p class="standard-line"><span>표준어</span>${escapeHTML(q.standard)}</p></div>
  <figure class="companion">${companion()}</figure></div>
  <div class="question-timer"><div class="timer-caption"><span>${icon('clock')} 남은 시간</span><strong role="timer" aria-label="남은 시간">30초</strong></div><div class="timer-track" aria-hidden="true"><div class="timer-fill"></div></div></div>
  <h2 class="options-heading" id="options-title">빈칸에 들어갈 낱말은?</h2><div class="options" role="group" aria-labelledby="options-title">
  ${q.options.map((word, i) => `<button class="option" data-choice="${i}"><span class="option-num" aria-hidden="true">${i+1}</span><span class="option-word">${escapeHTML(word)}</span><span class="option-state"></span></button>`).join('')}</div>
  <div class="assistance"><button class="hint-button" data-action="hint" aria-expanded="false" aria-controls="hint-text">${icon('hint')} 할망의 힌트</button><p id="hint-text" class="hint-box" hidden>${escapeHTML(q.hint)}</p></div>
  <div id="feedback" aria-live="polite"></div></article>`;
  renderTrail();
  focusPage($('#prompt'));
  const element = $('.question-timer', game);
  timer = {element, fill: $('.timer-fill', element), label: $('[role=timer]', element),
           deadline: null, remaining: questionMs};
  resumeTimer();
}

function hint() {
  if (!round || round.answers[round.index] !== undefined) return;
  round.hint = !round.hint;
  $('.hint-button').setAttribute('aria-expanded', String(round.hint));
  $('#hint-text').hidden = !round.hint;
  displayMood(round.hint ? 'hint' : 'idle');
}

function choose(index) {
  if (!timer || timer.deadline === null) return;
  enableCues();
  // Check the deadline even if rendering or background-tab throttling delayed a tick.
  updateTimer();
  if (!timer) return;
  const picked = round.questions[round.index].options[index];
  if (picked !== undefined) finishAnswer(picked);
}

function finishAnswer(picked) {
  if (!timer) return;
  stopTimer();
  const q = round.questions[round.index];
  round.answers[round.index] = picked;
  const right = picked === q.answer;
  playCue(right ? 'correct' : 'wrong');
  $('#prompt').innerHTML = sentenceMarkup(q, true);
  $('.assistance').hidden = true;
  document.querySelectorAll('.option').forEach((button,i) => {
    const word = q.options[i], correct = word === q.answer, wrong = !correct && word === picked;
    button.setAttribute('aria-disabled','true');
    button.tabIndex = -1;
    if (correct || wrong) {
      button.classList.add(correct ? 'correct' : 'wrong');
      $('.option-num',button).innerHTML = icon(correct ? 'check' : 'close');
      $('.option-state',button).textContent = correct ? '정답' : '내가 고른 답';
    }
  });
  displayMood(right ? 'happy' : 'sad');
  renderTrail();
  const title = right ? '정답이에요!' : picked === null ? '시간이 다 됐어요.' : '다시 한번 읽어봐요.';
  const chosenLine = picked !== null && !right
    ? `<div><dt>내가 고른 답</dt><dd><b>${escapeHTML(picked)}</b> (${escapeHTML(q.optionMeanings[picked])})</dd></div>` : '';
  $('#feedback').innerHTML = `<section class="feedback"><strong class="feedback-title stamp-in ${right ? '' : 'missed'}">${icon(right ? 'check' : 'look-back')}${title}</strong>
  <dl class="answer-meanings">${chosenLine}<div><dt>정답</dt><dd><b>${escapeHTML(q.answer)}</b> (${escapeHTML(q.meaning)})</dd></div></dl>
  <div class="actions">${listenButton(q)}<button class="primary" data-action="next">${round.index === 9 ? '결과 보기' : '다음 문제'} ${icon('arrow')}</button></div></section>`;
  if (right && soundOn) playVoice(q);
}

function next() {
  if (!round || round.answers[round.index] === undefined) return;
  stopCue();
  stopVoice();
  round.index++;
  round.hint = false;
  if (round.index === round.questions.length) renderResults();
  else renderQuestion();
}

function renderResults() {
  const missed = round.questions.filter((q,i) => q.answer !== round.answers[i]);
  const score = round.questions.length - missed.length;
  document.title = '결과와 복습 · 하나라봉';
  game.innerHTML = `<article class="book-sheet results-sheet"><div class="result-top"><div><div class="book-label">${icon('stone-grandfather')}오늘의 낱말책</div><h1>열 문장을 만났어요.</h1><div class="score" aria-label="10문제 중 ${score}문제 정답">${score}<span> / 10문제 정답</span></div><button class="primary" data-action="start">${icon('again')}다시 풀기</button></div><figure class="companion">${companion('happy')}</figure></div>
  <section class="review-section"><h2 class="review-heading">틀린 표현 복습${missed.length ? ' · '+missed.length+'개' : ''}</h2>
  ${missed.length ? missed.map(q => `<article class="review-item"><div>${round.answers[round.questions.indexOf(q)] === null ? '<small class="timeout-tag">시간 초과</small>' : ''}<h3>${escapeHTML(q.sentence)}</h3><p>${escapeHTML(q.standard)}</p></div>${listenButton(q, '듣기')}</article>`).join('') : '<p class="perfect">모두 맞혔어요! 열 낱말을 새로 뽑아 한 번 더 도전해 보세요.</p>'}</section></article>`;
  focusPage($('h1', game));
}

document.addEventListener('click', async event => {
  const button = event.target.closest('button');
  if (!button) return;
  const action = button.dataset.action;
  if (action === 'sound') {
    soundOn = !soundOn;
    try { localStorage.setItem('hanarabong.sound', soundOn ? 'on' : 'off'); } catch {}
    if (soundOn) enableCues();
    else { stopCue(); stopVoice(); }
    syncSound();
  } else if (action === 'help') showHelp();
  else if (action === 'start') startRound();
  else if (action === 'hint') hint();
  else if (action === 'next') next();
  else if (action === 'close') dialog.close();
  else if (action === 'home') goHome();
  else if (button.dataset.choice !== undefined) choose(Number(button.dataset.choice));
  else if (button.dataset.word || button.dataset.listen) {
    const id = button.dataset.word || button.dataset.listen;
    try {
      if (!catalog.length) await loadCatalog();
      if (!translationPage.hidden) return;
      const q = round?.questions.find(item => item.id === id) || catalog.find(item => item.id === id);
      if (!q) { announce('지금은 이 문장을 열 수 없어요.'); return; }
      if (button.dataset.word) showWord(q);
      else playVoice(q);
    } catch { announce('문장을 불러오지 못했어요. 잠시 뒤 다시 눌러주세요.'); }
  }
});

$('#home-link').addEventListener('click', event => {
  event.preventDefault();
  if (round && round.index < round.questions.length) {
    const stay = translationPage.hidden ? '이어서 풀기' : '여기 머무르기';
    showDialog(`<h2 id="dialog-title">첫 장으로 갈까요?</h2><p>지금 풀고 있는 판이 끝나요.</p><div class="dialog-actions"><button class="sub-button" data-action="close">${stay}</button><button class="small-button" data-action="home">첫 장으로</button></div>`);
  } else goHome();
});

const examples = [...translationPage.querySelectorAll('.suggestion')];
const sourceBox = $('#source-example');
const targetBox = $('#target-example');
const sourceLanguage = $('#source-language');
const targetLanguage = $('#target-language');
const swapButton = $('#swap-languages');
const statusLine = $('#translation-status');
const evidenceBox = $('#word-evidence');
const submitButton = $('.translation-submit');
const resetButton = $('#translation-reset');
const submitPercent = $('.submit-percent', submitButton);
const meter = $('#translation-progress');
const jejuListen = $('#jeju-listen');
let activeExample = examples[0];
let showingIntroExample = true;
let sourceIsStandard = true;
let translationRequest = null;
let completionTimer;
let speechRequest = null;
let clips = [];
let clipIndex = 0;
const offlineNote = '새 문장 번역과 음성 합성은 서버판에서 이용할 수 있어요.';

function refreshControls() {
  const empty = !sourceBox.value.trim();
  $('#source-count').textContent = sourceBox.value.length + ' / 300자';
  submitButton.disabled = empty || sourceBox.value.length > sourceBox.maxLength || !!translationRequest;
  resetButton.disabled = !sourceBox.value && !targetBox.textContent;
  submitButton.setAttribute('aria-busy', String(!!translationRequest));
  const playing = audio.src === clips[clipIndex] && !audio.paused;
  const action = speechRequest ? '낭독 준비 중' : playing ? '멈추기' : '듣기';
  jejuListen.disabled = !!speechRequest || (sourceIsStandard ? !targetBox.textContent : empty);
  $('use', jejuListen).setAttribute('href', playing ? '#pause' : '#sound');
  $('.listen-label', jejuListen).textContent = '제주어 ' + action;
  jejuListen.setAttribute('aria-busy', String(!!speechRequest));
  jejuListen.setAttribute('aria-pressed', String(playing));
}

function showProgress(value = null, stage = '') {
  clearTimeout(completionTimer);
  meter.hidden = value === null;
  meter.value = value ?? 0;
  submitButton.toggleAttribute('data-progress', value !== null);
  submitButton.style.setProperty('--fraction', meter.value / 100);
  submitPercent.hidden = value === null;
  submitPercent.textContent = value === null ? '' : (value === 100 ? '' : '약 ') + value + '%';
  if (stage && statusLine.textContent !== stage) statusLine.textContent = stage;
  if (value === 100) completionTimer = setTimeout(() => showProgress(), 1400);
}

function cancelTranslation() {
  showProgress();
  if (translationRequest) {
    translationRequest.abort();
    translationRequest = null;
    statusLine.textContent = '';
  }
}

function clearSentenceAudio() {
  speechRequest?.abort();
  speechRequest = null;
  if (clips.length) {
    stopVoice();
    if (!pocket) clips.forEach(url => URL.revokeObjectURL(url));
  }
  clips = [];
  clipIndex = 0;
  $('#speech-status').textContent = '';
  $('#result-speech-status').textContent = '';
}

function refreshLanguages() {
  const source = sourceIsStandard ? 'standard' : 'jeju';
  const target = sourceIsStandard ? 'jeju' : 'standard';
  const names = {jeju: '제주어', standard: '표준어'};
  sourceLanguage.textContent = names[source];
  targetLanguage.textContent = names[target];
  const jejuHeading = sourceIsStandard ? targetLanguage : sourceLanguage;
  if (jejuHeading.nextElementSibling !== jejuListen) jejuHeading.after(jejuListen);
  swapButton.title = names[target] + '에서 ' + names[source] + '로 번역 방향 바꾸기';
  swapButton.setAttribute('aria-label', swapButton.title);
  swapButton.classList.toggle('reversed', !sourceIsStandard);
  for (const card of examples) {
    card.setAttribute('aria-pressed', String(card === activeExample));
    $('use', card).setAttribute('href', card === activeExample ? '#check' : '#arrow');
    for (const line of card.querySelectorAll('[data-lang]')) line.hidden = line.dataset.lang !== source;
  }
  if (activeExample) {
    sourceBox.value = activeExample.querySelector('[data-lang=' + source + ']').textContent;
    targetBox.textContent = activeExample.querySelector('[data-lang=' + target + ']').textContent;
    statusLine.textContent = '제주특별자치도 생활방언에 실린 문장이에요.';
  }
  refreshControls();
}

function clearEntry() {
  cancelTranslation();
  clearSentenceAudio();
  targetBox.textContent = '';
  showEvidence([]);
  statusLine.textContent = pocket ? offlineNote : '';
}

function showEvidence(entries) {
  const list = $('ul', evidenceBox);
  list.replaceChildren();
  for (const entry of entries) {
    const item = document.createElement('li');
    item.append(entry.je + ' · ' + entry.ko);
    const link = document.createElement('a');
    link.href = entry.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = entry.source;
    item.append(link);
    if (entry.variants?.length) {
      const variants = document.createElement('small');
      variants.textContent = '함께 쓰는 말 · ' + entry.variants.join(', ');
      item.append(variants);
    }
    list.append(item);
  }
  evidenceBox.hidden = !entries.length;
}

function showTranslation(result) {
  targetBox.textContent = result.text;
  showEvidence(result.evidence || []);
  if (result.kind === 'unverified') {
    statusLine.textContent = '원문의 뜻과 말투를 보존한 번역을 확인하기 어려워요. 문장을 짧게 나눠 보세요.';
    showProgress();
    return;
  }
  if (result.kind === 'word') statusLine.textContent = '기본 낱말 · 대표 표현';
  else if (result.kind === 'corpus') statusLine.textContent = result.sources.join(' · ') + '에 실린 문장쌍이에요.';
  else statusLine.textContent = 'AI 번역 · ' + result.model + ' · 참고 자료 ' + result.sources.join(' · ');
  showProgress(100);
}

function resetEntry() {
  showingIntroExample = false;
  activeExample = null;
  sourceBox.value = '';
  clearEntry();
  statusLine.textContent = '';
  refreshLanguages();
}

function dismissIntroExample() {
  if (showingIntroExample) resetEntry();
}

sourceBox.addEventListener('pointerenter', dismissIntroExample);
sourceBox.addEventListener('focus', dismissIntroExample);
sourceBox.addEventListener('click', dismissIntroExample);
resetButton.addEventListener('click', () => {
  resetEntry();
  sourceBox.focus({preventScroll: true});
});

sourceBox.addEventListener('input', () => {
  showingIntroExample = false;
  activeExample = null;
  clearEntry();
  refreshLanguages();
});
for (const card of examples) {
  card.addEventListener('click', () => {
    showingIntroExample = false;
    clearEntry();
    activeExample = card;
    refreshLanguages();
  });
}
swapButton.addEventListener('click', () => {
  const previousSource = sourceBox.value;
  const previousResult = targetBox.textContent;
  clearEntry();
  sourceIsStandard = !sourceIsStandard;
  if (previousResult && !activeExample) {
    sourceBox.value = previousResult;
    targetBox.textContent = previousSource;
    statusLine.textContent = previousResult.length > sourceBox.maxLength
      ? '다시 번역하려면 입력 문장을 300자 이내로 줄여 주세요.' : '원문과 번역문의 위치를 바꿨어요.';
  }
  refreshLanguages();
});

async function requestSentence(path, body, signal) {
  let response;
  try {
    response = await fetch('/api/' + path, {method: 'POST', headers: {'Content-Type': 'application/json',
      'Accept': path === 'translate' ? 'text/event-stream' : 'audio/wav'},
      body: JSON.stringify(body), signal});
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error('서버에 연결할 수 없어요. 연결을 확인하고 다시 눌러 주세요.');
  }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || '요청을 처리하지 못했어요. 잠시 후 다시 눌러 주세요.');
  }
  return response;
}

async function receiveTranslation(response, signal) {
  if (!response.headers.get('Content-Type')?.includes('text/event-stream')) return response.json();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (!signal.aborted) {
      const {done, value} = await reader.read();
      if (done || signal.aborted) break;
      buffer += decoder.decode(value, {stream: true});
      const events = buffer.split('\n\n');
      buffer = events.pop();
      for (const event of events) {
        if (!event.startsWith('data: ')) continue;
        const update = JSON.parse(event.slice(6));
        if (update.error) throw new Error(update.error);
        if (update.result) return update.result;
        showProgress(update.progress, update.stage);
      }
    }
    throw new Error('번역 연결이 끊겼어요. 다시 눌러 주세요.');
  } finally { reader.releaseLock(); }
}

async function translateEntry() {
  const text = sourceBox.value.trim();
  if (submitButton.disabled || !text || translationRequest) return;
  if (sourceIsStandard) clearSentenceAudio();
  showEvidence([]);
  showProgress(0, '번역을 시작하고 있어요.');
  if (pocket) {
    const [source, target] = sourceIsStandard ? ['standard', 'sentence'] : ['sentence', 'standard'];
    const key = s => s.normalize('NFC').replace(/\s+/g, ' ').trim().replace(/\s+([,.?!])/g, '$1').replace(/\.+$/, '').trim();
    const term = key(text);
    const word = pocket.words.find(w => sourceIsStandard ? w.ko === term : w.je === term || w.variants?.includes(term));
    const match = pocket.questions.find(q => key(q[source]) === term);
    if (word) showTranslation({kind: 'word', text: sourceIsStandard ? word.je : word.ko, evidence: [word]});
    else if (match) showTranslation({kind: 'corpus', text: match[target], sources: ['생활방언']});
    else { targetBox.textContent = ''; statusLine.textContent = offlineNote; showProgress(); }
    refreshControls();
    return;
  }
  const request = translationRequest = new AbortController();
  targetBox.textContent = '';
  statusLine.textContent = '자료를 찾아 문장을 번역하고 있어요.';
  refreshControls();
  try {
    const response = await requestSentence('translate', {text, direction: sourceIsStandard ? 'ko-je' : 'je-ko'}, request.signal);
    const result = await receiveTranslation(response, request.signal);
    if (request.signal.aborted) return;
    showTranslation(result);
  } catch (error) {
    if (!request.signal.aborted) { showProgress(); statusLine.textContent = error.message; }
  } finally {
    if (translationRequest === request) { translationRequest = null; refreshControls(); }
  }
}
submitButton.addEventListener('click', translateEntry);
sourceBox.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.isComposing) {
    event.preventDefault();
    translateEntry();
  }
});

// A generated translation can be longer than the 300-character input limit.
function speechParts(text) {
  const parts = [];
  while (text.length > 300) {
    const lastSpace = text.lastIndexOf(' ', 300);
    const end = lastSpace > 0 ? lastSpace : 300;
    parts.push(text.slice(0, end));
    text = text.slice(end).trimStart();
  }
  if (text) parts.push(text);
  return parts;
}

async function playSentencePart() {
  const url = clips[clipIndex];
  const status = sourceIsStandard ? $('#result-speech-status') : $('#speech-status');
  activeVoice = null;
  if (audio.src !== url) audio.src = url;
  status.textContent = 'VoxCPM2 · AI Hub + JSS';
  try { await audio.play(); }
  catch (error) {
    if (clips[clipIndex] === url && error.name !== 'AbortError') status.textContent = '듣기를 다시 눌러 재생해 주세요.';
  }
  refreshControls();
}

async function readSentence() {
  const text = (sourceIsStandard ? targetBox.textContent : sourceBox.value).trim();
  const status = sourceIsStandard ? $('#result-speech-status') : $('#speech-status');
  if (!text || speechRequest) return;
  if (clips.length && audio.src === clips[clipIndex] && !audio.paused) {
    audio.pause();
    audio.currentTime = 0;
    clipIndex = 0;
    return;
  }
  if (!clips.length && pocket) {
    const q = pocket.questions.find(item => item.sentence === text);
    const url = pocket.media[q?.audio];
    if (!url) { status.textContent = offlineNote; return; }
    clips.push(url);
  }
  if (!clips.length) {
    const request = speechRequest = new AbortController();
    status.textContent = '할망이 읽을 준비를 하고 있어요.';
    refreshControls();
    try {
      for (const part of speechParts(text)) {
        const response = await requestSentence('speech', {text: part}, request.signal);
        const wav = await response.blob();
        if (request.signal.aborted) return;
        clips.push(URL.createObjectURL(wav));
      }
    } catch (error) {
      if (!request.signal.aborted) {
        clearSentenceAudio();
        status.textContent = error.message;
        refreshControls();
      }
      return;
    } finally {
      if (speechRequest === request) { speechRequest = null; refreshControls(); }
    }
  }
  await playSentencePart();
}
jejuListen.addEventListener('click', readSentence);
audio.addEventListener('ended', () => {
  if (audio.src !== clips[clipIndex]) return;
  if (clipIndex + 1 < clips.length) {
    clipIndex++;
    playSentencePart();
  } else clipIndex = 0;
});
for (const event of ['play', 'ended', 'pause']) audio.addEventListener(event, refreshControls);

let quizTitle = document.title;
function updateChapter() {
  const translating = location.hash === '#translate';
  if (translating === !translationPage.hidden) return;
  if (translating) {
    quizTitle = document.title;
    pauseTimer();
    stopCue();
    stopVoice();
  } else {
    cancelTranslation();
    clearSentenceAudio();
    refreshControls();
  }
  translationPage.hidden = !translating;
  welcome.hidden = translating || !!round;
  game.hidden = translating || !round;
  $('#quiz-settings').hidden = translating;
  $('.footer small').textContent = translating && !pocket ? 'JIT · AI Hub · 제주특별자치도 생활방언'
    : '제주특별자치도 생활방언 자료를 바탕으로 만들었어요.';
  for (const link of document.querySelectorAll('.chapter-nav a')) {
    if (link.hash === (translating ? '#translate' : '#quiz')) link.setAttribute('aria-current','page');
    else link.removeAttribute('aria-current');
  }
  if (dialog.open) dialog.close();
  document.title = translating ? '문장 번역 · 하나라봉' : round ? quizTitle : '하나라봉 · 할망의 제주어 낱말책';
  const visiblePage = translating ? translationPage : round ? game : welcome;
  focusPage($('h1', visiblePage));
  if (!translating) {
    resumeTimer();
    if (!catalog.length) loadCatalog().catch(() => {});
  }
}

$('.skip').addEventListener('click', event => {
  event.preventDefault();
  $('#main').focus();
});
window.addEventListener('hashchange', updateChapter);
refreshLanguages();
updateChapter();
syncSound();
if (translationPage.hidden) loadCatalog().catch(() => { /* Starting a round or opening a card will retry. */ });
