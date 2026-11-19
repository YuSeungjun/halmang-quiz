"""Run a reproducible, bidirectional Qwen3-32B translation LoRA pilot."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3

from translation import INDEX, PROJECT, clean

BASE = 'Qwen/Qwen3-32B'
SEED = 37
DAILY = re.compile('아빠|아버지|어머니|엄마|할아버지|할머니|동생|가족|사랑|좋아|마음|친구|학교|마트|시장|밥|식사|국수|같이|함께|갈래|줄래|주세요|시겠|갈까|싶어|만나')
CONVERSATION = [re.compile(p) for p in (
    r'(?:갈|올|실|을|할|볼)래(?:요)?(?:[?.!]|$)', r'같이|함께', r'주\s*(?:시겠|세요|실래|십시오)',
    r'좋아했|좋아하|사랑', r'고 있(?:었|어|습)', r'(?:으러|러) 가', r'(?:한|간|온|는)대[요?]',
)]
# Short standard/Jejueo parallel examples in 제주어학교 lesson 717. Keep punctuation
# for the contrast between a speaker's intention and an invitation to the listener.
LESSON = [('집에 갈래?', '집의 가젠?'), ('집에 갈래.', '집의 가젠.'),
          ('집에 가겠어요?', '집의 가쿠과?'), ('이제 갈까요?', '이제 가카마씨?'),
          ('밥 먹을래?', '밥 먹젠?'), ('밥 먹을래.', '밥 먹젠.')]
LANGS = {'ko-je': '표준어를 제주어로', 'je-ko': '제주어를 표준어로'}


def key(text):
    return re.sub(r'[^\w]', '', text.casefold())


def usable(ko, je):
    return (4 <= min(len(ko), len(je)) and max(len(ko), len(je)) <= 95 and
            min(len(ko), len(je)) * 2.5 >= max(len(ko), len(je)) and
            not re.search(r'[#<>\x00-\x1f]', ko + je) and key(ko) != key(je))


def datasets():
    rng = random.Random(SEED)
    jit = PROJECT / 'data/jit'
    with (jit / 'ko.dev').open() as ko, (jit / 'je.dev').open() as je:
        dev = [('JIT dev', clean(a), clean(b)) for a, b in zip(ko, je, strict=True)]
    labels = PROJECT / 'data/aihub-71558/extracted/Validation/labels'
    files = sorted((labels / 'dialogue').glob('*.json'))
    rng.shuffle(files)
    valid = []
    for path in files[:8]:
        for row in json.loads(path.read_text())['transcription']['sentences']:
            if row.get('standard') and row.get('dialect'):
                valid.append(('AI Hub validation', clean(row['standard']), clean(row['dialect'])))
    if not valid:
        raise ValueError('AI Hub validation dialogue labels missing')
    held = []
    excluded = set()
    for pool in (dev, valid):
        rng.shuffle(pool)
        accepted = 0
        for source, ko, je in pool:
            if usable(ko, je) and not {key(ko), key(je)} & excluded:
                held.append((source, ko, je))
                excluded.update((key(ko), key(je)))
                accepted += 1
            if accepted == 32:
                break
        if accepted != 32:
            raise ValueError(f'Too few held-out examples: {accepted}')
    selected = []
    with closing(sqlite3.connect(INDEX.as_uri() + '?mode=ro', uri=True)) as db:
        for source, limit in [('생활방언', 260), ('AI Hub', 420), ('JIT', 420)]:
            pool = db.execute('SELECT ko,je FROM pair WHERE source=? ORDER BY id', (source,)).fetchall()
            rng.shuffle(pool)
            # Keep both topical examples and a random sample of other utterances.
            conversations = [pair for pattern in CONVERSATION for pair in
                             [p for p in pool if pattern.search(p[0])][:18]]
            topical = [pair for pair in pool if DAILY.search(pair[0])][:limit // 3]
            accepted = 0
            for ko, je in conversations + topical + pool:
                if usable(ko, je) and not {key(ko), key(je)} & excluded:
                    selected.append((source, ko, je))
                    excluded.update((key(ko), key(je)))
                    accepted += 1
                if accepted == limit:
                    break
    held_keys = {key(text) for _, ko, je in held for text in (ko, je)}
    existing = {(ko, je) for _, ko, je in selected}
    for ko, je in LESSON:
        if not {key(ko), key(je)} & held_keys and (ko, je) not in existing:
            selected.append(('제주어학교', ko, je))
    rng.shuffle(selected)
    return selected, held


def encode(tokenizer, pair, direction):
    _, ko, je = pair
    source, target = (ko, je) if direction == 'ko-je' else (je, ko)
    system = f'{LANGS[direction]} 번역한다. 뜻, 시제, 높임과 말투를 유지한다. translation 필드에 완성된 번역문만 쓴다.'
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps({'입력 문장': source}, ensure_ascii=False)}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    answer = tokenizer.encode(json.dumps({'translation': target}, ensure_ascii=False), add_special_tokens=False) + [tokenizer.eos_token_id]
    return prefix + answer, [-100] * len(prefix) + answer


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    training, held = datasets()
    directions = list(LANGS)
    assignments = [(p, d) for i, p in enumerate(training)
                   for d in (directions if p[0] == '제주어학교' else [directions[i % 2]])]
    random.Random(SEED).shuffle(assignments)
    examples = [encode(tokenizer, pair, d) for pair, d in assignments]
    validation = {d: [encode(tokenizer, p, d) for p in held] for d in directions}
    report = {'base': BASE, 'seed': SEED, 'training_sources': dict(Counter(p[0] for p in training)),
              'training_directions': dict(Counter(d for _, d in assignments)),
              'validation_sources': dict(Counter(p[0] for p in held)), 'batch_size': args.batch,
              'data_sha256': hashlib.sha256(json.dumps([training, held], ensure_ascii=False).encode()).hexdigest()}
    print(report, flush=True)
    kw = {'gguf_file': str(args.gguf.resolve())} if args.gguf else {}
    base = AutoModelForCausalLM.from_pretrained(args.base, device_map={'': 0}, dtype=torch.bfloat16, **kw)
    if base.config.model_type != 'qwen3' or base.config.num_hidden_layers != 64 or base.config.hidden_size != 5120:
        raise ValueError('Expected Qwen3-32B')
    base.config.use_cache = False
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=['q_proj', 'v_proj'],
                        layers_to_transform=list(range(40,64)), task_type='CAUSAL_LM')
    model = get_peft_model(base, config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)

    def batch_loss(batch):
        width = max(len(x) for x, _ in batch)
        inputs = torch.tensor([x + [tokenizer.pad_token_id] * (width - len(x)) for x, _ in batch], device='cuda')
        labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch], device='cuda')
        return model(input_ids=inputs, attention_mask=inputs.ne(tokenizer.pad_token_id), labels=labels).loss

    @torch.inference_mode()
    def evaluate():
        model.eval()
        scores = {}
        for direction, dataset in validation.items():
            total, tokens = 0., 0
            for start in range(0, len(dataset), args.batch):
                batch = dataset[start:start+args.batch]
                count = sum(sum(v != -100 for v in y) for _, y in batch)
                total += batch_loss(batch).item() * count
                tokens += count
            scores[direction] = round(total / tokens, 4)
        model.train()
        return scores

    report['base_token_nll'] = evaluate()
    print('Base validation', report['base_token_nll'], flush=True)
    for step, start in enumerate(range(0, len(examples), args.batch), 1):
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(examples[start:start+args.batch])
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
        optimizer.step()
        if step % 10 == 0:
            print(f'step {step} / {(len(examples)+args.batch-1)//args.batch}, loss {loss.item():.3f}', flush=True)
    report['steps'] = step
    report['adapter_token_nll'] = evaluate()
    model.save_pretrained(args.output, safe_serialization=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, help='Train against the dequantized serving checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=6)
    train(parser.parse_args())
