# Qwen3 32B 제주어 번역 실험

하나라봉은 등록 낱말과 일치하는 문장쌍을 먼저 조회한다. 새 문장을 번역할 때 표준어 입력은 구절별로 나누고, Kiwi가 분석한 동사·형용사·명사와 과거·지속·부정·동행·제안·전언·높임 단서를 SQLite FTS로 검색한다. 후보 예문의 뜻과 문법 단서가 일치하는 순서로 짧은 목록을 만들며 유사한 중복 문장을 줄인다. ‘같이 갈래?’의 동사 ‘가다’와 ‘두 갈래’의 명사는 검색 키가 다르다. 제주어 입력에는 기존 원문 검색을 사용한다.

## 학습 자료와 쓰임

JIT 연구에서 사용한 양방향 병렬문 학습 방식을 참고했다. 원 논문은 별도의 인코더·디코더 Transformer를 학습했다. 이 실험에서는 서비스에서 쓰던 Qwen3 32B를 고정하고 마지막 24개 층의 query·value 투영에만 LoRA를 학습했다. 검토 요청은 기존 Qwen3 32B가 맡는다. [JIT 원 논문 5절](https://aclanthology.org/2020.lrec-1.318/) · [Qwen3 32B 모델 카드](https://huggingface.co/Qwen/Qwen3-32B)

첫 서비스 버전은 JIT 훈련 예문 420쌍, AI Hub 훈련 예문 420쌍, 제주특별자치도 생활방언 260쌍과 제주어학교의 짧은 대조 예문 6쌍을 사용했다. 가족·감정·부탁·반말 제안 같은 생활 대화를 우선 표집했다. 수업 용례는 질문과 진술의 문장부호를 유지해 양방향으로 학습했다. 1,106쌍에서 학습 입력 1,112개를 만들었다. 새로 의뢰한 제주어 화자의 감수는 이번 실험에 포함되지 않았다. [JIT 배포처](https://github.com/kakaobrain/jejueo) · [AI Hub 71558](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71558) · [제주어학교 음식점 회화](https://www.jejueo999.kr/index.php/contents/notice/class?act=view&bd_bcid=notice_class&page=&seq=717)

## 비교 결과

| 평가 | 방향 | 기본 Qwen3 32B | 제주어 어댑터 적용 |
| --- | --- | ---: | ---: |
| 정답 토큰당 음의 로그우도, 낮을수록 좋음 | 표준어 → 제주어 | 2.2634 | 1.4105 |
| 정답 토큰당 음의 로그우도 | 제주어 → 표준어 | 1.2810 | 0.7458 |
| chrF, 높을수록 참조문과 글자 배열이 비슷함 | 표준어 → 제주어 | 29.81 | 43.58 |
| chrF | 제주어 → 표준어 | 27.44 | 48.71 |

로그우도는 JIT 개발 세트 32쌍과 AI Hub 검증 대화 32쌍에서 번역문과 출력 JSON의 정답 토큰을 대상으로 측정했다. chrF는 표준어와 제주어 표기가 서로 다르고 훈련 색인과 수업 예문에도 없는 JIT 테스트 문장 24쌍을 고정해 양방향으로 측정했다. 두 모델에 같은 지시를 주고 검색과 자동 검토를 거치기 전의 번역을 비교했다. 표본이 작으며 구어 자연스러움은 별도의 화자 평가가 필요하다.

## 재실행

서비스 Python에 `kiwipiepy==0.22.2`와 `kiwipiepy-model==0.22.1`을 설치한다. `sh hanarabong/run.sh`가 시작 시 문맥 색인을 확인하고 원본 DB나 색인 버전이 달라지면 새 파일을 완성한 뒤 교체한다.

학습에는 PyTorch 2.8, Transformers 4.57.3, PEFT 0.17.1을 사용했다. 실제 실행에서는 Ollama의 `qwen3:32b` Q4_K_M 가중치를 BF16으로 풀어 GPU 한 장에서 학습했다. 재실행 시 같은 GGUF와 공식 토크나이저를 지정할 수 있다. 이 경로에는 `gguf==0.17.1`과 전체 가중치를 풀 수 있는 호스트 메모리가 필요하다. 동일한 가중치에서 변환한 Hugging Face 형식 BF16 디렉터리가 있으면 `--base`에 지정하고 `--gguf`를 생략할 수 있다. 학습률은 0.0001, rank는 16, alpha는 32, 배치 크기는 6이다. 학습 출력에는 새 디렉터리를 지정한다.

```sh
python hanarabong/train_translation.py --base "$QWEN_TOKENIZER_DIR" --gguf "$QWEN_BASE_GGUF" --output "$ADAPTER_DIR"
```

서빙 모델은 llama.cpp의 LoRA 변환기와 `llama-export-lora`로 어댑터를 원본 GGUF에 합친 후 등록했다. 변환에 사용한 llama.cpp 커밋은 `f8caa5ea30b5775512906ae3b7ee53b21a7a6e1d`다. 다음 명령은 프로젝트 루트에서 실행하며 디렉터리 변수에는 절대 경로를 사용한다. Ollama 실행 파일이 PATH에 있고 `qwen3:32b`가 등록되어 있어야 한다.

```sh
python "$LLAMA_CPP/convert_lora_to_gguf.py" --base "$QWEN_TOKENIZER_DIR" --outtype f16 --outfile "$ADAPTER_DIR/adapter.gguf" "$ADAPTER_DIR"
"$LLAMA_EXPORT_LORA" -m "$QWEN_BASE_GGUF" --lora "$ADAPTER_DIR/adapter.gguf" -o "$ADAPTER_DIR/qwen3-jeju.gguf"
export OLLAMA_HOST=127.0.0.1:18765
ollama show --modelfile qwen3:32b | sed 's|^FROM .*|FROM ./qwen3-jeju.gguf|' > "$ADAPTER_DIR/Modelfile"
ollama create hanarabong-qwen3:32b-jeju-v1 -f "$ADAPTER_DIR/Modelfile"
printf '%s\n' 'hanarabong-qwen3:32b-jeju-v1' > hanarabong/runtime/translation-model.txt
sh hanarabong/run.sh
```

`HANARABONG_TRANSLATION_MODEL`을 직접 지정하면 활성 모델 파일보다 우선한다. 둘 다 비어 있으면 기본 Qwen3 32B를 사용한다. 모델 선택은 서버 프로세스가 시작할 때 반영된다. chrF 비교에는 `sacrebleu==2.5.1`을 사용한다.

```sh
HANARABONG_TRANSLATION_MODEL=qwen3:32b python hanarabong/evaluate_translation.py --output hanarabong/runtime/eval-base.json
HANARABONG_TRANSLATION_MODEL=hanarabong-qwen3:32b-jeju-v1 python hanarabong/evaluate_translation.py --output hanarabong/runtime/eval-dialogue.json
```

어댑터와 평가 원문은 학습 서버의 `hanarabong/runtime/`에 저장한다. 실제 실행의 safetensors 어댑터 SHA-256은 `80e830d881a5890934697a641bbf1f610d63cc606194997aa76c012b184c21c3`이다.
