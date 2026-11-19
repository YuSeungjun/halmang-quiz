# 하나라봉

제주 할망과 함께 생활 속 제주어를 익히는 10문제 낱말 퀴즈와 제주어·표준어 문장 번역기입니다.

**바로 체험하려면** [하나라봉.html 다운로드](https://github.com/YuSeungjun/halmang-quiz/raw/refs/heads/main/hanarabong/하나라봉.html)를 눌러 파일을 저장한 뒤 브라우저에서 여세요. 퀴즈 56문항과 문장 음성 53개, 기본 낱말과 수록 예문의 양방향 번역이 파일에 들어 있습니다.

## 내 컴퓨터에서 퀴즈 서버 실행하기

Python 3.10 이상에서 저장소 루트를 기준으로 실행합니다.

```sh
python3 hanarabong/serve.py
```

브라우저에서 [하나라봉 열기](http://127.0.0.1:8766)를 선택하세요. 퀴즈와 수록된 문장 낭독은 GPU 없이 실행됩니다.

## AI 번역과 새 문장 낭독

자유 입력 번역에는 JIT 학습 문장쌍(`data/jit/`), AI Hub 71558 훈련 라벨(`data/aihub-71558/extracted/Training/labels/`), 로컬 Ollama의 Qwen3 32B가 필요합니다. 새 제주어 문장을 낭독하려면 VoxCPM2 실행 환경, AI Hub + JSS로 학습한 `step_0002362` 체크포인트, 목소리 03 참조 음성을 별도로 준비하세요. 해당 환경에서 `sh hanarabong/run.sh`를 실행하면 번역 색인을 만들고 서버를 시작합니다.

학습 서버의 문맥 검색은 Kiwi를 사용합니다. Qwen3 32B 제주어 어댑터의 학습 자료, 설치 버전과 양방향 평가 결과는 [번역 모델 실험 기록](hanarabong/TRANSLATION.md)에 있습니다.

퀴즈 규칙과 화면 검증 방법은 [학습자용 사이트 설명](hanarabong/README.md)에 있습니다.
