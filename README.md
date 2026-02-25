# contexter — 임베디드 녹음기 연동 AI 회의록 자동화

프로젝트: contexter (기간: 2025.12 ~ 2026.01, 진행 중)

역할: AI Backend Engineer (팀 6명)

Tech: Python, Whisper, Sherpa-ONNX, LangChain, Docker, Ollama / Gemini

개요
- 경량화 STT(Whisper 기반)와 Sherpa-ONNX 화자 분리 파이프라인
- LangChain 기반 상황 인지형 텍스트 후처리(교정·요약), 로컬/클라우드 LLM 지원
- Docker로 패키징하여 재현 가능한 배포 환경 제공


- 비밀키는 `.env`로 관리하세요. 예시 파일은 `.env.example`에 있습니다. 실제 키는 절대 커밋하지 마세요.

간단 실행 예 (로컬)
```bash
# 예: Ollama 사용
python module/stt_lanch.py --audio_file ../data/interview_1.wav --provider ollama

# 또는 Gemini 사용 (환경변수로 키 제공)
export GEMINI_API_KEY=your_key
python module/stt_lanch.py --audio_file ../data/interview_1.wav --provider gemini
```

주요 파일
- `module/stt_lanch.py` — 음성인식·화자분리·LLM 처리 파이프라인 (CLI 포함)
- `module/requirements.txt` — Python 의존성
- `.env.example` — 환경변수 예시



