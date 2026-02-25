# Python 3.10 기반 이미지
FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치 (오디오 처리 라이브러리)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# models와 module 폴더만 복사
COPY models/ /app/models/
COPY module/ /app/module/

# requirements.txt 복사 및 설치
COPY module/requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# 작업 디렉토리 설정
WORKDIR /app/module

# Ollama 서버 주소 (docker-compose에서 'ollama'로 정의됨)
ENV OLLAMA_HOST=http://ollama:11434

# FastAPI 포트
EXPOSE 8000

# API 서버 실행
CMD ["uvicorn", "stt_api:app", "--host", "0.0.0.0", "--port", "8000"]
