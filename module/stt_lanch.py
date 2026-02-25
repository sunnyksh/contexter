import os
import time
import sherpa_onnx
from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import soundfile as sf
from argparse import ArgumentParser
import os
import sys
import getpass
import numpy as np
import soxr

# --- [설정: 모델 경로 및 API 키] ---
# Docker 환경에서는 경로가 /app/models로 시작, 로컬에서는 ../models로 시작
MODEL_PATHS = {
    "whisper": "base", # 혹은 'large-v3'
    "sherpa_seg": os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "segmentation_model.onnx"),
    "sherpa_emb": os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "3dspeaker_embedding.onnx"),
}
#device = "cuda" if torch.cuda.is_available() else "cpu"
max_duration = 200.0  # 최대 오디오 길이 (초 단위), 필요시 조정 (argument로 오버라이드 가능)
# --- [함수 1: Whisper STT (텍스트 및 시간 추출)] ---
def run_whisper(audio_path, output_raw, prompt_subject=None, max_duration=200.0):
    print("\n[1/4] Whisper STT 시작...")
    model = WhisperModel(MODEL_PATHS["whisper"], device = "cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, 
                                   beam_size=1,
                                   language="ko",
                                   initial_prompt=f"이 내용은 {prompt_subject} 입니다. " \
                                      "전문 용어가 많이 포함되어 있으니 주의 깊게 들어주세요.",
                                    clip_timestamps=f"0,{max_duration}")

    results = []
    for s in segments:
        results.append({'start': s.start, 'end': s.end, 'text': s.text.strip()})
    with open(output_raw, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[{r['start']:.2f}s -> {r['end']:.2f}s] {r['text']}\n")
    return results

# --- [함수 2: Sherpa-ONNX 화자 분리] ---
def run_diarization(audio_path, num_speakers=0, max_duration=200.0):
    print("[2/4] Sherpa-ONNX 화자 분리 시작...")
    seg_model_path = os.path.abspath(MODEL_PATHS["sherpa_seg"])
    emb_model_path = os.path.abspath(MODEL_PATHS["sherpa_emb"])
    # 실제 모델 경로가 있어야 작동합니다.
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_model_path),
            num_threads=4, # 성능을 위해 스레드 추가
            debug=False),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model_path),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers), # 0이면 자동 감지
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)


    # ... 함수 내부 ...

    # 1. 파일 정보 읽기 (데이터 로드 없이 메타데이터만 확인)
    with sf.SoundFile(audio_path) as f:
        sr = f.samplerate
        # max_duration(초)을 샘플 수로 변환 (예: 200초 * 16000Hz)
        frames_to_read = int(max_duration * sr)
        
        # 2. 지정된 길이만큼만 읽기 (dtype='float32'로 지정해야 Sherpa-ONNX와 호환됨)
        data = f.read(frames=frames_to_read, dtype='float32')

    # 3. 모노(1채널) 변환 (2채널인 경우 평균내기)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # 4. 리샘플링 (만약 원본이 16000Hz가 아닐 경우)
    if sr != 16000:
        print(f"고속 리샘플링 중... {sr}Hz -> 16000Hz")
        data = soxr.resample(data, sr, 16000) # librosa보다 훨씬 빠름
    
    segments = sd.process(data) 
    return segments

# --- [함수 3: 시간 기반 병합 (Whisper + Sherpa)] ---
def merge_segments(whisper_data, diarization_data):
    print("[3/4] 데이터 병합 중...")
    # 1. 소스 코드 확인 결과: SortByStartTime()이 세그먼트 리스트를 반환합니다.
    # 메서드 이름이 파이썬에서는 스네이크 케이스(sort_by_start_time)일 것입니다.
    segments_list = diarization_data.sort_by_start_time()
    
    merged_text = ""
    last_speaker = None
    
    for w in whisper_data:
        mid_time = (w['start'] + w['end']) / 2
        speaker = "Unknown"
        
        # 2. 반환받은 리스트를 순회합니다.
        for d in segments_list:
            # C++ 소스 기준 Start(), End(), Speaker() 메서드 혹은 속성 접근
            if d.start <= mid_time <= d.end:
                speaker = f"화자 {d.speaker}"
                break
        
        if speaker != last_speaker:
            merged_text += f"\n\n[{speaker}]: {w['text']}"
            last_speaker = speaker
        else:
            merged_text += f" {w['text']}"
    
    return merged_text.strip()


# --- [함수 4: LangChain 처리 (교정 및 요약)] ---
def run_langchain_logic(full_text, mode="", provider="ollama", api_key=None, prompt_subject=None):
    print(f"[4/4] LangChain {provider} 분석 시작 (모드: {mode})...")
    
    # 1. 모델 설정
    if provider == "ollama":
        llm = ChatOllama(model="llama3.2",
                          temperature=0.3
                          )
        use_split = True
    else:
        # Support GEMINI API key and model via environment variables as fallback
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        gemini_key = api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("Gemini API key not provided. Set `api_key` argument or GEMINI_API_KEY environment variable.")
        llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=gemini_key)
        use_split = False

    # 2. 모드별 프롬프트 정의
    # Prompt templates use a {input} placeholder. If prompt_subject is provided
    # we prefix each template with topic information.
    prompts = {
        "seminar": {
            "refine": "세미나 강연록입니다. 전문 용어를 교정하고 문장을 매끄럽게 다듬으세요:\n{input}",
            "summary": "세미나 강연록입니다. 세미나의 핵심 인사이트를 5줄로 요약하세요:\n{input}"
        },
        "interview": {
            "refine": "인터뷰 녹취록입니다. 화자 간의 대화 맥락을 살려 오타를 수정하세요:\n{input}",
            "summary": "인터뷰 녹취록입니다. 질문과 답변의 핵심 내용을 바탕으로 인터뷰 내용을 요약하세요:\n{input}"
        }
    }

    if prompt_subject:
        topic_prefix = f"주제: {prompt_subject}.\n"
        for m in prompts:
            prompts[m]["refine"] = topic_prefix + prompts[m]["refine"]
            prompts[m]["summary"] = topic_prefix + prompts[m]["summary"]

    # 3. 교정(Refine) - 로컬 모델일 경우 분할 처리
    refined_text = ""
    if use_split:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            print(f"   > 조각 처리 중 ({i+1}/{len(chunks)})")
            res = llm.invoke(prompts[mode]["refine"].format(input=chunk))
            refined_text += res.content + "\n"
    else:
        res = llm.invoke(prompts[mode]["refine"].format(input=full_text))
        refined_text = res.content

    # 4. 최종 요약
    summary = llm.invoke(prompts[mode]["summary"].format(input=refined_text))
    
    return refined_text, summary.content

# --- [메인 실행부] ---
def final_pipeline(audio_file, mode="seminar", provider="ollama", api_key=None, prompt_subject=None, num_speakers=0, max_duration=200.0, output_raw="raw_text.txt", base_filename=None, result_dir="result"):
    start_time = time.time()
    
    # 입력 파일명에서 확장자 제거하여 기본 파일명 생성 (또는 사용자 지정값 사용)
    if base_filename is None:
        base_filename = os.path.splitext(os.path.basename(audio_file))[0]
    
    # result 폴더 경로 설정 및 생성
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 각 단계별 출력 파일명 자동 생성 (result 폴더에 저장)
    step1_output = os.path.join(result_dir, f"{base_filename}_step1_whisper.txt")
    step2_output = os.path.join(result_dir, f"{base_filename}_step2_diarization.txt")
    step3_output = os.path.join(result_dir, f"{base_filename}_step3_merged.txt")
    step4_refined_output = os.path.join(result_dir, f"{base_filename}_step4_refined.txt")
    step4_summary_output = os.path.join(result_dir, f"{base_filename}_step4_summary.txt")
    
    # 1. STT
    whisper_res = run_whisper(audio_file, step1_output, prompt_subject, max_duration=max_duration)
    print(f"   ✓ 저장: {step1_output}")
    
    # 2. 화자 분리 Sherpa 실행
    diarization_res = run_diarization(audio_file, num_speakers=num_speakers, max_duration=max_duration)
    # 화자 분리 결과 저장
    segments_list = diarization_res.sort_by_start_time()
    with open(step2_output, "w", encoding="utf-8") as f:
        for d in segments_list:
            f.write(f"[{d.start:.2f}s -> {d.end:.2f}s] 화자 {d.speaker}\n")
    print(f"   ✓ 저장: {step2_output}")
    
    # 3. 합치기
    formatted_text = merge_segments(whisper_res, diarization_res)
    with open(step3_output, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    print(f"   ✓ 저장: {step3_output}")
    
    # 4. LLM 처리
    refined, summary = run_langchain_logic(formatted_text, mode, provider, api_key, prompt_subject=prompt_subject)
    
    # 4단계 결과 저장
    with open(step4_refined_output, "w", encoding="utf-8") as f:
        f.write(refined)
    with open(step4_summary_output, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"   ✓ 저장: {step4_refined_output}")
    print(f"   ✓ 저장: {step4_summary_output}")
    
    print(f"\n✨ 전체 공정 완료! (총 소요시간: {time.time() - start_time:.1f}초)")
    return refined, summary

# 테스트 실행 시
# final_pipeline("test_audio.wav", mode="interview", provider="gemini", api_key="YOUR_KEY")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio_file", "-f",type=str, default="../data/interview_1.wav",help ="오디오 파일 경로")
    parser.add_argument("--mode", "-m", type=str, choices=["seminar", "interview"], default="seminar", help="분석 모드 선택")
    parser.add_argument("--provider", "-p", type=str, choices=["ollama", "gemini"], default="ollama", help="LLM 제공자 선택")
    parser.add_argument("--api_key", "-k", type=str, default=None, help="LLM API 키 (gemini , etc 선택 시 필요)")
    parser.add_argument("--output_raw", "-o", type=str, default=None, help="원본 텍스트 저장 파일명 (기본값: {입력파일명}_step1_whisper.txt)")
    parser.add_argument("--output_refined", "-r", type=str, default=None, help="교정된 텍스트 저장 파일명 (기본값: {입력파일명}_step4_refined.txt)")
    parser.add_argument("--output_summary", "-s", type=str, default=None, help="요약 저장 파일명 (기본값: {입력파일명}_step4_summary.txt)")   
    parser.add_argument("--object", "-a", type=str,  help="prompt 주제")
    parser.add_argument("--num_speakers", "-n", type=int, default=0, help="화자 수 (기본값: 0 = 자동 탐지)")
    parser.add_argument("--max_duration", "-t", type=float, default=200.0, help="최대 오디오 길이 (초 단위, 기본값: 200.0 -- filetest중 200초까지, 0 = 전체)")
    parser.add_argument("--base_filename", "-b", type=str, default=None, help="저장 파일의 기본 파일명 (기본값: 입력 오디오 파일명)")
    parser.add_argument("--result_dir", "-d", type=str, default="../result", help="결과 저장 폴더 경로 (기본값: result)")
    args = parser.parse_args()

    # window path
    abs_path = os.path.abspath(args.audio_file)
    audio_path = os.path.normpath(abs_path)

    # Determine effective API key: prefer explicit arg, then env var, then interactive prompt
    effective_api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if args.provider == "gemini" and not effective_api_key:
        if sys.stdin.isatty():
            try:
                effective_api_key = getpass.getpass("Enter Gemini API key (input hidden): ")
            except Exception:
                effective_api_key = None
        if not effective_api_key:
            print("Error: Gemini provider selected but no API key provided. Set GEMINI_API_KEY or use --api_key.")
            sys.exit(1)

    refined_text, summary = final_pipeline(
        args.audio_file,
        args.mode,
        args.provider,
        effective_api_key,
        prompt_subject=args.object,
        num_speakers=args.num_speakers,
        max_duration=args.max_duration if args.max_duration > 0 else 999999,
        base_filename=args.base_filename,
        result_dir=args.result_dir,
    )
    
    print("\n[교정된 텍스트]")
    print(refined_text)
    print("\n[요약]")
    print(summary)            