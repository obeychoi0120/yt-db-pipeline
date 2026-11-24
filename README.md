# yt-db-pipeline

직접 수집한 YouTube 영상에서 장면 그래프를 추출하고, Gemini(교사)의 출력으로 Gemma 4 E2B(학생)를 LoRA 학습하는 파이프라인입니다. HGVRec의 추출 코드를 바탕으로 만들었습니다.

| 단계 | 하는 일 | 명령 |
| --- | --- | --- |
| 1. Preparation | 영상 목록 → manifest, 영상 단위 분할, 다운로드, 30초 창마다 키프레임 6장(640×360) | `python -m preparation …` |
| 2. Extraction | Gemini(Vertex AI) 또는 로컬 vLLM 모델로 장면별 Context·Entities·Actions 그래프 생성 | `python -m extraction …` |
| 3. Scoring | 그래프를 주장 단위로 나눠 d3 Decision 모델이 키프레임과 대조 | `python -m scoring …` |
| 4. 프롬프트 개선 | 수동. 3단계 보고서를 근거로 `prompts/`를 고치고 `prompts/update_note.md`에 기록 | – |
| 5. LoRA | 교사 그래프로 SFT 데이터 생성, ms-swift 학습, best 체크포인트 병합 | `python -m finetuning …`, `src/finetuning/train_lora.sh` |
| 6. 평가 | 학습 전·후 학생과 교사를 d3 점수, 교사 일치도, 형식 건전성으로 비교 | `python -m evaluation …` |

## 설치

```bash
conda create -n vc_cloud python=3.12 -y
conda activate vc_cloud
conda install -c conda-forge ffmpeg -y
pip install -r requirements-scoring.txt   # requirements-main.txt 포함
```

- 확인한 조합: torch 2.13.0+cu130, transformers 5.19.0, vllm 0.32.0, yt-dlp 2026.10.26, google-genai 2.23.0, accelerate 1.16.0.
- LoRA 학습은 별도 환경에 [requirements-finetune.txt](requirements-finetune.txt)를 설치해 실행합니다.
- Gemini는 `gcloud auth application-default login`으로 인증합니다.

모든 명령은 저장소 루트에서 `PYTHONPATH=src`로 실행합니다(`pip install -e .`을 했다면 생략).

## 설정

```bash
cp config.example.yaml config.yaml
```

| 키 | 내용 |
| --- | --- |
| `preparation.video_list` | `[Category]` 헤더와 `001_Category001 https://…` 줄로 된 TXT 목록 |
| `preparation.data_root` | 내려받은 영상(`videos/`)과 메타데이터(`metadata/`)를 둘 위치 |
| `preparation.download` | 선호 해상도(`height`, 기본 360), 쿠키 파일 경로, 대기 시간 |
| `preparation.download.proxy`, `ca_cert_file`, `no_check_certificates` | 사내망용. 프록시 주소와 프록시의 CA 인증서를 주거나, 인증서 검사를 끕니다(`SSL: CERTIFICATE_VERIFY_FAILED`가 날 때) |
| `models.<이름>` | 생성 모델은 `backend: vllm`(로컬 체크포인트 `path`) 또는 `backend: gemini`. `d3`, `bge`는 `path`만 둡니다 |
| `models.<이름>.mm_processor_kwargs` | vLLM 모델의 이미지 처리 인자. Gemma는 `max_soft_tokens: 140`(키프레임당 120토큰) |
| `extraction.graph.<이름>.scene_max_new_tokens` | 모델별 출력 토큰 한도 |

`YT_DB_PIPELINE_CONFIG=/path/to/other.yaml`로 다른 설정 파일을 쓸 수 있습니다.

## 실행

```bash
export PYTHONPATH=src
RUN_ID=R01_v8_2
PROMPT=prompts/scene_graph_v8_2.md
D3="--d3-server http://localhost:8899"   # 서버 없이 GPU마다 d3를 올리려면 D3=""

# 1. Preparation (Run과 무관하게 한 번)
python -m preparation create-manifest
python -m preparation make-split
python -m preparation download-videos
python -m preparation extract-keyframes

# 2. Extraction: 교사는 전체, 학생은 val (프롬프트 개선용)
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemini
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b --split val

# 3. Scoring: 교사는 전체(5단계 점수 필터에 필요), 비교와 보고서는 val
python -m scoring score --run-id $RUN_ID --model gemini $D3
python -m scoring score --run-id $RUN_ID --model gemma4_e2b --split val $D3
python -m scoring coverage --run-id $RUN_ID --models gemini gemma4_e2b --split val $D3
python -m scoring report --run-id $RUN_ID --models gemini gemma4_e2b --split val

# 5. LoRA
python -m finetuning build-dataset --run-id $RUN_ID --schema $PROMPT --min-scene-score 0.7
python -m finetuning model-view --model gemma4_e2b --output /path/to/gemma4_e2b_view
conda activate gemma_ft
bash src/finetuning/train_lora.sh --model /path/to/gemma4_e2b_view \
    --train-data artifacts/runs/$RUN_ID/sft/train.jsonl --val-data artifacts/runs/$RUN_ID/sft/val.jsonl \
    --output-dir artifacts/runs/$RUN_ID/lora --epochs 3
python -m finetuning merge --training-dir artifacts/runs/$RUN_ID/lora --output /path/to/gemma4_e2b_lora
conda activate vc_cloud

# 6. 최종 평가 (test): config.yaml의 models에 gemma4_e2b_lora(backend: vllm, 병합 모델 path)를 추가한 뒤
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b --split test
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b_lora --split test
python -m scoring score --run-id $RUN_ID --model gemma4_e2b --split test $D3
python -m scoring score --run-id $RUN_ID --model gemma4_e2b_lora --split test $D3
python -m evaluation compare --run-id $RUN_ID --students gemma4_e2b gemma4_e2b_lora --split test
```

- 모든 단계는 다시 실행하면 끝난 부분을 건너뜁니다. 처음부터 다시 하려면 `--force`를 씁니다.
- `--split train|val|test`와 `--limit N`으로 대상 영상을 줄일 수 있습니다. 키프레임이 준비되지 않은 영상은 건너뜁니다.
- `splits.csv`가 있으면 기존 배정은 바뀌지 않고 새 영상만 배정됩니다.
- 분할은 용도를 나눠 씁니다. val은 프롬프트 개선, train·val은 LoRA 학습, test는 최종 평가에만 씁니다. 프롬프트를 고칠 때 test 결과를 보지 않습니다.
- `$D3`는 d3 채점 방식입니다. `D3="--d3-server http://localhost:8899"`이면 vLLM Semantic Router 서버에 질문을 보내고, `D3=""`이면 GPU마다 d3를 직접 올립니다. 서버는 다른 터미널에서 먼저 띄웁니다: `vllm-sr serve vllm-sr/d3-mini -e -dp 4 --device-ids 0,1,2,3 --startup-timeout 7200`. 같은 d3 가중치이므로 두 방식은 채점 결과를 서로 재사용합니다(서버의 요청 배치로 확률이 거의 같은 판정 일부가 바뀔 수 있습니다).
- `scoring report`와 `evaluation compare`의 `--split`은 그 분할의 영상만 집계합니다. 빼면 채점한 장면 전체를 합칩니다.

## 산출물

```text
<data_root>/videos/<content_id>.mp4, metadata/<content_id>.json
artifacts/preparation/manifest.csv, splits.csv
artifacts/preparation/resized_keyframes/640_360/<content_id>/<SSSS>.png
artifacts/preparation/source_assets/<content_id>/timestamp_fixed_30s.json
artifacts/runs/<RUN_ID>/extraction/graph_<모델>/<content_id>.jsonl
artifacts/runs/<RUN_ID>/scoring/graph_<모델>/<content_id>.jsonl
artifacts/runs/<RUN_ID>/sft/{train,val}.jsonl
artifacts/runs/<RUN_ID>/reports/*.md
```

장면 행은 `content_id, scene_idx, tokens, warning, scene_graph`입니다. `scene_graph`는 `medium, format, topics, entities[{id,name,attributes}], actions[{actor,action,target,tool,place}]`이고, 파싱에 실패하면 원문 텍스트가 `PARSE_FAILED` 태그와 함께 들어갑니다.

## Scoring이 재는 것

| 단위 | 주장 | 질문 형태 | 점수 |
| --- | --- | --- | --- |
| Context | medium, format | 프롬프트의 값 중 선택 | 그래프가 고른 값의 확률 |
| Context | topic | 없음 / 부수적 / 주제 척도 | 기대값 ÷ 2 |
| Entity | 존재, 속성 | 지지 / 불일치 / 판단 불가 | "지지" 확률 |
| Action | 행동 발생 | 지지 / 불일치 / 판단 불가 | "지지" 확률 |
| 역할 연결 | target, tool, place | 선언된 엔티티 + "해당 없음" 중 선택 | 그래프가 지정한 값의 확률 |

- 속성은 엔티티, 행동은 actor, 역할 연결은 행동의 점수를 곱해 반영합니다.
- 다른 영상에서 가져온 미끼 주장으로 Judge의 오탐률을 함께 잽니다.
- 재현율은 비교한 모델들의 검증된 주장을 모은 합집합을 기준으로 합니다. 사람이 만든 정답이 아닙니다.

## 프롬프트

파서는 [scene_graph_v8_2.md](prompts/scene_graph_v8_2.md)(ID 형식)와 [scene_graph_v9_base.md](prompts/scene_graph_v9_base.md)(이름 형식)를 기준으로 합니다. 프롬프트를 바꾸면 [prompts/update_note.md](prompts/update_note.md)를 함께 갱신합니다. 학습과 추론은 같은 프롬프트와 같은 `max_soft_tokens`를 써야 하므로, `build-dataset`은 교사 추출에 쓴 프롬프트와 다르면 중단합니다.

## 테스트

```bash
python -m pytest
ruff check src tests
```
