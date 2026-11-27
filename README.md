# yt-db-pipeline

직접 수집한 YouTube 영상에서 장면 그래프를 추출하고, Gemini(교사)의 출력으로 Gemma 4 E2B(학생)를 LoRA 학습하는 파이프라인입니다. HGVRec의 추출 코드를 바탕으로 만들었습니다.

| 단계 | 하는 일 | 명령 |
| --- | --- | --- |
| 1. Preparation | 영상 목록 → manifest, 영상 단위 분할, 다운로드, 30초 창마다 키프레임 6장(640×360, 0·5·10…25초) | `python -m preparation …` |
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

- 확인한 조합: torch 2.13.0+cu130, transformers 5.19.0, vllm 0.32.0, yt-dlp 2026.10.26, google-genai 3.6.1, accelerate 1.16.0.
- LoRA 학습은 별도 환경에 [requirements-finetune.txt](requirements-finetune.txt)를 설치해 실행합니다.
- Gemini는 `gcloud auth application-default login`으로 인증합니다.
- IAB 분류(`extract-iab-gt`), 세그먼트 분류(`extract-segment-gt`), UMD 분류(`classify-umd-categories`)만 Gemini API 키를 씁니다. Decisions API가 Vertex AI에 없기 때문입니다. Cloud Console의 사용자 인증 정보에서 API 제한을 "Gemini API"로 둔 키를 만들어 `config.yaml`의 `extraction.iab.api_key`에 넣습니다.

모든 명령은 저장소 루트에서 `PYTHONPATH=src`로 실행합니다(`pip install -e .`을 했다면 생략).

## 설정

```bash
cp config.example.yaml config.yaml
```

| 키 | 내용 |
| --- | --- |
| `preparation.video_list` | `Category_001.mp4 https://…` 줄로 된 TXT 목록. 번호는 카테고리마다 001부터 세고, `[Category]` 헤더는 없어도 된다. 예전 `001_Category001 https://…` 형식도 읽는다 |
| `preparation.data_root` | 내려받은 영상(`videos/`)과 메타데이터(`metadata/`)를 둘 위치 |
| `preparation.download` | 선호 해상도(`height`, 기본 360), 쿠키 파일 경로, 대기 시간 |
| `preparation.umd` | UMD 분류용. `classes`(분류표 CSV), `max_frames`(영상당 보낼 키프레임 수, 기본 50), `low_confidence`(경고 기준, 기본 0.3), `threads`(동시에 분류할 영상 수, 기본 8). API 키와 `model_id`는 `extraction.iab`를 같이 씀 |
| `preparation.download.proxy`, `ca_cert_file`, `no_check_certificates` | 사내망용. 프록시 주소와 프록시의 CA 인증서를 주거나, 인증서 검사를 끕니다(`SSL: CERTIFICATE_VERIFY_FAILED`가 날 때) |
| `extraction.models.<이름>` | `--model`로 고르는 생성 모델. `backend: vllm`(로컬 체크포인트 `path`) 또는 `backend: gemini` |
| `extraction.models.<이름>.mm_processor_kwargs` | vLLM 모델의 이미지 처리 인자. Gemma는 `max_soft_tokens: 140`(키프레임당 120토큰) |
| `extraction.models.<이름>.scene_max_new_tokens` | 모델별 장면당 출력 토큰 한도 |
| `extraction.models.<이름>.threads` | Gemini 모델의 동시 요청 수(기본 32) |
| `scoring.d3_path` | d3 Judge 체크포인트 경로. `--d3-server`만 쓰면 없어도 됩니다 |
| `evaluation.bge_path` | 교사 일치도에 쓰는 BGE 인코더 경로 |
| `extraction.iab` | IAB·세그먼트·UMD 분류용. `api_key`(Gemini API 키, `config.yaml`에만 둠), `model_id`(기본 `gemini-decision-1`), `threads`(동시에 분류할 장면 수, 기본 16) |

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
# (선택) 영상별 UMD 소분류. 키프레임을 최대 50장 골라 Gemini Decisions API에 보냄
python -m preparation classify-umd-categories

# 2. Extraction: 교사는 전체, 학생은 val (프롬프트 개선용)
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemini
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b --split val

# (선택) 장면별 IAB Content Taxonomy 3.1 라벨. 메인 파이프라인과 무관하며 --split을 빼면 전체
python -m extraction extract-iab-gt --run-id $RUN_ID --taxonomy data/content_taxonomy_3.1_converted.csv
# (선택) 장면별 TV 장르 세그먼트 라벨. 설정은 extraction.iab를 같이 씀
python -m extraction extract-segment-gt --run-id $RUN_ID --segments data/audience_collectives_segment_repository.csv

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

# 6. 최종 평가 (test): config.yaml의 extraction.models에 gemma4_e2b_lora(backend: vllm, 병합 모델 path)를 추가한 뒤
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b --split test
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b_lora --split test
python -m scoring score --run-id $RUN_ID --model gemma4_e2b --split test $D3
python -m scoring score --run-id $RUN_ID --model gemma4_e2b_lora --split test $D3
python -m evaluation compare --run-id $RUN_ID --students gemma4_e2b gemma4_e2b_lora --split test
```

- 모든 단계는 다시 실행하면 끝난 부분을 건너뜁니다. 처음부터 다시 하려면 `--force`를 씁니다.
- `--split train|val|test`와 `--limit N`으로 대상 영상을 줄일 수 있습니다. 키프레임이 준비되지 않은 영상은 건너뜁니다.
- `extract-iab-gt`는 장면의 키프레임을 Gemini Decisions API에 보내 IAB 카테고리를 고릅니다. 한 질문의 선택지가 255개로 제한되므로 Tier 1(37개)에서 시작해 고른 카테고리의 하위 항목으로 한 단계씩 내려가고, 맞는 하위 항목이 없다는 답이 나오거나 더 내려갈 곳이 없으면 멈춥니다(장면당 1~4회 호출). 장면 그래프를 쓰지 않으므로 `extract-graph-scenes`보다 먼저 돌려도 됩니다. 분류표나 `model_id`가 바뀌면 기존 라벨을 지우고 다시 만듭니다.
- `extract-segment-gt`는 같은 방식으로 세그먼트 표의 Segment Name 가운데 하나를 고릅니다. 표의 세그먼트 전부(31개)와 `none`을 한 질문의 선택지로 내므로 장면당 1회 호출합니다. 일반 장르와 더 구체적인 장르가 모두 맞으면(`Sports`와 `Baseball`, `Drama`와 `Crime Drama`) 구체적인 쪽을, 맞는 장르가 없으면 `none`을 고르게 합니다. 선택지 문구는 Segment Name이고, Audience Definition의 "who watch X contents"가 이름과 다르면 `Action (action or adventure)`처럼 X를 붙입니다. 표나 `model_id`가 바뀌면 기존 라벨을 지우고 다시 만듭니다.
- 키프레임은 30초 창을 6등분한 각 구간의 시작(0, 5, 10, …, 25초)에서 뽑습니다. 영상 끝에서 잘린 마지막 창도 같은 간격입니다(65초 영상이면 마지막 창은 60초 한 장). 예전에는 구간 가운데(2.5, 7.5, …초)였습니다. 타임스탬프 JSON이 달라지므로 다음 `extract-keyframes`에서 모든 영상이 자동으로 다시 추출되고, 예전 PNG(`0002_5.png` 등)는 지워지지 않고 남습니다. 어느 단계도 그 파일을 읽지 않으니 필요 없으면 직접 지우세요. 예전 키프레임으로 만든 그래프·점수·라벨이 있는 Run은 새 키프레임과 섞이지 않도록 새 `RUN_ID`로 추출부터 다시 돌립니다.
- `classify-umd-categories`(선택)는 영상의 키프레임 전체에서 처음과 끝을 포함해 균등 간격으로 최대 `max_frames`장(기본 50)을 골라, 한 요청에 Gemini Decisions API로 보냅니다. 50장이면 4분 10초까지는 5초 간격 그대로이고, 그보다 긴 영상은 간격이 넓어집니다. 질문은 하나이고, 선택지는 분류표의 소분류 232개 전부와 `none`입니다. 이름의 대분류(`Sports_001`의 `Sports`)는 모델에 주지 않으며, 선택지 문구는 `Law (Action and Thriller)`처럼 소속 대분류를 괄호로 붙입니다. 키프레임은 디스크의 PNG를 그대로 보냅니다. 분류표, `model_id`, `max_frames`가 바뀌면 기존 라벨을 지우고 다시 만듭니다. 키프레임이 없는 영상은 건너뛰고, 실패한 영상만 다음 실행에서 다시 묻습니다. 다른 단계는 이 결과를 쓰지 않습니다.
- `splits.csv`가 있으면 기존 배정은 바뀌지 않고 새 영상만 배정됩니다.
- 분할은 용도를 나눠 씁니다. val은 프롬프트 개선, train·val은 LoRA 학습, test는 최종 평가에만 씁니다. 프롬프트를 고칠 때 test 결과를 보지 않습니다.
- `$D3`는 d3 채점 방식입니다. `D3="--d3-server http://localhost:8899"`이면 vLLM Semantic Router 서버에 질문을 보내고, `D3=""`이면 GPU마다 d3를 직접 올립니다. 서버는 다른 터미널에서 먼저 띄웁니다: `vllm-sr serve vllm-sr/d3-mini -e -dp 4 --device-ids 0,1,2,3 --startup-timeout 7200`. 같은 d3 가중치이므로 두 방식은 채점 결과를 서로 재사용합니다(서버의 요청 배치로 확률이 거의 같은 판정 일부가 바뀔 수 있습니다).
- `train_lora.sh`는 GPU마다 학습 프로세스를 하나씩 띄웁니다(DDP). 쓸 GPU는 `--gpus 0,1`로 정하고, 빼면 `CUDA_VISIBLE_DEVICES`, 그것도 없으면 전체 GPU를 씁니다. vllm-sr이 쓰는 GPU는 빼고 지정하세요. GPU 수와 관계없이 전체 batch는 16입니다.
- `scoring report`와 `evaluation compare`의 `--split`은 그 분할의 영상만 집계합니다. 빼면 채점한 장면 전체를 합칩니다.
- `evaluation compare` 보고서의 "medium·format 정확도"는 교사의 medium·format을 정답으로 본 장면 단위 정확도와 macro-F1, 값별 F1, 자주 틀리는 조합(교사 → 학생)입니다. 정확도는 값이 그대로 같은 비율이고, 값별 F1과 macro-F1은 프롬프트 목록의 값으로만 계산합니다(목록 밖의 값은 `(목록 밖)` 한 줄로 묶음). 학생 그래프가 없는 장면은 오답으로 셉니다. "교사 일치도" 표의 medium·format 일치는 같은 값을 영상별로 평균한 것입니다.

## 산출물

```text
<data_root>/videos/<content_id>.mp4, metadata/<content_id>.json
artifacts/preparation/manifest.csv, splits.csv
artifacts/preparation/resized_keyframes/640_360/<content_id>/<SSSS>.png   # 초 4자리, 소수가 있으면 <SSSS>_<십분의 일초>.png
artifacts/preparation/source_assets/<content_id>/timestamp_fixed_30s.json
artifacts/preparation/umd_categories/labels.jsonl, labels.csv
artifacts/runs/<RUN_ID>/extraction/graph_<모델>/<content_id>.jsonl
artifacts/runs/<RUN_ID>/extraction/iab_gt/<content_id>.jsonl
artifacts/runs/<RUN_ID>/extraction/segment_gt/<content_id>.jsonl
artifacts/runs/<RUN_ID>/scoring/graph_<모델>/<content_id>.jsonl
artifacts/runs/<RUN_ID>/sft/{train,val}.jsonl
artifacts/runs/<RUN_ID>/reports/*.md
```

장면 행은 `content_id, scene_idx, tokens, warning, scene_graph`입니다. `scene_graph`는 `medium, format, topics, entities[{id,name,attributes}], actions[{actor,action,target,tool,place}]`이고, 파싱에 실패하면 원문 텍스트가 `PARSE_FAILED` 태그와 함께 들어갑니다.

IAB 라벨 행은 `content_id, scene_idx, iab_id, iab_name, iab_path, confidence, tiers[{tier,id,name,confidence,top}]`입니다. `iab_id`는 분류표의 Unique ID, `iab_path`는 Tier 1부터의 이름, `confidence`는 마지막으로 고른 Tier의 확률, `top`은 그 Tier에서 확률이 높은 3개입니다. 하위 항목이 없다는 답으로 멈췄으면 그 확률이 `stop_confidence`에 들어가고, 실패한 장면은 `error`만 가집니다.

UMD 라벨 행(`labels.jsonl`, 영상당 한 행)은 `content_id, query_category, pred_umd_categories, warning, frames_total, frames_used`입니다. `query_category`는 영상 이름에서 온 대분류, `pred_umd_categories`는 확률이 높은 순으로 소분류(UMD category) 상위 5개와 그 확률(`none`도 들어갈 수 있음), `warning`은 가장 높은 확률이 `low_confidence`(기본 0.3) 미만이면 `LOW_CONFIDENCE`, 아니면 `null`입니다. `frames_total`은 영상의 키프레임 수, `frames_used`는 실제로 보낸 수입니다. 실패한 영상은 `error`를 가지고 `pred_umd_categories`가 `null`입니다. `labels.csv`는 같은 내용을 `Baseball:0.62; Softball:0.21; …`처럼 한 칸에 적은 검토용 파일입니다. `warning`은 실행할 때마다 지금의 기준으로 다시 계산하므로, 기준을 바꿔도 모델에 다시 묻지 않습니다.

세그먼트 라벨 행은 `content_id, scene_idx, segment_id, segment, confidence, probabilities`입니다. `segment`는 고른 Segment Name, `segment_id`는 그 이름을 소문자와 밑줄로 바꾼 선택지 id, `confidence`는 고른 선택지의 확률, `probabilities`는 `none`을 포함한 모든 선택지의 확률(높은 순)입니다. `none`을 골랐으면 `segment_id`와 `segment`가 `null`이고, 실패한 장면은 `error`만 가집니다.

## Scoring이 재는 것

| 단위 | 주장 | 질문 형태 | 점수 |
| --- | --- | --- | --- |
| Context | medium, format | 프롬프트의 값 중 선택 | 그래프가 고른 값의 확률 |
| Context | topic | 없음 / 부수적 / 주제 척도 | 기대값 ÷ 2 |
| Entity | 엔티티 하나 (속성 포함) | 지지 / 불일치 / 판단 불가 | "지지" 확률 |
| Action | 행동 하나 (actor, target, tool, place 포함) | 지지 / 불일치 / 판단 불가 | "지지" 확률 × actor 엔티티 점수 |

- d3는 질문마다 키프레임을 다시 읽으므로 채점 시간은 질문 수에 비례합니다. 그래서 속성은 엔티티 질문에, 역할은 행동 질문에 합쳐 묻습니다. 속성이나 역할 하나가 틀리면 그 엔티티나 행동 전체의 점수가 낮아집니다.
- 역할이 `none`이거나 `unknown`이면 행동 질문에 넣지 않습니다.
- 다른 영상에서 가져온 미끼 엔티티(장면당 1개)로 Judge의 오탐률을 함께 잽니다.
- 재현율은 비교한 모델들의 검증된 주장을 모은 합집합을 기준으로 합니다. 사람이 만든 정답이 아닙니다.

## d3에게 주는 질문

질문 문장은 [questions.py](src/scoring/questions.py), 주장을 나누는 방식은 [claims.py](src/scoring/claims.py)에 있습니다. 질문 문장이나 선택지를 바꾸면 이 절도 함께 고쳐 주세요.

### 한 장면의 요청

장면 하나가 d3 요청 하나입니다. 요청에는 키프레임 6장, 아래 상태 문장, 그 장면의 질문 전부가 들어갑니다. d3는 질문마다 키프레임과 상태 문장을 함께 읽고, 선택지마다 확률을 돌려줍니다.

```text
The images are chronological keyframes sampled from one 30-second segment of a video. Judge only what is visible in these keyframes.
```

### 질문 종류

| 질문 | 개수 | 문장 | 선택지 | 점수 |
| --- | --- | --- | --- | --- |
| medium | 1 | `What is the medium of this segment?` | `live_action`, `animation`, `gameplay`, `screen_recording`, `mixed`, `unknown` | 그래프가 고른 값의 확률 |
| format | 1 | `What is the presentation format of this segment?` | `demonstration`, `explanation`, `review`, `narrative`, `highlights`, `performance`, `interview`, `other`, `unknown` | 그래프가 고른 값의 확률 |
| topic | topic마다 | `How central is this topic to the segment: "{topic}"?` | 3단계 척도: `Not shown in the segment` / `Shown but incidental` / `A main subject of the segment` | 기대값 ÷ 2 (0, 0.5, 1) |
| 엔티티 | 엔티티마다 | 속성이 없으면 `Is this visible in at least one keyframe: {이름}?`<br>속성이 있으면 `Is this visible in at least one keyframe, matching the whole description: {이름} ({속성}; {속성})?` | 판정 3개 | `supported` 확률 |
| 행동 | 행동마다 | 아래 여러 줄 문장 | 판정 3개 | `supported` 확률 × actor 엔티티 점수 |
| 미끼 엔티티 | 장면당 1 | 엔티티 질문과 같은 문장. 이름은 다른 영상의 엔티티에서 뽑음 | 판정 3개 | 점수에 넣지 않고 오탐률로만 집계 |

행동 질문은 그래프가 엔티티를 지정한 역할만 줄로 붙입니다. 역할이 `none`이거나 `unknown`이면 그 줄을 넣지 않습니다.

```text
Do the keyframes show this action with every listed detail correct?
Actor: {actor 이름 (속성; 속성)}
Action: "{행동}"
Acted on: {target}
Tool or means: {tool}
Place: {place}
```

**판정 3개**는 엔티티, 행동, 미끼 질문에서 같습니다.

| 키 | 선택지 설명 |
| --- | --- |
| `supported` | `Yes, the keyframes show this` |
| `contradicted` | `No, the keyframes show this is wrong or absent` |
| `cannot_tell` | `The keyframes do not show enough to decide` |

- 점수는 `supported` 확률을 씁니다. `cannot_tell` 확률은 기권 확률로 따로 보고합니다.
- medium과 format의 선택지에는 각 값의 설명(예: `live_action`: `Real people, objects or places filmed with a camera`)이 함께 들어갑니다. 설명 문장은 [claims.py](src/scoring/claims.py)의 `MEDIA`, `FORMATS`에 있습니다.
- medium이나 format이 목록에 없는 값이면 질문하지 않고 0점을 줍니다.
- 엔티티와 actor는 `이름 (속성; 속성)` 형식으로 적어 같은 이름의 엔티티를 구분합니다.

### 예시

아래 그래프 하나에서 질문 10개가 만들어집니다(그래프 주장 9개 + 미끼 1개).

```text
[Context]
medium: live_action
format: demonstration
topics: salmon sashimi; knife work
[Entities]
actor_1: woman; striped apron
target_1: salmon fillet; raw
tool_1: long knife
[Actions]
actor_1 - slicing thinly - target_1; tool_1; unknown
actor_1 - plating - target_1; none; unknown
```

| 주장 ID | 질문 |
| --- | --- |
| `medium` | `What is the medium of this segment?` (정답 후보 `live_action`) |
| `format` | `What is the presentation format of this segment?` (정답 후보 `demonstration`) |
| `topic:0` | `How central is this topic to the segment: "salmon sashimi"?` |
| `topic:1` | `How central is this topic to the segment: "knife work"?` |
| `entity:actor_1` | `Is this visible in at least one keyframe, matching the whole description: woman (striped apron)?` |
| `entity:target_1` | `Is this visible in at least one keyframe, matching the whole description: salmon fillet (raw)?` |
| `entity:tool_1` | `Is this visible in at least one keyframe: long knife?` |
| `action:0` | `Do the keyframes show this action with every listed detail correct?`<br>`Actor: woman (striped apron)`<br>`Action: "slicing thinly"`<br>`Acted on: salmon fillet (raw)`<br>`Tool or means: long knife` |
| `action:1` | `Do the keyframes show this action with every listed detail correct?`<br>`Actor: woman (striped apron)`<br>`Action: "plating"`<br>`Acted on: salmon fillet (raw)` |
| `foil:entity:0` | `Is this visible in at least one keyframe: elephant?` (다른 영상에서 뽑은 이름) |

`action:0`의 place는 `unknown`, `action:1`의 tool은 `none`이라 해당 줄이 빠졌습니다.

### 재현율 질문 (`scoring coverage`)

재현율은 키프레임을 쓰지 않습니다. 대상 모델의 그래프를 텍스트로 상태에 넣고, 기준 항목마다 질문합니다.

- 상태: `This is a scene graph extracted from one 30-second video segment:` 다음 줄부터 그래프 텍스트(ID 형식)
- 질문: `Does the graph include {항목}? Count the same thing worded differently as included.`
  - `{항목}`은 `the entity: {이름}`, `the topic: {topic}`, `the action: {actor 이름} {행동}` 중 하나입니다.
- 선택지: `covered` (`Yes, the graph contains it or a clear equivalent`) / `missing` (`No, the graph does not contain it`)
- 점수: `covered` 확률

## 프롬프트

파서는 [scene_graph_v8_2.md](prompts/scene_graph_v8_2.md)(ID 형식)와 [scene_graph_v9_base.md](prompts/scene_graph_v9_base.md)(이름 형식)를 기준으로 합니다. 프롬프트를 바꾸면 [prompts/update_note.md](prompts/update_note.md)를 함께 갱신합니다. 학습과 추론은 같은 프롬프트와 같은 `max_soft_tokens`를 써야 하므로, `build-dataset`은 교사 추출에 쓴 프롬프트와 다르면 중단합니다.

## 테스트

```bash
python -m pytest
ruff check src tests
```
