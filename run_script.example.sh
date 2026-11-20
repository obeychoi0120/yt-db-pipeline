#!/usr/bin/env bash
# 실행: conda activate vc_cloud && bash run_script.sh
# 돌릴 줄만 남기고 나머지는 앞에 #을 붙여 주석 처리하세요.
# 모든 단계는 다시 실행하면 끝난 부분을 건너뜁니다.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=src

RUN_ID=R01
PROMPT=prompts/scene_graph_v8_2.md

# 1. Preparation: manifest, 분할, 다운로드, 키프레임
python -m preparation create-manifest
python -m preparation make-split
python -m preparation download-videos
python -m preparation extract-keyframes

# 2. Extraction: 교사(Gemini)는 전체, 학생(Gemma)은 test 분할
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemini
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b --split test

# 3. Scoring: d3 채점, 재현율, 보고서
python -m scoring score --run-id $RUN_ID --model gemini
python -m scoring score --run-id $RUN_ID --model gemma4_e2b --split test
python -m scoring coverage --run-id $RUN_ID --models gemini gemma4_e2b --split test
python -m scoring report --run-id $RUN_ID --models gemini gemma4_e2b

# 5-1. LoRA 데이터: 점수 0.7 미만인 교사 장면은 제외
python -m finetuning build-dataset --run-id $RUN_ID --schema $PROMPT --min-scene-score 0.7
python -m finetuning model-view --output artifacts/models/gemma4_e2b_view

# 5-2. LoRA 학습과 병합 (이 두 줄은 conda activate gemma_ft 환경에서 실행)
MODEL_PATH=artifacts/models/gemma4_e2b_view 
TRAIN_DATA=artifacts/runs/$RUN_ID/sft/train.jsonl 
VAL_DATA=artifacts/runs/$RUN_ID/sft/val.jsonl 
bash src/finetuning/train_lora.sh artifacts/runs/$RUN_ID/lora 3
python -m finetuning merge --training-dir artifacts/runs/$RUN_ID/lora --output artifacts/models/gemma4_e2b_lora

# 6. 평가 (먼저 config.yaml의 models와 extraction.graph에 gemma4_e2b_lora를 추가)
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b_lora --split test
python -m scoring score --run-id $RUN_ID --model gemma4_e2b_lora --split test
python -m evaluation compare --run-id $RUN_ID --students gemma4_e2b gemma4_e2b_lora --split test