#!/usr/bin/env bash
# 실행: conda activate vc_cloud && bash run_script.sh
# 돌릴 줄만 남기고 나머지는 앞에 #을 붙여 주석 처리하세요.
# 모든 단계는 다시 실행하면 끝난 부분을 건너뜁니다.
#
# 분할 사용: val은 프롬프트 개선(2~4단계), train·val은 LoRA 학습(5단계),
# test는 최종 평가(6단계)에서만 봅니다. 프롬프트를 고칠 때 test 결과를 보지 마세요.
# set -e
cd "$(dirname "$0")"
export PYTHONPATH=src

RUN_ID=R01
PROMPT=prompts/scene_graph_v8_2.md
# d3 채점 방식: vllm-sr 서버를 먼저 띄웠으면 첫 줄, 서버 없이 GPU마다 모델을 직접 올리려면 둘째 줄
#   서버 실행 (다른 터미널): vllm-sr serve vllm-sr/d3-mini -e -dp 4 --device-ids 0,1,2,3 --startup-timeout 7200
D3="--d3-server http://localhost:8899"
# D3=""

# 1. Preparation: manifest, 분할, 다운로드, 키프레임
python -m preparation create-manifest
python -m preparation make-split
python -m preparation download-videos
python -m preparation extract-keyframes

# 2. Extraction: 교사(Gemini)는 전체(학습 정답과 평가 기준), 학생(Gemma)은 val
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemini
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b --split val

# 3. Scoring: 교사는 전체(5단계 점수 필터에 필요), 비교와 보고서는 val
python -m scoring score --run-id $RUN_ID --model gemini $D3
python -m scoring score --run-id $RUN_ID --model gemma4_e2b --split val $D3
python -m scoring coverage --run-id $RUN_ID --models gemini gemma4_e2b --split val $D3
python -m scoring report --run-id $RUN_ID --models gemini gemma4_e2b --split val

# 4. 프롬프트 개선: 3단계 val 보고서를 보고 수동으로 진행. 고친 프롬프트는 새 RUN_ID로 2단계부터 다시

# 5-1. LoRA 데이터: train·val의 교사 장면 중 점수 0.7 미만은 제외
python -m finetuning build-dataset --run-id $RUN_ID --schema $PROMPT --min-scene-score 0.7
python -m finetuning model-view --output artifacts/models/gemma4_e2b_view

# 5-2. LoRA 학습과 병합 (이 부분은 conda activate gemma_ft 환경에서 실행)
MODEL_VIEW=artifacts/models/gemma4_e2b_view
TRAIN_DATA=artifacts/runs/$RUN_ID/sft/train.jsonl
VAL_DATA=artifacts/runs/$RUN_ID/sft/val.jsonl
LORA_DIR=artifacts/runs/$RUN_ID/lora
bash src/finetuning/train_lora.sh --model $MODEL_VIEW --train-data $TRAIN_DATA --val-data $VAL_DATA --output-dir $LORA_DIR --epochs 3
python -m finetuning merge --training-dir $LORA_DIR --output artifacts/models/gemma4_e2b_lora

# 6. 최종 평가 (test): 먼저 config.yaml의 models와 extraction.graph에 gemma4_e2b_lora를 추가
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b --split test
python -m extraction extract-graph-scenes --run-id $RUN_ID --schema $PROMPT --model gemma4_e2b_lora --split test
python -m scoring score --run-id $RUN_ID --model gemma4_e2b --split test $D3
python -m scoring score --run-id $RUN_ID --model gemma4_e2b_lora --split test $D3
python -m evaluation compare --run-id $RUN_ID --students gemma4_e2b gemma4_e2b_lora --split test
