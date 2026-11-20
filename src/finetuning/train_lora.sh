#!/bin/bash
#
# LoRA fine-tuning of the student with ms-swift.
#
# Run inside the conda env that has ms-swift (requirements-finetune.txt):
#   MODEL_PATH=<model view> TRAIN_DATA=<run>/sft/train.jsonl VAL_DATA=<run>/sft/val.jsonl \
#       bash src/finetuning/train_lora.sh <output_dir> [epochs] [extra swift sft args...]
#
# MODEL_PATH must be the directory made by `python -m finetuning model-view`, so that the
# image token budget is the same as in extraction.
#
# Environment overrides: MODEL_TYPE (gemma4), TEMPLATE (gemma4_nothinking, the ms-swift
# default for E2B; a local directory name cannot select it), LORA_RANK (16), LORA_ALPHA (32),
# TARGET_MODULES (all-linear), LEARNING_RATE (1e-4).
# all-linear with the vision tower and aligner frozen adapts the language model's linear
# layers only. Plain names such as "q_proj v_proj" also match Gemma 4 vision/audio layers
# (Gemma4ClippableLinear), which PEFT cannot wrap.

set -e

OUTPUT_DIR="${1:?usage: train_lora.sh <output_dir> [epochs] [extra swift args]}"
NUM_EPOCHS="${2:-3}"
shift $(( $# < 2 ? $# : 2 ))
EXTRA_ARGS=("$@")
: "${MODEL_PATH:?set MODEL_PATH to the model view directory}"
: "${TRAIN_DATA:?set TRAIN_DATA}"
: "${VAL_DATA:?set VAL_DATA}"
MODEL_TYPE="${MODEL_TYPE:-gemma4}"
read -r -a TARGET_MODULES <<< "${TARGET_MODULES:-all-linear}"

for path in "$TRAIN_DATA" "$VAL_DATA"; do
    [ -s "$path" ] || { echo "[ERROR] dataset missing or empty: $path"; exit 1; }
done
# The pip package named "swift" is OpenStack Swift, not ms-swift: never auto-install it.
command -v swift >/dev/null 2>&1 || { echo "[ERROR] 'swift' (ms-swift) not found; activate its env"; exit 1; }

echo "model=$MODEL_PATH type=$MODEL_TYPE train=$TRAIN_DATA val=$VAL_DATA out=$OUTPUT_DIR epochs=$NUM_EPOCHS"

run_sft() {
    local resume_args=()
    [ -n "$1" ] && resume_args=(--resume_from_checkpoint "$1") && echo "[resume] $1"
    swift sft \
        --model "$MODEL_PATH" \
        --model_type "$MODEL_TYPE" \
        --template "${TEMPLATE:-gemma4_nothinking}" \
        --dataset "$TRAIN_DATA" \
        --val_dataset "$VAL_DATA" \
        --tuner_type lora \
        --lora_rank "${LORA_RANK:-16}" \
        --lora_alpha "${LORA_ALPHA:-32}" \
        --target_modules "${TARGET_MODULES[@]}" \
        --freeze_vit true \
        --freeze_aligner true \
        --learning_rate "${LEARNING_RATE:-1e-4}" \
        --num_train_epochs "$NUM_EPOCHS" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --save_strategy epoch \
        --eval_strategy epoch \
        --logging_steps 50 \
        --output_dir "$OUTPUT_DIR" \
        --torch_dtype bfloat16 \
        --seed 42 \
        "${resume_args[@]}" \
        "${EXTRA_ARGS[@]}"
}

# If the process crashes, resume from the newest checkpoint written by THIS invocation.
mkdir -p "$OUTPUT_DIR"
START_MARK="$OUTPUT_DIR/.run_started_$$"
touch "$START_MARK"
trap 'rm -f "$START_MARK"' EXIT

attempt=0
while true; do
    LATEST=""
    if [ "$attempt" -gt 0 ]; then
        LATEST=$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' \
            -newer "$START_MARK" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    fi
    if run_sft "$LATEST"; then break; fi
    attempt=$((attempt + 1))
    [ "$attempt" -ge "${MAX_RETRIES:-3}" ] && { echo "[ERROR] training failed $attempt times"; exit 1; }
    echo "[retry] swift sft failed (attempt $attempt); restarting in 10s"
    sleep 10
done

echo "Done. Merge the best checkpoint with:"
echo "  python -m finetuning merge --training-dir $OUTPUT_DIR --output <new model dir>"
