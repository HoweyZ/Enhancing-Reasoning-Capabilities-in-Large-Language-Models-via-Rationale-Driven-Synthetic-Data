set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/selfplay_output"
DATA_FILE="${1:-$PROJECT_DIR/data/selfplay_training.jsonl}"

echo "===== MiniPromptCoT Self-Play 训练 ====="
echo "数据文件: $DATA_FILE"
echo "输出目录: $OUTPUT_DIR"

# 训练
python -m MiniPromptCoT.training.selfplay_trainer \
    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "$OUTPUT_DIR" \
    --train_data_path "$DATA_FILE" \
    --learning_rate 1e-6 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --num_train_epochs 3 \
    --beta 0.1 \
    --logging_steps 10 \
    --save_steps 500

echo ""
echo "===== Self-Play 训练完成 ====="
echo "模型保存到: $OUTPUT_DIR"
