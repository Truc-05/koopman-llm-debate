#!/usr/bin/env bash
# Chạy chan2 + defbase trên qwen2.5:7b, PIN GPU1 (server ollama riêng port 11435).
# Dùng: nohup bash run_qwen_gpu1.sh > logs/qwen_gpu1_run.log 2>&1 &
set -euo pipefail
cd /home/alex/venvs/trucn/koopman-debate
PY=/home/alex/venvs/env/bin/python
PORT=11435
BASE="http://127.0.0.1:${PORT}/v1/chat/completions"

# 1) Ollama pin GPU1, port riêng, dùng CHUNG thư mục models (để thấy qwen2.5:7b)
export CUDA_VISIBLE_DEVICES=1
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
ollama serve > logs/ollama_gpu1.log 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT   # tắt server GPU1 khi xong/đứt (khỏi orphan)

# 2) đợi server lên (tối đa 60s)
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 || { echo "LỖI: ollama GPU1 không lên — xem logs/ollama_gpu1.log"; exit 1; }
echo "OK: ollama GPU1 up @:${PORT} (pid $OLLAMA_PID)"

# 3) chạy tuần tự cả 2 experiment (qwen), trỏ vào server GPU1
echo "=== [1/2] chan2 (qwen, GPU1) ==="
$PY experiments/intervention_channel2.py --topics experiments/topics/mmlu_clean6.json --n 40 \
    --model qwen2.5:7b --base-url "$BASE"
echo "=== [2/2] defbase (qwen, GPU1) ==="
$PY experiments/defense_baselines.py     --topics experiments/topics/mmlu_clean6.json --n 40 \
    --model qwen2.5:7b --base-url "$BASE"

echo "=== XONG cả 2 (qwen, GPU1) ==="
