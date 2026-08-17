"""Deploy vLLM server OpenAI-compatible trên Modal ($30 credit free/tháng).

Chuẩn bị (chỉ làm 1 lần):
    /home/alex/venvs/env/bin/pip install modal
    /home/alex/venvs/env/bin/modal setup                       # login browser
    modal secret create huggingface-secret HF_TOKEN=$HF_TOKEN  # llama là gated repo
    modal secret create vllm-api-key VLLM_API_KEY=<chuỗi tự đặt bất kỳ>

Deploy (chạy từ thư mục repo):
    modal deploy experiments/modal_vllm.py                # 8B trên L40S (~$2/giờ warm)
    MODEL_SIZE=70b modal deploy experiments/modal_vllm.py # 70B trên 2xH100 (~$8/giờ warm)

Deploy xong Modal in URL dạng https://<workspace>--koopman-debate-vllm-server.modal.run
-> dán vào khối Modal trong configs/debate.yaml, rồi:
    export VLLM_API_KEY=<chuỗi đã đặt ở trên>
    # chạy probe trước cho server ấm (lần đầu tải weights, có thể phải chạy 2-3 lần):
    /home/alex/venvs/env/bin/python experiments/probe_api.py
    # rồi mới chạy pipeline như thường lệ.

Tiền chỉ tính khi container warm; tự tắt sau SCALEDOWN_MIN phút không có request.
Tắt hẳn ngay: modal app stop koopman-debate-vllm
"""
import os

import modal

# Chọn size lúc deploy qua env var (đọc ở máy local, trước khi lên cloud).
MODEL_SIZE = os.environ.get("MODEL_SIZE", "8b").lower()
CONFIGS = {
    # 8B fp16 ~16GB weights: L40S 48GB dư dả (A10G 24GB rẻ hơn ~$1.1/giờ nếu muốn tiết kiệm)
    "8b": {"model": "meta-llama/Llama-3.1-8B-Instruct", "gpu": "L40S:1", "tp": 1},
    # 70B fp16 ~140GB: 2xH100 = 160GB, phải kèm --max-model-len thấp mới đủ chỗ KV cache
    "70b": {"model": "meta-llama/Llama-3.1-70B-Instruct", "gpu": "H100:2", "tp": 2},
}
CFG = CONFIGS[MODEL_SIZE]

MINUTES = 60
VLLM_PORT = 8000
SCALEDOWN_MIN = 5          # phút idle trước khi tự tắt — để thấp cho đỡ tốn credit

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04",
                              add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})   # tải weights từ HF nhanh hơn
)

# Volume cache: lần deploy sau không phải tải lại weights (~16GB/140GB).
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("koopman-debate-vllm")


@app.server(
    image=vllm_image,
    gpu=CFG["gpu"],
    scaledown_window=SCALEDOWN_MIN * MINUTES,
    startup_timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # HF_TOKEN cho gated llama
        modal.Secret.from_name("vllm-api-key"),        # VLLM_API_KEY chặn người lạ
    ],
    port=VLLM_PORT,
    target_concurrency=16,
    # URL public nhưng vLLM tự check Authorization: Bearer <VLLM_API_KEY> —
    # đúng header mà orchestrator.py đang gửi (proxy auth của Modal thì không khớp).
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm", "serve", CFG["model"],
            # alias "llm": configs/debate.yaml để model: llm là chạy được cả 8B lẫn 70B
            "--served-model-name", CFG["model"], "llm",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--api-key", os.environ["VLLM_API_KEY"],
            # prompt của debate ngắn (transcript_window=4) — 8192 là dư,
            # và là điều kiện để 70B nhét vừa 2xH100
            "--max-model-len", "8192",
            "--tensor-parallel-size", str(CFG["tp"]),
            # giữ CUDA graphs (boot chậm thêm ~1-2 phút, bù lại decode nhanh hơn
            # hẳn cho run dài; chỉ test nhanh thì đổi thành --enforce-eager)
            "--no-enforce-eager",
        ]
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
