"""Probe 1 call tới endpoint trong configs/debate.yaml rồi in NGUYÊN response
JSON (finish_reason, usage, content) — chạy cái này TRƯỚC khi đốt cả run
mmlu_40 với provider mới. Dùng đúng Agent.call_model nên payload y hệt pipeline.

    /home/alex/venvs/env/bin/python experiments/probe_api.py
"""
import json
import os
import sys
import time

import requests
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.debate.orchestrator import Agent  # noqa: E402

with open(os.path.join(ROOT, "configs", "debate.yaml")) as f:
    cfg = yaml.safe_load(f)

api_key = os.environ.get(cfg.get("api_key_env", ""), "")
if not api_key and "localhost" not in cfg["base_url"]:
    sys.exit(f"{cfg['api_key_env']} chưa được set trong môi trường.")

# Server cloud (vd Modal) cold start mất vài phút — 503 trong lúc đó là bình
# thường. Poll /health (vLLM không bắt auth ở endpoint này) tới khi sẵn sàng.
root_url = cfg["base_url"].split("/v1/")[0]
if "localhost" not in root_url:
    print(f"== đợi server sẵn sàng: GET {root_url}/health "
          f"(cold start lần đầu tải weights, có thể 5-10 phút)")
    t0 = time.time()
    while True:
        try:
            status = requests.get(root_url + "/health", timeout=10).status_code
        except Exception as e:
            status = f"lỗi mạng ({e})"
        if status == 200:
            print(f"   server OK sau {time.time() - t0:.0f}s")
            break
        if time.time() - t0 > 15 * 60:
            sys.exit("   15 phút vẫn chưa sẵn sàng — xem log server: "
                     "modal app logs koopman-debate-vllm")
        print(f"   {status} — thử lại sau 15s (đã đợi {time.time() - t0:.0f}s)")
        time.sleep(15)

agent = Agent(0, model=cfg["model"], temperature=cfg["temperature"],
              max_tokens=cfg["max_tokens"], base_url=cfg["base_url"],
              api_key=api_key or "none", retries=0,
              timeout=cfg.get("timeout", 120))

prompt = ('Question: 2+2=? Options: 0) 3  1) 4  2) 5. '
          'Give your reasoning in at most 2 sentences, then on the LAST line '
          'output ONLY a JSON object, format: {"probs": [p0, p1, p2]}')

# In cả response thô (monkeypatch mỏng quanh requests.post) để thấy usage.
import src.debate.orchestrator as orch  # noqa: E402
_post = orch.requests.post


def _post_and_dump(*a, **kw):
    r = _post(*a, **kw)
    print(f"== HTTP {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False)[:3000])
    except Exception:
        print(r.text[:3000])
    return r


orch.requests.post = _post_and_dump
text = agent.call_model(prompt)
print("\n== content mà pipeline nhìn thấy:")
print(repr(text))
print("\n== parse_probs ->", agent.parse_probs(text, 3))
