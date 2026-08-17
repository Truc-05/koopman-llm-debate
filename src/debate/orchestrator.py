# src/debate/orchestrator.py
import os
import re
import json
import time
import subprocess
from fractions import Fraction
import numpy as np
import requests

from src.debate.state import build_state, split_state

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Bộ đếm dùng chung cho mọi agent: round-robin đa vùng (mỗi Vertex region có quota
# RPM riêng → xoay vòng để gộp throughput, né 429 mà không cần nâng quota).
_CALL_COUNTER = [0]


def _parse_wait_seconds(msg):
    """Lấy số giây từ message 429 của Groq: 'Please try again in 7.66s' /
    '1m30.5s' / '2h3m'. Trả None nếu không parse được."""
    m = re.search(r"try again in ([0-9hms\.]+)", msg)
    if not m:
        return None
    total = 0.0
    for val, unit in re.findall(r"([0-9.]+)(h|m|s)", m.group(1)):
        total += float(val) * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
    return total or None

# Model free sinh text trên Groq (đổi trong configs/debate.yaml nếu cần).
# KHÔNG dùng meta-llama/llama-prompt-guard-2-86m làm agent — nó là model phân
# loại an toàn, không sinh được lập luận + JSON probs.
DEFAULT_MODEL = "llama-3.1-8b-instant"


def _coerce_prob(tok):
    """1 phần tử prob -> float. Chấp nhận số, chuỗi thập phân/khoa học ('0.5',
    '1e-3') VÀ chuỗi phân số ('1/18', '-3/4') mà llama hay xuất. Ném ValueError/
    ZeroDivisionError nếu không parse được (caller bắt để RETRY, không crash)."""
    if isinstance(tok, bool):                       # True/False lọt vào np.asarray thành 1/0 -> chặn
        raise ValueError("bool không phải prob")
    if isinstance(tok, (int, float)):
        return float(tok)
    s = re.sub(r"\s+", "", str(tok)).strip('"').strip("'")
    if not s:
        raise ValueError("token rỗng")
    try:
        return float(s)                             # '0.5', '1e-3', '.25'
    except ValueError:
        return float(Fraction(s))                   # '1/18', '-3/4'


class Agent:
    """One debater backed by an OpenAI-compatible chat endpoint (Groq).

    API key: pass api_key=... or set env GROQ_API_KEY."""

    def __init__(self, agent_id, model=DEFAULT_MODEL, system_prompt=None,
                 temperature=0.7, max_tokens=300, api_key=None,
                 base_url=GROQ_URL, timeout=60, retries=5, transcript_window=8,
                 parse_retries=2, api_key_cmd=None, base_urls=None):
        self.agent_id = agent_id
        self.model = model
        self.system_prompt = system_prompt or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.base_url = base_url
        # base_urls: danh sách endpoint (nhiều region) để round-robin + fallback.
        self.base_urls = list(base_urls) if base_urls else None
        self._start = 0
        self.timeout = timeout
        self.retries = retries
        self.transcript_window = transcript_window  # giới hạn tokens/phút free tier
        self.parse_retries = parse_retries           # gọi lại khi probs sai độ dài
        # api_key_cmd: lệnh mint token (vd "gcloud auth print-access-token") cho
        # provider dùng OAuth token ngắn hạn (Vertex AI). Tự làm mới, khỏi export tay.
        self.api_key_cmd = api_key_cmd
        self._tok = None
        self._tok_ts = 0.0
        self.history = []

    def build_prompt(self, question, answers, transcript):
        opts = "\n".join(f"{i}: {a}" for i, a in enumerate(answers))
        parts = [f"Question: {question}", f"Options:\n{opts}"]
        if transcript:
            recent = transcript[-self.transcript_window:] \
                if self.transcript_window else transcript
            parts.append("Debate so far:" if len(recent) == len(transcript)
                         else f"Debate so far (last {len(recent)} turns):")
            for turn in recent:
                parts.append(f"Agent {turn['agent_id']}: {turn['text']}")
        K = len(answers)
        parts.append(
            "Give your reasoning in at most 2 sentences, then on the LAST line "
            f"output ONLY a JSON object with EXACTLY {K} probabilities (one per "
            f"option 0..{K - 1}, in order) summing to 1, format: "
            f'{{"probs": [{", ".join(f"p{i}" for i in range(K))}]}}'
        )
        return "\n\n".join(parts)

    def _bearer(self):
        """Token cho header Authorization. Có api_key_cmd (vd gcloud) thì tự mint
        + cache 50' (Vertex token sống ~60'); không thì dùng api_key tĩnh."""
        if not self.api_key_cmd:
            return self.api_key
        if self._tok is None or time.time() - self._tok_ts > 3000:
            try:
                self._tok = subprocess.check_output(
                    self.api_key_cmd, shell=True, text=True, timeout=30).strip()
                self._tok_ts = time.time()
            except Exception as e:
                print(f"[agent {self.agent_id}] mint token lỗi: {e}")
        return self._tok or ""

    def _pick_url(self, attempt):
        """URL cho lần thử: round-robin đa region (self._start) lệch theo attempt
        → 429/404 ở region này thì lần thử sau nhảy region kế (fallback)."""
        if not self.base_urls:
            return self.base_url
        return self.base_urls[(self._start + attempt) % len(self.base_urls)]

    def call_model(self, prompt):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Họ GPT-5 (reasoning model): API từ chối max_tokens (phải dùng
        # max_completion_tokens) và mọi temperature khác 1. reasoning_effort
        # minimal để token suy nghĩ không nuốt hết budget trước khi ra JSON.
        if self.model.split("/")[-1].startswith("gpt-5"):
            del payload["temperature"], payload["max_tokens"]
            payload["max_completion_tokens"] = self.max_tokens
            payload["reasoning_effort"] = "minimal"
        if self.base_urls:      # chọn region khởi đầu cho call này (xoay vòng chung)
            self._start = _CALL_COUNTER[0] % len(self.base_urls)
            _CALL_COUNTER[0] += 1
        for attempt in range(self.retries + 1):
            key = self._bearer()
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            url = self._pick_url(attempt)
            if key and not self.api_key_cmd and "googleapis.com" in url:
                headers["x-goog-api-key"] = key   # Google nhận API key qua header này (AI Studio/Vertex)
            try:
                r = requests.post(url, headers=headers, json=payload,
                                  timeout=self.timeout)
                if r.status_code == 401 and self.api_key_cmd:
                    self._tok = None   # token hết hạn -> mint lại, thử lại
                    print(f"[agent {self.agent_id}] 401 token hết hạn -> làm mới + thử lại")
                    continue
                if r.status_code == 404 and self.base_urls:
                    print(f"[agent {self.agent_id}] 404 (model không có ở region này) "
                          f"-> thử region khác")
                    continue
                if r.status_code == 429:
                    msg = ""
                    try:
                        msg = r.json().get("error", {}).get("message", "")
                    except Exception:
                        pass
                    # Quota NGÀY (TPD/RPD): retry trong vài phút là vô ích.
                    if "per day" in msg or "(TPD)" in msg or "(RPD)" in msg:
                        print(f"[agent {self.agent_id}] HẾT QUOTA NGÀY (RPD/TPD "
                              f"— giới hạn theo tier, không phải số dư), dừng "
                              f"retry: {msg[:200]}")
                        return ""
                    # Quota PHÚT (TPM/RPM): đợi đúng số giây Groq yêu cầu.
                    wait = (_parse_wait_seconds(msg)
                            or float(r.headers.get("retry-after", 0) or 0)
                            or 2 ** (attempt + 1))
                    wait = min(wait + 0.5, 120.0)
                    print(f"[agent {self.agent_id}] 429 rate limit "
                          f"(thử {attempt + 1}/{self.retries + 1}, đợi {wait:.0f}s) "
                          f"{msg[:160]}")
                    time.sleep(wait)
                    continue
                if r.status_code >= 400:
                    # In body để thấy lỗi thật của server (OOM, model not found...)
                    print(f"[agent {self.agent_id}] HTTP {r.status_code}: "
                          f"{r.text[:300]}")
                    if r.status_code < 500:
                        return ""      # lỗi config/request, retry vô ích
                    time.sleep(min(2 ** attempt, 30))
                    continue           # 5xx: thử lại
                data = r.json()
                choice = data["choices"][0]
                content = choice["message"]["content"] or ""
                if not content.strip():
                    # 200 nhưng rỗng (vd reasoning nuốt hết token budget) —
                    # nếu im lặng thì mọi agent rơi về uniform mà không ai biết.
                    print(f"[agent {self.agent_id}] CẢNH BÁO content rỗng, "
                          f"finish_reason={choice.get('finish_reason')}, "
                          f"usage={data.get('usage')}")
                return content
            except Exception as e:  # network failure -> retry rồi fallback
                print(f"[agent {self.agent_id}] API error: {e}")
                time.sleep(min(2 ** attempt, 30))
        print(f"[agent {self.agent_id}] hết {self.retries + 1} lần thử -> uniform fallback")
        return ""

    def parse_probs(self, text, n_answers):
        """Last valid {"probs": [...]} of EXACT length n_answers, else None.

        Trả None (không phải uniform) để act() biết mà RETRY — uniform im lặng
        chính là thứ đã làm hỏng ~19% lượt và tạo ổ τ=1 giả trong r5."""
        for m in reversed(re.findall(r"\{[^{}]*\}", text)):
            raw = None
            try:
                obj = json.loads(m)
                if isinstance(obj, dict) and "probs" in obj:
                    raw = obj["probs"]
            except json.JSONDecodeError:
                pass
            if raw is None:                         # JSON hỏng (vd phân số KHÔNG ngoặc kép: [1/18,...]) -> vớt mảng bằng regex
                mm = re.search(r'"probs"\s*:\s*\[([^\]]*)\]', m)
                if not mm:
                    continue
                raw = [t for t in mm.group(1).split(",") if t.strip()]
            if not isinstance(raw, (list, tuple)):
                continue
            try:
                probs = np.asarray([_coerce_prob(v) for v in raw], dtype=float).reshape(-1)
            except (ValueError, TypeError, ZeroDivisionError):
                continue                            # phần tử không parse được -> bỏ candidate, RETRY (không crash)
            if probs.shape[0] != n_answers or not np.all(np.isfinite(probs)) \
                    or probs.sum() <= 0 or np.any(probs < 0):
                continue
            probs = np.clip(probs, 1e-6, None)
            return probs / probs.sum()
        return None

    def probs_to_logits(self, probs):
        return np.log(probs)

    def act(self, question, answers, transcript):
        prompt = self.build_prompt(question, answers, transcript)
        K = len(answers)
        text, probs = "", None
        for attempt in range(self.parse_retries + 1):
            text = self.call_model(prompt)
            probs = self.parse_probs(text, K)
            if probs is not None:
                break
            if text.strip():   # có trả lời nhưng probs sai độ dài/thiếu JSON -> gọi lại
                print(f"[agent {self.agent_id}] probs không hợp lệ "
                      f"(thử {attempt + 1}/{self.parse_retries + 1}) -> gọi lại")
        parse_failed = probs is None
        if parse_failed:
            probs = np.ones(K) / K
            print(f"[agent {self.agent_id}] CẢNH BÁO uniform fallback sau "
                  f"{self.parse_retries + 1} lần thử (probs vẫn không hợp lệ)")
        logits = self.probs_to_logits(probs)
        self.history.append({"agent_id": self.agent_id, "text": text,
                             "probs": probs, "parse_failed": parse_failed})
        return logits, text


class DebateOrchestrator:
    """Sequential round-robin debate. Trajectory INCLUDES z_0 (uniform beliefs)
    so snapshots cover the first transition — early warning needs rounds 1-2.

    interventions: optional dict round_idx -> {"order": [...], "temperature": x}
    (the moderator actions Koopman-MPC acts through).
    request_interval: giây nghỉ giữa hai lần gọi API (free tier ~30 req/min)."""

    def __init__(self, agents, question, answers, n_rounds, speaking_order=None,
                 interventions=None, request_interval=0.0):
        self.agents = agents
        self.question = question
        self.answers = answers
        self.n_rounds = n_rounds
        self.speaking_order = speaking_order or list(range(len(agents)))
        self.interventions = interventions or {}
        self.request_interval = request_interval
        self.transcript = []
        self.trajectory = []

    def run(self):
        n_agents = len(self.agents)
        n_answers = len(self.answers)
        current_logits = [np.zeros(n_answers) for _ in range(n_agents)]
        self.trajectory = [build_state(current_logits)]  # z_0

        for round_idx in range(self.n_rounds):
            iv = self.interventions.get(round_idx, {})
            order = iv.get("order", self.speaking_order)
            if "temperature" in iv:
                for a in self.agents:
                    a.temperature = iv["temperature"]

            round_logits = list(current_logits)
            for k, idx in enumerate(order):
                if self.request_interval > 0 and (round_idx > 0 or k > 0):
                    time.sleep(self.request_interval)
                agent = self.agents[idx]
                logits, text = agent.act(self.question, self.answers, self.transcript)
                round_logits[idx] = logits
                self.transcript.append({"agent_id": idx, "round": round_idx, "text": text})
            current_logits = round_logits
            self.trajectory.append(build_state(current_logits))

        return np.stack(self.trajectory, axis=0)

    def get_snapshots(self):
        Z = np.stack(self.trajectory, axis=0)
        return Z[:-1], Z[1:]

    def final_belief(self):
        _, p = split_state(self.trajectory[-1], len(self.agents), len(self.answers))
        return p


def run_debate_batch(topics, agent_factory, n_agents, n_rounds, speaking_order=None):
    results = []
    for topic in topics:
        agents = [agent_factory(i) for i in range(n_agents)]
        orchestrator = DebateOrchestrator(
            agents, topic["question"], topic["answers"], n_rounds, speaking_order)
        traj = orchestrator.run()
        results.append({
            "topic": topic,
            "trajectory": traj,
            "transcript": orchestrator.transcript,
            "final_belief": orchestrator.final_belief(),
        })
    return results
