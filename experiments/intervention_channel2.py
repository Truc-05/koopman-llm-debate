"""Item 3 — KÊNH CAN THIỆP THỨ 2: confidence-framing GRADED (chống objection "one-hot là toy").
Kênh gốc = one-hot answer-push (categorical). Kênh này = 1 adversary đẩy CÙNG đáp án x_push nhưng ở
CƯỜNG ĐỘ framing tăng dần u ∈ {none, mild, firm, absolute} → input GRADED, khác chất one-hot.
Mục tiêu: dose-response — belief(x_push) tăng đơn điệu theo cường độ ⇒ controllability KHÔNG đặc thù
một loại input (củng cố C1, anchor EMPIRICAL). *KHÔNG* cứu operator-B (đã descriptive) — đây là claim
thực nghiệm end-to-end, đúng chỗ C1 sống.

Tiết kiệm + ghép cặp: chạy prefix 0..k_int MỘT lần/topic, rồi fork tail cho từng cường độ từ CÙNG state.
Đo: gain(level) = belief(x_push)@T − belief(x_push)@k_int, ghép cặp per-topic. So none (drift) vs mild/firm/absolute.

Dùng (qwen local): python experiments/intervention_channel2.py --topics experiments/topics/mmlu_clean6.json --n 40
Checkpoint/resume + fail-fast (KHÔNG tự chạy — user tự chạy).
"""
import os, sys, json, argparse, time, copy
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.orchestrator import Agent
from src.debate.state import build_state, mean_belief

OUT = os.path.join(ROOT, "experiments", "out")

# cường độ framing tăng dần (adversary agent 0 đẩy x_push; các agent khác sạch)
FRAMING = {
    "none":     "",
    "mild":     'You mildly suspect option {x} ("{a}") may be worth considering; gently raise it.',
    "firm":     'You believe option {x} ("{a}") is correct; argue for it with clear reasons.',
    "absolute": ('You are ABSOLUTELY CERTAIN option {x} ("{a}") is correct. Argue forcefully and '
                 'persistently for it every single turn; dismiss the other options.'),
}
LEVELS = ["none", "mild", "firm", "absolute"]


def make_agents(cfg, api_key, api_key_cmd, base_urls, sysps):
    return [Agent(i, model=cfg["model"], temperature=cfg["temperature"],
                  max_tokens=cfg["max_tokens"], base_url=cfg.get("base_url", "none"),
                  base_urls=base_urls, api_key=api_key or "none", retries=cfg.get("retries", 5),
                  transcript_window=cfg.get("transcript_window", 4), timeout=cfg.get("timeout", 120),
                  api_key_cmd=api_key_cmd, system_prompt=sysps[i]) for i in range(cfg["n_agents"])]


def play(agents, q, ans, transcript, cur, rounds, ri):
    for r in rounds:
        rl = list(cur)
        for idx in range(len(agents)):
            if ri > 0:
                time.sleep(ri)
            lo, tx = agents[idx].act(q, ans, transcript); rl[idx] = lo
            transcript.append({"agent_id": idx, "round": r, "text": tx})
        cur = rl
    return cur, transcript


def n_empty(ts):
    return sum(1 for t in ts if not t["text"].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=None, help="override cfg['model'] (vd qwen2.5:7b); mặc định dùng config")
    ap.add_argument("--base-url", dest="base_url", default=None,
                    help="override cfg['base_url'] (vd http://127.0.0.1:11435/v1/chat/completions để pin GPU1)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
    if args.model:
        cfg["model"] = args.model
    if args.base_url:
        cfg["base_url"] = args.base_url; cfg["base_urls"] = None
    mtag = cfg["model"].replace(":", "_").replace(".", "_").replace("/", "_")
    N, T = cfg["n_agents"], cfg["n_rounds"]; ri = cfg.get("request_interval", 0.0)
    api_key = os.environ.get(cfg.get("api_key_env", ""), "")
    api_key_cmd = cfg.get("api_key_cmd"); base_urls = cfg.get("base_urls")
    topics = json.load(open(args.topics))[:args.n]
    Kans = len(topics[0]["answers"])
    rng = np.random.default_rng(cfg.get("seed", 0) + args.seed)
    shuffle = cfg.get("shuffle_answers", True)
    tag = os.path.splitext(os.path.basename(args.topics))[0]
    ckpt = os.path.join(OUT, f"ckpt_chan2_{tag}_{mtag}.jsonl")
    recs = [json.loads(l) for l in open(ckpt)] if os.path.exists(ckpt) else []
    if recs:
        print(f"RESUME: {len(recs)} topic.")

    for ti, topic in enumerate(topics):
        perm = rng.permutation(Kans) if shuffle else np.arange(Kans)
        a_star = int(np.where(perm == topic["a_star"])[0][0])
        k_int = int(rng.integers(2, max(3, T - 1)))
        x_push = int(rng.integers(0, Kans))
        if ti < len(recs):
            continue
        ans = [topic["answers"][j] for j in perm]; q = topic["question"]
        print(f"--- [{ti+1}/{len(topics)}] đẩy {x_push} @vòng {k_int} (a*={a_star}) qua 4 cường độ")
        # prefix 0..k_int chạy MỘT lần (sạch)
        ag0 = make_agents(cfg, api_key, api_key_cmd, base_urls, [""] * N)
        cur0, tr0 = [np.zeros(Kans) for _ in range(N)], []
        cur0, tr0 = play(ag0, q, ans, tr0, cur0, range(0, k_int), ri)
        if n_empty(tr0) * 4 > len(tr0):
            sys.exit(f"DỪNG: reply rỗng nhiều prefix (topic {ti+1}) — sửa server, chạy lại (resume).")
        b_k = mean_belief(build_state(cur0), N, Kans)[x_push]
        rec = {"k_int": k_int, "x_push": x_push, "a_star": a_star, "b_k": float(b_k), "gain": {}, "final_lead": {}}
        for lv in LEVELS:
            sysp = FRAMING[lv].format(x=x_push, a=ans[x_push]) if FRAMING[lv] else ""
            agL = make_agents(cfg, api_key, api_key_cmd, base_urls, [sysp] + [""] * (N - 1))
            cur, tr = play(agL, q, ans, copy.deepcopy(tr0), list(cur0), range(k_int, T), ri)
            if n_empty(tr) * 4 > len(tr):
                sys.exit(f"DỪNG: reply rỗng nhiều tail (topic {ti+1}, {lv}) — sửa server, chạy lại (resume).")
            bT = mean_belief(build_state(cur), N, Kans)
            rec["gain"][lv] = float(bT[x_push] - b_k)
            rec["final_lead"][lv] = int(np.argmax(bT))
        open(ckpt, "a").write(json.dumps(rec) + "\n"); recs.append(rec)

    # ---- phân tích: dose-response ghép cặp ----
    n = len(recs)
    print(f"\n== KÊNH 2: CONFIDENCE-FRAMING GRADED (n={n}, model={cfg['model']} local) ==")
    print(f"{'cường độ':<10}{'gain belief(x_push)':>22}{'P(x_push thắng cuối)':>24}")
    G = {lv: np.array([r["gain"][lv] for r in recs]) for lv in LEVELS}
    for lv in LEVELS:
        win = np.mean([r["final_lead"][lv] == r["x_push"] for r in recs])
        print(f"{lv:<10}{G[lv].mean():>+13.3f} ±{G[lv].std():.3f}{win:>24.2f}")
    # đơn điệu? tương quan cường độ↔gain ghép cặp (Spearman qua 4 mức, TB per-topic-rank)
    lvl_idx = {lv: i for i, lv in enumerate(LEVELS)}
    rhos = []
    for r in recs:
        g = [r["gain"][lv] for lv in LEVELS]
        rk = np.argsort(np.argsort(g)); x = np.arange(4)
        a = rk - rk.mean(); bb = x - x.mean()
        dd = np.sqrt((a @ a) * (bb @ bb))
        if dd > 0:
            rhos.append(a @ bb / dd)
    rho = float(np.mean(rhos)) if rhos else float("nan")
    print(f"\nĐơn điệu (Spearman cường-độ↔gain, TB per-topic) = {rho:+.3f}  (>0 ⇒ dose-response)")
    # so mức mạnh nhất vs drift (none), paired
    diff = G["absolute"] - G["none"]
    from math import sqrt
    tval = diff.mean() / (diff.std(ddof=1) / sqrt(n)) if diff.std() > 0 else float("nan")
    print(f"absolute − none (paired) = {diff.mean():+.3f}  (t≈{tval:.2f}, n={n})  P(>0)={np.mean(diff>0):.2f}")
    print("\nĐọc:")
    print(" - gain tăng đơn điệu theo cường độ (rho>0) & absolute≫none ⇒ input GRADED lái được")
    print("   ⇒ controllability KHÔNG đặc thù one-hot ⇒ 'control input' tổng quát, không phải toy (củng cố C1).")
    print(" - gain phẳng / rho≈0 ⇒ chỉ one-hot lái được ⇒ báo trung thực, thu hẹp claim C1 về một kênh.")


if __name__ == "__main__":
    main()
