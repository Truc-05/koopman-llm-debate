"""ĐO CONTROLLABILITY: đẩy lập luận cho đáp án X có DỊCH được niềm tin về X không?

Điều kiện CẦN để Koopman-MPC thắng. Nếu bơm arg mà belief trơ / dịch loạn -> MPC vô ích,
biết sớm khỏi phí. Nếu đẩy X làm belief(X) tăng đáng kể (nhất là khi X đang THUA) -> có
controllability -> đáng dựng full EDMDc + MPC.

Thiết kế: mỗi debate, tại vòng k_int ngẫu nhiên, moderator đẩy đáp án x_push NGẪU NHIÊN.
Đo 1 bước:
  Δ_push    = belief(x_push)@(k+1) − @k       (sau khi đẩy)
  Δ_natural = belief(x_push)@k − @(k−1)        (drift tự nhiên, chưa đẩy) — control cùng đáp án
  Δ_others  = TB thay đổi các đáp án KHÔNG đẩy
+ P(x_push thành phe cuối) — đẩy có đổi KẾT CỤC không.

Dùng (qwen local): python experiments/check_controllability.py --topics experiments/topics/mmlu_clean6.json --n 60
Checkpoint/resume + fail-fast như run_debate_groq.
"""
import os, sys, json, argparse
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.orchestrator import Agent
from src.debate.state import build_state, mean_belief
import time

OUT = os.path.join(ROOT, "experiments", "out")


def make_agents(cfg, api_key, api_key_cmd, base_urls, system_prompt=""):
    return [Agent(i, model=cfg["model"], temperature=cfg["temperature"],
                  max_tokens=cfg["max_tokens"], base_url=cfg.get("base_url", "none"),
                  base_urls=base_urls, api_key=api_key or "none",
                  retries=cfg.get("retries", 5), transcript_window=cfg.get("transcript_window", 4),
                  timeout=cfg.get("timeout", 120), api_key_cmd=api_key_cmd, system_prompt=system_prompt)
            for i in range(cfg["n_agents"])]


def play(agents, question, answers, transcript, cur, rounds, ri):
    states = []
    for r in rounds:
        rl = list(cur)
        for idx in range(len(agents)):
            if ri > 0:
                time.sleep(ri)
            logits, text = agents[idx].act(question, answers, transcript)
            rl[idx] = logits
            transcript.append({"agent_id": idx, "round": r, "text": text})
        cur = rl
        states.append(build_state(cur))
    return cur, states


def mod_prompt(x, answers):
    return ("You are the debate moderator. Present the STRONGEST possible case FOR "
            f"option {x} (\"{answers[x]}\"), then ask each agent to seriously reconsider "
            "whether that option might be correct.")


def n_empty(ts):
    return sum(1 for t in ts if not t["text"].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
    N, T = cfg["n_agents"], cfg["n_rounds"]
    ri = cfg.get("request_interval", 0.0)
    api_key = os.environ.get(cfg.get("api_key_env", ""), "")
    api_key_cmd = cfg.get("api_key_cmd"); base_urls = cfg.get("base_urls")
    topics = json.load(open(args.topics))[:args.n]
    Kans = len(topics[0]["answers"])
    rng = np.random.default_rng(cfg.get("seed", 0) + args.seed)
    shuffle = cfg.get("shuffle_answers", True)

    tag = os.path.splitext(os.path.basename(args.topics))[0]
    ckpt = os.path.join(OUT, f"ckpt_ctrl_{tag}.jsonl")
    recs = [json.loads(l) for l in open(ckpt)] if os.path.exists(ckpt) else []
    if recs:
        print(f"RESUME: {len(recs)} debate đã có.")

    for ti, topic in enumerate(topics):
        perm = rng.permutation(Kans) if shuffle else np.arange(Kans)
        k_int = int(rng.integers(2, max(3, T - 1)))     # vòng can thiệp (>=2 để có drift tự nhiên)
        x_push = int(rng.integers(0, Kans))
        if ti < len(recs):
            continue
        answers = [topic["answers"][j] for j in perm]
        a_star = int(np.where(perm == topic["a_star"])[0][0])
        print(f"--- [{ti+1}/{len(topics)}] đẩy đáp án {x_push} tại vòng {k_int}: {topic['question'][:55]}")

        agents = make_agents(cfg, api_key, api_key_cmd, base_urls)
        transcript, cur = [], [np.zeros(Kans) for _ in range(N)]
        traj = [build_state(cur)]
        cur, st = play(agents, topic["question"], answers, transcript, cur, range(0, k_int), ri)
        traj += st                                      # z_0..z_{k_int}
        if n_empty(transcript) * 4 > len(transcript):
            sys.exit(f"DỪNG: nhiều reply rỗng vòng đầu (topic {ti+1}) — sửa server, chạy lại (resume).")
        # đẩy x_push từ vòng k_int
        agents2 = make_agents(cfg, api_key, api_key_cmd, base_urls, system_prompt=mod_prompt(x_push, answers))
        cur, st2 = play(agents2, topic["question"], answers, transcript, cur, range(k_int, T), ri)
        traj += st2
        Z = np.stack(traj)
        b = np.stack([mean_belief(Z[t], N, Kans) for t in range(T + 1)])   # (T+1, K)
        rec = {"k_int": k_int, "x_push": x_push, "a_star": a_star,
               "b_km1": b[k_int - 1].tolist(), "b_k": b[k_int].tolist(), "b_kp1": b[k_int + 1].tolist(),
               "final_leader": int(np.argmax(b[-1])), "correct": bool(np.argmax(b[-1]) == a_star)}
        with open(ckpt, "a") as f:
            f.write(json.dumps(rec) + "\n")
        recs.append(rec)

    # ---- phân tích ----
    d_push, d_nat, d_oth, push_is_leader, x_wins, x_was_leader = [], [], [], [], [], []
    for r in recs:
        x = r["x_push"]; bk1, bk, bk2 = np.array(r["b_km1"]), np.array(r["b_k"]), np.array(r["b_kp1"])
        d_push.append(bk2[x] - bk[x])
        d_nat.append(bk[x] - bk1[x])
        oth = [j for j in range(len(bk)) if j != x]
        d_oth.append(float(np.mean(bk2[oth] - bk[oth])))
        lead = int(np.argmax(bk))
        push_is_leader.append(x == lead)
        x_was_leader.append(x == lead)
        x_wins.append(r["final_leader"] == x)
    d_push, d_nat, d_oth = map(np.array, (d_push, d_nat, d_oth))
    pil = np.array(push_is_leader); xw = np.array(x_wins)

    print(f"\n== CONTROLLABILITY (n={len(recs)}) ==")
    print(f"Δ_push (đẩy X, 1 vòng)   = {d_push.mean():+.3f} ± {d_push.std():.3f}  (P(>0)={np.mean(d_push>0):.2f})")
    print(f"Δ_natural (X chưa đẩy)   = {d_nat.mean():+.3f} ± {d_nat.std():.3f}")
    print(f"Δ_others (đáp án khác)   = {d_oth.mean():+.3f}")
    print(f"Δ_push − Δ_natural       = {(d_push - d_nat).mean():+.3f}  (>0 => đẩy THÊM lực so với tự nhiên)")
    if (~pil).any():
        print(f"\nĐẩy đáp án ĐANG THUA (n={int((~pil).sum())}): Δ_push={d_push[~pil].mean():+.3f}, "
              f"P(nó thành phe cuối)={xw[~pil].mean():.2f}  <- promote được kẻ thua = controllability MẠNH")
    print(f"P(đáp án được đẩy thắng cuối) = {xw.mean():.2f}  (ngẫu nhiên ~{1/Kans:.2f})")
    # lát cắt: đẩy TRÚNG chân lý vs đẩy SAI -> hệ quả accuracy (ceiling & nguy cơ của steering)
    push_truth = np.array([r["x_push"] == r["a_star"] for r in recs])
    corr = np.array([r["correct"] for r in recs])
    print(f"\n== steering ↔ accuracy ==")
    if push_truth.any():
        print(f"đẩy TRÚNG đáp án đúng (n={int(push_truth.sum())}): acc = {corr[push_truth].mean():.2f}")
    if (~push_truth).any():
        print(f"đẩy SAI đáp án        (n={int((~push_truth).sum())}): acc = {corr[~push_truth].mean():.2f}")
    print(f"baseline không can thiệp (mmlu_clean6) ~ 0.61")
    print(" => đẩy trúng cao / đẩy sai thấp: lái tăng HAY giảm accuracy tùy CHỌN ĐÚNG đích")
    print("    (giá trị nằm ở bộ chọn đích, không phải lực lái). Đây là câu chuyện (B) cần verifier.")
    print("\nĐọc:")
    print(" - Δ_push>0 rõ & > Δ_natural, > Δ_others => ĐẨY DỊCH ĐƯỢC belief => có controllability => MPC có cửa.")
    print(" - đẩy kẻ thua mà nó lên/ thắng => điều khiển mạnh, MPC đáng dựng full.")
    print(" - Δ_push ≈ Δ_natural ≈ 0 hoặc loạn dấu => bơm arg belief TRƠ => MPC vô ích, dừng hướng điều khiển.")


if __name__ == "__main__":
    main()
