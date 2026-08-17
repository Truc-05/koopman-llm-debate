"""Koopman-with-control (EDMDc): fit Ψ(z_{t+1}) = K·Ψ(z_t) + B·u_t từ debate có can thiệp
ngẫu nhiên, rồi KIỂM B có mã hóa đúng "đẩy đáp án X → nâng niềm tin X" không.

= trái tim "Koopman control theory of debate". Nếu B đúng => có mô hình điều khiển phổ
(không chỉ "đẩy thấy dịch") => nền cho MPC. u_t = one-hot đáp án moderator đẩy tại vòng t
(0 nếu chưa đẩy); đẩy từ vòng k_int ngẫu nhiên.

Dùng (qwen local): python experiments/run_edmdc_control.py --topics experiments/topics/mmlu_clean6.json --n 80
Thu xong tự fit + phân tích. Checkpoint/resume + fail-fast.
"""
import os, sys, json, argparse, time
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.orchestrator import Agent
from src.debate.state import build_state, mean_belief, split_state
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc

OUT = os.path.join(ROOT, "experiments", "out")


def make_agents(cfg, api_key, api_key_cmd, base_urls, sysp=""):
    return [Agent(i, model=cfg["model"], temperature=cfg["temperature"],
                  max_tokens=cfg["max_tokens"], base_url=cfg.get("base_url", "none"),
                  base_urls=base_urls, api_key=api_key or "none", retries=cfg.get("retries", 5),
                  transcript_window=cfg.get("transcript_window", 4), timeout=cfg.get("timeout", 120),
                  api_key_cmd=api_key_cmd, system_prompt=sysp) for i in range(cfg["n_agents"])]


def play(agents, q, ans, ts, cur, rounds, ri):
    states = []
    for r in rounds:
        rl = list(cur)
        for idx in range(len(agents)):
            if ri > 0:
                time.sleep(ri)
            lo, tx = agents[idx].act(q, ans, ts)
            rl[idx] = lo; ts.append({"agent_id": idx, "round": r, "text": tx})
        cur = rl; states.append(build_state(cur))
    return cur, states


def mod_prompt(x, ans):
    return ("You are the debate moderator. Present the STRONGEST possible case FOR "
            f"option {x} (\"{ans[x]}\"), then ask each agent to seriously reconsider it.")


def collect(args, cfg):
    N, T = cfg["n_agents"], cfg["n_rounds"]
    ri = cfg.get("request_interval", 0.0)
    api_key = os.environ.get(cfg.get("api_key_env", ""), "")
    api_key_cmd = cfg.get("api_key_cmd"); base_urls = cfg.get("base_urls")
    topics = json.load(open(args.topics))[:args.n]
    Kans = len(topics[0]["answers"])
    rng = np.random.default_rng(cfg.get("seed", 0) + args.seed)
    shuffle = cfg.get("shuffle_answers", True)
    tag = os.path.splitext(os.path.basename(args.topics))[0]
    ckpt = os.path.join(OUT, f"ckpt_edmdc_{tag}.jsonl")
    recs = [json.loads(l) for l in open(ckpt)] if os.path.exists(ckpt) else []
    if recs:
        print(f"RESUME: {len(recs)} debate.")
    for ti, topic in enumerate(topics):
        perm = rng.permutation(Kans) if shuffle else np.arange(Kans)
        k_int = int(rng.integers(1, max(2, T - 2)))   # đẩy sớm để có nhiều bước có-điều-khiển
        x_push = int(rng.integers(0, Kans))
        if ti < len(recs):
            continue
        ans = [topic["answers"][j] for j in perm]
        a_star = int(np.where(perm == topic["a_star"])[0][0])
        print(f"--- [{ti+1}/{len(topics)}] đẩy {x_push}@vòng{k_int}")
        ag = make_agents(cfg, api_key, api_key_cmd, base_urls)
        ts, cur = [], [np.zeros(Kans) for _ in range(N)]; traj = [build_state(cur)]
        cur, st = play(ag, topic["question"], ans, ts, cur, range(0, k_int), ri); traj += st
        if sum(1 for t in ts if not t["text"].strip()) * 4 > len(ts):
            sys.exit(f"DỪNG: reply rỗng nhiều (topic {ti+1}) — sửa server, chạy lại (resume).")
        ag2 = make_agents(cfg, api_key, api_key_cmd, base_urls, sysp=mod_prompt(x_push, ans))
        cur, st2 = play(ag2, topic["question"], ans, ts, cur, range(k_int, T), ri); traj += st2
        rec = {"traj": np.stack(traj).tolist(), "k_int": k_int, "x_push": x_push,
               "a_star": a_star, "correct": bool(np.argmax(mean_belief(np.stack(traj)[-1], N, Kans)) == a_star)}
        open(ckpt, "a").write(json.dumps(rec) + "\n"); recs.append(rec)
    return recs, N, T, Kans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True); ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--reg", type=float, default=1e-6)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
    recs, N, T, K = collect(args, cfg)

    # ---- build snapshots có điều khiển + fit EDMDc ----
    d = PolynomialDictionary(degree=1)
    Xc, Yc, Uc = [], [], []
    for r in recs:
        Z = np.asarray(r["traj"]); k_int, x = r["k_int"], r["x_push"]
        for t in range(len(Z) - 1):
            Xc.append(Z[t]); Yc.append(Z[t + 1])
            u = np.zeros(K); u[x] = 1.0 if t >= k_int else 0.0; Uc.append(u)
    Psi_X = d.transform(np.array(Xc)).T; Psi_Y = d.transform(np.array(Yc)).T
    U = np.array(Uc).T
    Kop, B = fit_edmdc(Psi_X, Psi_Y, U, reg=args.reg)
    print(f"\nEDMDc: {Psi_X.shape[0]} features, {Psi_X.shape[1]} transitions, u_dim={K}")

    # ---- KIỂM B: đẩy đáp án X có nâng logit đáp án X không? ----
    ss = d.state_slice
    print("\n== validate B: đẩy đáp án X -> thay đổi logit trung bình mỗi đáp án ==")
    hit = 0
    for x in range(K):
        u = np.zeros(K); u[x] = 1.0
        dz = np.real((B @ u)[ss]).reshape(N, K).mean(0)   # logit-delta TB/agent trên mỗi đáp án
        top = int(np.argmax(dz))
        ok = (top == x)
        hit += ok
        print(f"  đẩy {x}: Δlogit = {np.round(dz,2)}  -> nâng mạnh nhất đáp án {top} {'✓' if ok else '✗'}")
    print(f"\nB đúng hướng {hit}/{K} đáp án  => {'MÔ HÌNH ĐIỀU KHIỂN HỢP LỆ' if hit >= K-1 else 'B YẾU/nhiễu'}")

    # ---- reachability trong-mô-hình: cuộn K,B với push X có kéo belief về X không ----
    print("\n== reachability (mô hình): từ trạng thái uniform, đẩy X liên tục -> belief cuối ==")
    z0 = np.zeros(N * K)
    for x in [0, 1]:
        psi = d.transform(z0.reshape(1, -1))[0]; u = np.zeros(K); u[x] = 1.0
        for _ in range(T):
            psi = Kop @ psi + B @ u
        bel = mean_belief(np.real(psi[ss]), N, K)
        print(f"  đẩy {x} {T} vòng -> belief mô hình = {np.round(bel,2)}  (P[{x}]={bel[x]:.2f})")

    # ---- steering ↔ accuracy (thực nghiệm, từ data) ----
    pt = np.array([r["x_push"] == r["a_star"] for r in recs]); co = np.array([r["correct"] for r in recs])
    print(f"\n== steering ↔ accuracy (thực) ==  đẩy TRÚNG: acc={co[pt].mean() if pt.any() else float('nan'):.2f} "
          f"(n={int(pt.sum())}) | đẩy SAI: acc={co[~pt].mean() if (~pt).any() else float('nan'):.2f} "
          f"(n={int((~pt).sum())})  | baseline~0.61")
    print("\nĐọc: B đúng hướng + reachability kéo được belief về X => Koopman-control model HỢP LỆ.")
    print("Bước sau: MPC chọn can thiệp tối ưu + bộ chọn đích (verifier) -> đo Δaccuracy thật.")


if __name__ == "__main__":
    main()
