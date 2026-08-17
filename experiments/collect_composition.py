"""#1+#2 COLLECT — chỉ thu TEST compositions (train = collect single-intervention CŨ). CÓ gọi LLM (user chạy).
Mỗi câu: chọn 2 kênh c1,c2 (≠x_adv, ≠nhau), chạy 5 điều kiện, đẩy defense theo LỊCH (schedule) từ K_DEF:
  nodef  : [-1..]                 (K_0)
  s1     : c1,c1,c1,c1            (ΔK_{u1} đơn)
  s2     : c2,c2,c2,c2            (ΔK_{u2} đơn)
  seq12  : c1,c1, c2,c2          (compose A→B)     ← #1 unseen composition
  seq21  : c2,c2, c1,c1          (compose B→A)     ← #2 order/commutator
Lưu full traj mỗi điều kiện. CHECKPOINT/RESUME. Topic index bắt đầu SAU collect+eval (khỏi trùng train).
Dùng: python experiments/collect_composition.py --topics experiments/topics/mmlu_clean6.json --n 24
      # qwen/GPU1: --model qwen2.5:7b --base-url http://127.0.0.1:11435/v1/chat/completions
"""
import os, sys, json, argparse, re, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.run_twin_mpc_defense import ADV, DEFP, make, empty_bad, K_DEF
from src.debate.state import build_state, mean_belief

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))

def run_sched(cfg, ak, akc, burls, q, ans, x_adv, sched, T, ri, W):
    """như run_debate nhưng defense đẩy theo sched[r] (—1 = không đẩy)."""
    Nn = cfg["n_agents"]; advp = ADV.format(x=x_adv, a=ans[x_adv])
    ts, cur = [], [np.zeros(len(ans)) for _ in range(Nn)]; traj = [build_state(cur)]
    for r in range(T):
        c = sched[r] if r < len(sched) else -1
        defp = DEFP.format(c=c, a=ans[c]) if c >= 0 else ""
        ag = make(cfg, ak, akc, burls, [advp] + [defp] * (Nn - 1), W)
        rl = list(cur)
        for idx in range(Nn):
            if ri > 0: time.sleep(ri)
            lo, tx = ag[idx].act(q, ans, ts); rl[idx] = lo
            ts.append({"agent_id": idx, "round": r, "text": tx})
        cur = rl; traj.append(build_state(cur))
    return np.stack(traj), ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--offset", type=int, default=150, help="bỏ qua index train (collect 60 + eval 90)")
    ap.add_argument("--seed", type=int, default=23); ap.add_argument("--model", default=None); ap.add_argument("--base-url", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
    if args.model: cfg["model"] = args.model
    if args.base_url: cfg["base_url"] = args.base_url
    Nn, T = cfg["n_agents"], cfg["n_rounds"]; ri = cfg.get("request_interval", 0.0); W = cfg.get("transcript_window", 4)
    ak = os.environ.get(cfg.get("api_key_env", ""), ""); akc = cfg.get("api_key_cmd"); burls = cfg.get("base_urls")
    topics = json.load(open(args.topics)); Kans = len(topics[0]["answers"])
    base_seed = cfg.get("seed", 0) + args.seed; shuffle = cfg.get("shuffle_answers", True)
    mslug = re.sub(r"[^0-9a-zA-Z]+", "_", cfg["model"]).strip("_")
    tag = os.path.splitext(os.path.basename(args.topics))[0] + "_" + mslug
    half = (T - K_DEF) // 2
    def prep(ti):
        topic = topics[ti]; trng = np.random.default_rng(base_seed + ti)
        perm = trng.permutation(Kans) if shuffle else np.arange(Kans)
        a_star = int(np.where(perm == topic["a_star"])[0][0])
        x_adv = int(trng.choice([a for a in range(Kans) if a != a_star]))
        c1, c2 = trng.choice([a for a in range(Kans) if a != x_adv], size=2, replace=False)
        return [topic["answers"][j] for j in perm], topic["question"], a_star, x_adv, int(c1), int(c2)
    def sched(kind, c1, c2):
        pre = [-1] * K_DEF
        return {"nodef": pre + [-1] * (T - K_DEF), "s1": pre + [c1] * (T - K_DEF), "s2": pre + [c2] * (T - K_DEF),
                "seq12": pre + [c1] * half + [c2] * (T - K_DEF - half),
                "seq21": pre + [c2] * half + [c1] * (T - K_DEF - half)}[kind]

    ck = os.path.join(OUT, f"ckpt_composition_{tag}.jsonl")
    done = [json.loads(l) for l in open(ck)] if os.path.exists(ck) else []
    print(f"composition collect: model={cfg['model']} T={T} half={half} | resume {len(done)}/{args.n} | ckpt {os.path.basename(ck)}")
    for j in range(len(done), args.n):
        ti = args.offset + j
        ans, q, a_star, x_adv, c1, c2 = prep(ti)
        print(f"[{j+1}/{args.n}] ti={ti} adv={x_adv} c1={c1} c2={c2} — 5 điều kiện")
        conds = {}
        for kind in ["nodef", "s1", "s2", "seq12", "seq21"]:
            traj, ts = run_sched(cfg, ak, akc, burls, q, ans, x_adv, sched(kind, c1, c2), T, ri, W)
            if empty_bad(ts): sys.exit("DỪNG: reply rỗng — sửa server, chạy lại (resume giữ nguyên).")
            conds[kind] = traj.tolist()
        open(ck, "a").write(json.dumps({"ti": ti, "x_adv": x_adv, "a_star": a_star, "c1": c1, "c2": c2, "conds": conds}) + "\n")
        done.append(1)
    print("=== XONG composition collect ===  giờ: python experiments/analyze_composition.py")

if __name__ == "__main__":
    main()
