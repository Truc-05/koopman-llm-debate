"""Củng cố C1 từ data đã có (analysis-only, free). 3 phân tích:
  #2 POOLED effect-size : Δacc(exhaustive − no-def) mỗi dataset + bootstrap CI, và POOLED qua MAIN (Kans≤5).
  #3 ASR label-free     : Attack Success Rate = P(argmax belief cuối == x_adv), no-def vs defense; robustness=1−ASR.
  #4 label-free objective: P(honest_lead c_hon == a_star) & P(exhaustive-pick == a_star) — honest-margin có bám chân lý?
Defense = exhaustive honest-margin (c tối đa b[c_hon]−b[x_adv], c_hon=honest_lead(z_kdef)). Khớp analyze_twindef.
Dùng: python experiments/analyze_c1_strength.py
"""
import os, sys, json, glob
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.state import split_state

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))
cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
N = cfg["n_agents"]
rng = np.random.default_rng(0)


def honest_lead(z, x, Kans):
    _, p = split_state(np.asarray(z), N, Kans)
    h = p[1:].mean(axis=0).astype(float); h[x] = -np.inf
    return int(np.argmax(h))


def boot_ci(diffs, B=5000):
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return (np.nan, np.nan)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load(tag):
    ev = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_twindef_eval_{tag}.jsonl")) if l.strip()]
    Kans = len(next(iter(ev[0]["outs"].values()))["bel"]); cands = [-1] + list(range(Kans))
    rows = []
    for r in ev:
        x = r["x_adv"]; a = r["a_star"]; outs = r["outs"]; z = r["z_kdef"]
        ch = honest_lead(z, x, Kans)
        exc = max(cands, key=lambda c: np.asarray(outs[str(c)]["bel"])[ch] - np.asarray(outs[str(c)]["bel"])[x])
        nd, ex = outs["-1"], outs[str(exc)]
        rows.append({
            "nd_correct": bool(nd["correct"]), "ex_correct": bool(ex["correct"]),
            "nd_asr": int(np.argmax(nd["bel"]) == x), "ex_asr": int(np.argmax(ex["bel"]) == x),
            "c_hon_is_true": int(ch == a), "exc_is_true": int(exc == a)})
    return Kans, rows


def main():
    tags = sorted(os.path.basename(f)[len("ckpt_twindef_eval_"):-len(".jsonl")]
                  for f in glob.glob(os.path.join(OUT, "ckpt_twindef_eval_*.jsonl")))
    D = {}
    for t in tags:
        Kans, rows = load(t); D[t] = {"Kans": Kans, "rows": rows, "n": len(rows), "main": Kans <= 5}

    # ---- #2 POOLED effect-size ----
    print("\n===== #2 EFFECT-SIZE Δacc = exhaustive − no-def (paired, bootstrap CI95) =====")
    print(f"{'dataset':14s} {'n':>3s} {'no-def':>7s} {'exhaust':>7s} {'Δacc':>7s} {'CI95':>16s} {'main':>5s}")
    pooled = []
    for t, d in D.items():
        diffs = [int(r["ex_correct"]) - int(r["nd_correct"]) for r in d["rows"]]
        nd = np.mean([r["nd_correct"] for r in d["rows"]]); ex = np.mean([r["ex_correct"] for r in d["rows"]])
        lo, hi = boot_ci(diffs)
        star = "*" if lo > 0 else " "
        print(f"{t:14s} {d['n']:>3d} {nd:>7.3f} {ex:>7.3f} {np.mean(diffs):>+7.3f} [{lo:>+.3f},{hi:>+.3f}]{star} {str(d['main']):>5s}")
        if d["main"] and d["n"] >= 90:
            pooled.append(np.asarray(diffs, float))
    if pooled:
        pt = float(np.mean([p.mean() for p in pooled]))
        bs = [np.mean([p[rng.integers(0, len(p), len(p))].mean() for p in pooled]) for _ in range(5000)]
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print(f"{'POOLED (MAIN)':14s} {sum(len(p) for p in pooled):>3d} {'':>7s} {'':>7s} "
              f"{pt:>+7.3f} [{lo:>+.3f},{hi:>+.3f}]{'*' if lo>0 else ' '}   ({len(pooled)} dataset, stratified boot)")

    # ---- #3 ASR label-free ----
    print("\n===== #3 ASR = P(argmax belief cuối == x_adv) — label-free manipulation =====")
    print(f"{'dataset':14s} {'ASR no-def':>11s} {'ASR defense':>12s} {'robust(1−ASR)':>14s} {'ΔASR':>7s}")
    for t, d in D.items():
        an = np.mean([r["nd_asr"] for r in d["rows"]]); ax = np.mean([r["ex_asr"] for r in d["rows"]])
        print(f"{t:14s} {an:>11.3f} {ax:>12.3f} {1-ax:>14.3f} {ax-an:>+7.3f}")

    # ---- #4 label-free objective validation ----
    print("\n===== #4 honest-margin có BÁM chân lý? (label-free) =====")
    print(f"{'dataset':14s} {'P(c_hon=true)':>14s} {'P(exh-pick=true)':>17s}   (cao = objective label-free trỏ đúng)")
    for t, d in D.items():
        ph = np.mean([r["c_hon_is_true"] for r in d["rows"]]); pe = np.mean([r["exc_is_true"] for r in d["rows"]])
        print(f"{t:14s} {ph:>14.3f} {pe:>17.3f}")
    json.dump({t: {"n": d["n"], "Kans": d["Kans"]} for t, d in D.items()},
              open(os.path.join(OUT, "c1_strength.json"), "w"), indent=1)
    print("\n→ Δacc CI>0 = C1 chắc; ASR giảm = defense chặn manipulation (label-free); P(pick=true) cao = objective hợp lệ.")


if __name__ == "__main__":
    main()
