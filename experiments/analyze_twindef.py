"""Phân tích HỢP NHẤT twin-MPC-defense qua MỌI dataset — 1 file, tái lập từ jsonl đã commit.

Kết quả cần (map vào 3 claim):
  C1 headline  : exhaustive(honest-margin) vs no-def  -> Δacc, Wilson CI, McNemar ex>no-def (STAT TRỤ).
  Obj ablation : min-adv(CŨ) vs honest-margin(MỚI)    -> gain phụ thuộc objective (min-adv KHÔNG nâng).
  Twin limit   : twin vs no-def (yếu) & twin vs exhaustive (gap) + match + regret -> observability limit.
  Generalize   : cùng pattern qua các dataset; tiêu chí MAIN = Kans<=5 (loại bbh 7-choice -> appendix).

Twin refit (Kop,B) từ collect NGAY TRONG file -> cột twin reproduce được, không chỉ tin log.
Mirror logic của run_twin_mpc_defense.evaluate(); import helper thuần từ đó để khỏi lệch.

Dùng:  python experiments/analyze_twindef.py         # in bảng + ghi experiments/out/analysis_twindef.json
"""
import os, sys, json, glob
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief
from experiments.run_twin_mpc_defense import wilson, mcnemar_exact, honest_lead, K_DEF  # helper thuần, khỏi drift

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))
MAIN = lambda Kans: Kans <= 5                                    # tiêu chí bộ chính (nêu TRƯỚC trong Methods)
SCORES = {"min_adv": lambda b, x, ch: -b[x],                     # CŨ: chỉ dìm adversary (lệch accuracy)
          "honest_margin": lambda b, x, ch: b[ch] - b[x]}        # MỚI: nâng honest, dìm adversary (label-free)


def fit_twin(col, N, T, Kans):
    """Refit EDMDc (K autonomous gồm adversary, B=defense one-hot) -> closure dự đoán belief cuối."""
    d = PolynomialDictionary(degree=1)
    d.transform(np.asarray(col[0]["traj"])[:1]); ss = d.state_slice
    Xc, Yc, Uc = [], [], []
    for r in col:
        Z = np.asarray(r["traj"])
        for t in range(len(Z) - 1):
            u = np.zeros(Kans)
            if r["c"] >= 0 and t >= K_DEF:
                u[r["c"]] = 1.0
            Xc.append(Z[t]); Yc.append(Z[t + 1]); Uc.append(u)
    Kop, B = fit_edmdc(d.transform(np.array(Xc)).T, d.transform(np.array(Yc)).T, np.array(Uc).T, reg=1e-6)

    def twin_pred(z_kdef, c):
        psi = d.transform(z_kdef.reshape(1, -1))[0]
        for _ in range(K_DEF, T):
            u = np.zeros(Kans)
            if c >= 0:
                u[c] = 1.0
            psi = Kop @ psi + B @ u
        return mean_belief(np.real(psi[ss]), N, Kans)
    return twin_pred


def analyze(tag, N, T):
    ce = os.path.join(OUT, f"ckpt_twindef_eval_{tag}.jsonl")
    cc = os.path.join(OUT, f"ckpt_twindef_collect_{tag}.jsonl")
    ev = [json.loads(l) for l in open(ce) if l.strip()]
    col = [json.loads(l) for l in open(cc) if l.strip()]
    Kans = len(next(iter(ev[0]["outs"].values()))["bel"]); cands = [-1] + list(range(Kans))
    twin_pred = fit_twin(col, N, T, Kans)
    res = {"n": len(ev), "Kans": Kans, "main": bool(MAIN(Kans)), "obj": {}}
    for oname, sf in SCORES.items():
        nd, tw, ex, match, reg = [], [], [], [], []
        for r in ev:
            x = r["x_adv"]; outs = r["outs"]; z = np.asarray(r["z_kdef"]); ch = honest_lead(z, x, N, Kans)
            twc = max(cands, key=lambda c: sf(twin_pred(z, c), x, ch))           # twin: belief DỰ ĐOÁN
            exc = max(cands, key=lambda c: sf(np.asarray(outs[str(c)]["bel"]), x, ch))  # vét cạn: belief THẬT
            nd.append(outs["-1"]["correct"]); tw.append(outs[str(twc)]["correct"]); ex.append(outs[str(exc)]["correct"])
            match.append(twc == exc)
            reg.append(sf(np.asarray(outs[str(exc)]["bel"]), x, ch) - sf(np.asarray(outs[str(twc)]["bel"]), x, ch))
        n = len(ev)

        def acc(a):
            k = int(np.sum(a)); lo, hi = wilson(k, n); return {"acc": k / n, "ci": [lo, hi]}
        e_tn = mcnemar_exact(ex, nd); t_tn = mcnemar_exact(tw, nd); t_te = mcnemar_exact(tw, ex)
        res["obj"][oname] = {
            "no_def": acc(nd), "twin": acc(tw), "exhaustive": acc(ex),
            "delta_ex_nd": float(np.mean(ex) - np.mean(nd)), "delta_tw_nd": float(np.mean(tw) - np.mean(nd)),
            "mcnemar_ex_vs_nd": {"up": e_tn[0], "down": e_tn[1], "p": e_tn[2]},   # <- STAT TRỤ (log gốc thiếu)
            "mcnemar_tw_vs_nd": {"up": t_tn[0], "down": t_tn[1], "p": t_tn[2]},
            "mcnemar_tw_vs_ex": {"up": t_te[0], "down": t_te[1], "p": t_te[2]},
            "twin_match_ex": float(np.mean(match)), "regret": float(np.mean(reg))}
    return res


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
    N, T = cfg["n_agents"], cfg["n_rounds"]
    tags = sorted(os.path.basename(f)[len("ckpt_twindef_eval_"):-len(".jsonl")]
                  for f in glob.glob(os.path.join(OUT, "ckpt_twindef_eval_*.jsonl")))
    all_res = {t: analyze(t, N, T) for t in tags}
    json.dump(all_res, open(os.path.join(OUT, "analysis_twindef.json"), "w"), indent=1)

    def fp(p):
        return f"{p:.3f}{'*' if p < 0.05 else ' '}"

    for scope, keep in [("MAIN (Kans<=5)", True), ("APPENDIX (Kans>5)", False)]:
        rows = [(t, r) for t, r in all_res.items() if r["main"] == keep]
        if not rows:
            continue
        print(f"\n===== {scope} =====")
        print(f"{'dataset':14s} {'K':>2s} {'n':>3s} | {'no-def':>7s} {'twin':>6s} {'exhaust':>7s} "
              f"{'Δex':>6s} {'p(ex>nd)':>9s} | {'Δtw':>6s} {'p(tw>nd)':>9s} {'match':>6s} {'reg':>6s}")
        for t, r in rows:
            o = r["obj"]["honest_margin"]
            print(f"{t:14s} {r['Kans']:>2d} {r['n']:>3d} | "
                  f"{o['no_def']['acc']:>7.3f} {o['twin']['acc']:>6.3f} {o['exhaustive']['acc']:>7.3f} "
                  f"{o['delta_ex_nd']:>+6.3f} {fp(o['mcnemar_ex_vs_nd']['p']):>9s} | "
                  f"{o['delta_tw_nd']:>+6.3f} {fp(o['mcnemar_tw_vs_nd']['p']):>9s} "
                  f"{o['twin_match_ex']:>6.2f} {o['regret']:>+6.3f}")

    print("\n----- Objective ablation (Δ exhaustive − no-def; chỉ MAIN set) -----")
    print(f"{'dataset':14s} {'min-adv(CŨ)':>12s} {'honest-margin(MỚI)':>19s}")
    for t, r in all_res.items():
        if r["main"]:
            print(f"{t:14s} {r['obj']['min_adv']['delta_ex_nd']:>+12.3f} {r['obj']['honest_margin']['delta_ex_nd']:>+19.3f}")

    sig = [t for t, r in all_res.items() if r["main"] and r["obj"]["honest_margin"]["mcnemar_ex_vs_nd"]["p"] < 0.05]
    main_n = [t for t, r in all_res.items() if r["main"]]
    print(f"\nC1 generalize (honest-margin, ex>no-def p<0.05): {len(sig)}/{len(main_n)} MAIN dataset  -> {sig}")
    print(f"→ experiments/out/analysis_twindef.json")


if __name__ == "__main__":
    main()
