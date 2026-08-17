"""FIDELITY — contrib (2) vế sau: (K,B) rollout H-step ĐÚNG hơn persistence & DeGroot,
và NGANG model phi tuyến (n.s.). Đọc thẳng collect jsonl (có cho cả llama & qwen).

4 model, cùng chuỗi control THẬT u_t = onehot(c) khi c>=0 & t>=K_DEF:
  persistence : ẑ_h = z_0                          (belief đóng băng — baseline tầm thường)
  degroot     : psi_{t+1}=A·psi_t                  (tuyến-tính TỰ TRỊ, không B = DeGroot/FJ affine)
  koopman     : psi_{t+1}=K·psi_t + B·u_t          (K,B — bậc-1 lifted, CÓ control)
  mlp         : z_{t+1}=f([z_t,u_t])               (phi tuyến; kỳ vọng ≈ koopman)

Cross-fit KFold theo DEBATE (out-of-sample). Metric = belief-MSE tại mỗi horizon h.
Headline: %gain koopman vs persistence trên control-horizon; koopman−mlp paired bootstrap CI (n.s.?).

Dùng:  python experiments/check_fidelity.py           # mọi dataset trong experiments/out
       python experiments/check_fidelity.py math_llama3_1_8b
       OUT=experiments/out_qwen python experiments/check_fidelity.py   # đổi model
"""
import os, sys, json, glob, warnings
import numpy as np, yaml
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmd, fit_edmdc
from src.debate.state import mean_belief
from experiments.run_twin_mpc_defense import K_DEF

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))
RNG = np.random.default_rng(0)


def ctrl_seq(c, T, Kans):
    U = np.zeros((T - 1, Kans))
    if c >= 0:
        for t in range(K_DEF, T - 1):
            U[t, c] = 1.0
    return U


def snapshots(debates, d, Kans):
    X, Y, Uc = [], [], []
    for Z, c in debates:
        U = ctrl_seq(c, len(Z), Kans)
        for t in range(len(Z) - 1):
            X.append(Z[t]); Y.append(Z[t + 1]); Uc.append(U[t])
    Px = d.transform(np.array(X)).T; Py = d.transform(np.array(Y)).T
    return Px, Py, np.array(Uc).T, np.array(X), np.array(Y), np.array(Uc)


def rollout_belief(pred_fn, z0, U, T, N, Kans):
    """pred_fn(z_t,u_t)->z_{t+1}; trả belief (Kans) tại mỗi step 1..T-1."""
    z = z0.copy(); bel = []
    for t in range(T - 1):
        z = pred_fn(z, U[t]); bel.append(mean_belief(z, N, Kans))
    return np.array(bel)


def analyze(tag, N):
    col = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_twindef_collect_{tag}.jsonl")) if l.strip()]
    debates = [(np.asarray(r["traj"]), int(r["c"])) for r in col]
    T = len(debates[0][0]); Kans = len(debates[0][0][0]) // N
    d0 = PolynomialDictionary(degree=1); d0.transform(debates[0][0][:1]); ss = d0.state_slice
    idx = np.arange(len(debates)); RNG.shuffle(idx)
    folds = np.array_split(idx, 5)
    # per-debate, per-horizon belief-MSE cho mỗi model (mỗi debate test đúng 1 lần)
    err = {m: np.full((len(debates), T - 1), np.nan) for m in ["persist", "degroot", "koopman", "mlp"]}

    for fold in folds:
        te = set(fold.tolist()); tr = [debates[i] for i in idx if i not in te]
        d = PolynomialDictionary(degree=1); d.transform(tr[0][0][:1])
        Px, Py, Ucol, Xs, Ys, Us = snapshots(tr, d, Kans)
        Kdeg, _, _ = fit_edmd(Px, Py, reg=1e-6)                 # DeGroot: tự trị (fit_edmd -> K,G,A)
        K, B = fit_edmdc(Px, Py, Ucol, reg=1e-6)               # (K,B)
        mlp = MLPRegressor(hidden_layer_sizes=(64,), solver="lbfgs", max_iter=500, random_state=0)
        mlp.fit(np.hstack([Xs, Us]), Ys)                       # phi tuyến state-space

        def f_deg(z, u): return (Kdeg @ d.transform(z.reshape(1, -1))[0])[ss]
        def f_kop(z, u): return (K @ d.transform(z.reshape(1, -1))[0] + B @ u)[ss]
        def f_mlp(z, u): return mlp.predict(np.hstack([z, u]).reshape(1, -1))[0]
        def f_per(z, u): return z

        for i in fold:
            Z, c = debates[i]; U = ctrl_seq(c, T, Kans)
            true = np.array([mean_belief(Z[t + 1], N, Kans) for t in range(T - 1)])
            for m, fn in [("persist", f_per), ("degroot", f_deg), ("koopman", f_kop), ("mlp", f_mlp)]:
                pred = rollout_belief(fn, Z[0], U, T, N, Kans)
                err[m][i] = np.mean((pred - true) ** 2, axis=1)

    # control-horizon = các step t>=K_DEF (control đã bật). trung bình theo horizon rồi theo debate.
    hz = slice(K_DEF, T - 1)
    per_deb = {m: np.nanmean(err[m][:, hz], axis=1) for m in err}        # (n_debate,)
    mean = {m: float(np.mean(per_deb[m])) for m in err}
    gain = 100 * (mean["persist"] - mean["koopman"]) / mean["persist"]
    # koopman vs mlp: paired bootstrap CI trên chênh lệch per-debate (âm = koopman tốt hơn)
    dif = per_deb["koopman"] - per_deb["mlp"]
    bs = np.array([np.mean(dif[RNG.integers(0, len(dif), len(dif))]) for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ns = lo <= 0 <= hi
    return {"tag": tag, "T": T, "Kans": Kans, "n": len(debates),
            "per_horizon": {m: np.nanmean(err[m], axis=0).round(4).tolist() for m in err},
            "mean_horizon": mean, "gain_vs_persist_pct": gain,
            "koop_minus_mlp": float(np.mean(dif)), "ci": [float(lo), float(hi)], "koop_eq_mlp_ns": bool(ns)}


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml"))); N = cfg["n_agents"]
    tags = sorted(os.path.basename(f)[len("ckpt_twindef_collect_"):-len(".jsonl")]
                  for f in glob.glob(os.path.join(OUT, "ckpt_twindef_collect_*.jsonl")))
    if len(sys.argv) > 1:
        tags = [sys.argv[1]]
    res = {t: analyze(t, N) for t in tags}
    print(f"\n===== FIDELITY  (belief-MSE trên control-horizon t>={K_DEF}; OUT={OUT}) =====")
    print(f"{'dataset':26s} {'n':>3s} | {'persist':>7s} {'degroot':>7s} {'koopman':>7s} {'mlp':>7s} "
          f"| {'gain%':>6s} {'K−MLP':>8s} {'ns?':>4s}")
    for t, r in res.items():
        m = r["mean_horizon"]
        print(f"{t:26s} {r['n']:>3d} | {m['persist']:>7.4f} {m['degroot']:>7.4f} {m['koopman']:>7.4f} "
              f"{m['mlp']:>7.4f} | {r['gain_vs_persist_pct']:>+5.0f}% {r['koop_minus_mlp']:>+8.4f} "
              f"{'YES' if r['koop_eq_mlp_ns'] else 'no':>4s}")
    print("\nĐọc: koopman < persistence & degroot = (K,B) fidelity thật; gain% ~ paper '~40%'.")
    print("     ns=YES (CI koop−mlp chứa 0) = (K,B) ≈ phi tuyến, khớp claim 'n.s.'.")
    json.dump(res, open(os.path.join(OUT, "fidelity.json"), "w"), indent=1)
    print(f"→ {os.path.join(OUT, 'fidelity.json')}")


if __name__ == "__main__":
    main()
