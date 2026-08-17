"""STRUCTURAL control-theoretic analysis (analysis-only, FREE — không debate mới).
Biến chuỗi kết-quả-âm (predict/twin/certificate chết) thành ĐỊNH LÝ dương bằng cách tính THẲNG
controllability & observability của debate-as-LTI từ operator EDMDc đã fit, mỗi dataset.

Hệ rút gọn trên state z = logit N×Kans (degree-1 dict ⇒ state-block khép kín):
  z_{t+1} = A z_t + Bz u_t   (A = state-block của K, Bz = state-block của B)
  y_t     = C z_t            (C = mean_belief tuyến-tính-hóa = TRUNG BÌNH logit qua N agent)
C có rank = Kans (gộp N agent → mất per-agent) ⇒ observability phụ thuộc A có "kéo" mode
bất-đối-xứng-agent vào trung bình theo thời gian không.

Đo:  controllability  (rank 𝒞, phổ+cond Gramian W_c)  — kỳ vọng ĐẦY/khỏe  (XANH, backing C1)
     observability    (rank 𝒪, cond Gramian W_o, unobs-dim) — kỳ vọng THIẾU/ill-cond (ĐỎ, backing predictive-limit)
     phổ K: ρ(A) + top eig (ổn định/consensus, nhất quán cross-dataset)

Dùng: python experiments/analyze_structure.py   → bảng + experiments/out/structure_analysis.json
"""
import os, sys, json, glob
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from experiments.run_twin_mpc_defense import K_DEF

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))


def fit_KB(col, Kans):
    d = PolynomialDictionary(degree=1); d.transform(np.asarray(col[0]["traj"])[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for r in col:
        Z = np.asarray(r["traj"])
        for t in range(len(Z) - 1):
            u = np.zeros(Kans)
            if r["c"] >= 0 and t >= K_DEF:
                u[r["c"]] = 1.0
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    K, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    L = K.shape[0]; ss_idx = np.arange(L)[ss]
    return np.real(K), np.real(B), ss_idx


def reduce_system(K, B, ss_idx, N, Kans):
    """Rút (A, Bz, C) trên state z n_x=N·Kans từ operator lift."""
    nx = N * Kans
    P = np.zeros((nx, K.shape[0]))
    for k in range(nx):
        P[k, ss_idx[k]] = 1.0
    A = P @ K @ P.T                      # state-transition n_x×n_x
    Bz = P @ B                           # n_x×Kans
    C = np.zeros((Kans, nx))             # y_j = mean_agent logit_j
    for j in range(Kans):
        for a in range(N):
            C[j, a * Kans + j] = 1.0 / N
    return A, Bz, C, nx


def numrank(M, rtol=1e-9):
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > rtol * s[0]).sum()), s


def gram_stats(W):
    ev = np.sort(np.real(np.linalg.eigvalsh((W + W.T) / 2)))[::-1]
    mn = float(max(ev[-1], 0.0)); mx = float(ev[0])
    cond = mx / mn if mn > 0 else np.inf
    return {"min_eig": mn, "max_eig": mx, "log10_cond": (float(np.log10(cond)) if np.isfinite(cond) else None)}


def analyze(tag, N):
    col = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_twindef_collect_{tag}.jsonl")) if l.strip()]
    Kans = np.asarray(col[0]["traj"]).shape[1] // N
    K, B, ss_idx = fit_KB(col, Kans)
    A, Bz, C, nx = reduce_system(K, B, ss_idx, N, Kans)
    H = A.shape[0]                                             # horizon = n_x (đủ cho rank/gramian)

    # controllability
    ctrb = np.hstack([np.linalg.matrix_power(A, t) @ Bz for t in range(nx)])
    rc, _ = numrank(ctrb)
    Wc = sum((np.linalg.matrix_power(A, t) @ Bz) @ (np.linalg.matrix_power(A, t) @ Bz).T for t in range(H))
    # observability
    obs = np.vstack([C @ np.linalg.matrix_power(A, t) for t in range(nx)])
    ro, _ = numrank(obs)
    Wo = sum(np.linalg.matrix_power(A, t).T @ C.T @ C @ np.linalg.matrix_power(A, t) for t in range(H))
    # spectrum
    eig = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    return {
        "n_x": nx, "Kans": Kans, "rank_C_readout": int(np.linalg.matrix_rank(C)),
        "ctrb": {"rank": rc, "full": rc == nx, "gram": gram_stats(Wc)},
        "obsv": {"rank": ro, "full": ro == nx, "unobs_dim": nx - ro,
                 "observable_via_dynamics": ro - int(np.linalg.matrix_rank(C)), "gram": gram_stats(Wo)},
        "spectral_radius": float(eig[0]), "top_eig_mag": [float(x) for x in eig[:3]]}


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
    N = cfg["n_agents"]
    tags = sorted(os.path.basename(f)[len("ckpt_twindef_collect_"):-len(".jsonl")]
                  for f in glob.glob(os.path.join(OUT, "ckpt_twindef_collect_*.jsonl")))
    res = {}
    for t in tags:
        try:
            res[t] = analyze(t, N)
        except Exception as e:
            res[t] = {"error": str(e)}
    json.dump(res, open(os.path.join(OUT, "structure_analysis.json"), "w"), indent=1)

    print(f"\n===== STRUCTURAL ANALYSIS (debate-as-LTI, N={N}) =====")
    print(f"{'dataset':14s} {'n_x':>4s} | {'CTRB rank':>9s} {'log10κ(Wc)':>10s} | "
          f"{'OBSV rank':>9s} {'unobs':>5s} {'via-dyn':>7s} {'log10κ(Wo)':>10s} | {'ρ(A)':>6s}")
    for t, r in res.items():
        if "error" in r:
            print(f"{t:14s}  ERROR: {r['error']}"); continue
        c, o = r["ctrb"], r["obsv"]
        kc = c["gram"]["log10_cond"]; ko = o["gram"]["log10_cond"]
        print(f"{t:14s} {r['n_x']:>4d} | "
              f"{c['rank']:>4d}/{r['n_x']:<4d} {(f'{kc:.1f}' if kc is not None else 'inf'):>10s} | "
              f"{o['rank']:>4d}/{r['n_x']:<4d} {o['unobs_dim']:>5d} {o['observable_via_dynamics']:>7d} "
              f"{(f'{ko:.1f}' if ko is not None else 'inf'):>10s} | {r['spectral_radius']:>6.3f}")

    print("\nĐọc:")
    print(" - CTRB rank đầy & κ(Wc) vừa  → debate CONTROLLABLE (backing C1, phe điều-khiển XANH).")
    print(" - OBSV rank<n_x hoặc κ(Wo) khổng-lồ → NOT observable từ mean-belief (readout gộp agent):")
    print("   giải thích structural vì sao predict/twin/certificate chết + predictive-limitation (myopic≈learned-obs).")
    print(" - 'via-dyn' = số chiều bất-đối-xứng-agent mà A kéo vào quan-sát-được; thấp = ẩn agent-disagreement.")
    print(" - ρ(A)≲1 nhất quán → dynamics ổn định/consensus. → experiments/out/structure_analysis.json")


if __name__ == "__main__":
    main()
