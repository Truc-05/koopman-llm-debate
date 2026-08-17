"""Chốt B v2 — sửa 2 confound của bản v1, để TÁCH "thiếu data" khỏi "thiếu tin".

v1 lộ: (i) observable giàu hơn → TỆ hơn + CV R² tụt = curse-of-dim tại n=60 (KHÔNG kết luận
được "richer vô dụng"); (ii) thang R1/R2 gộp cả belief adversary (agent 0) = 4 chiều nhiễu.

v2 sửa:
  - Observable **HONEST-ONLY** (loại agent 0): R0 mean(3 honest)=4 | R1 per-honest=12 |
    R2 +entropy(3)+std(4)=19.
  - **LEARNING CURVE**: với mỗi observable, train ở m ∈ {20,30,40,50,60} (subset deterministic,
    trung bình vài offset) → recov.head(m). Đọc DỐC:
       R1/R2 recov ĐANG LÊN theo m  ⇒ THIẾU DATA → scale collect (+140 debate) sẽ giúp → gap CÓ THỂ đóng.
       tất cả PHẲNG-thấp                ⇒ THIẾU TIN/noise → observation-gap/irreducible firmer.
       R0 phẳng sớm (m≈40)          ⇒ R0 đủ-lực → trần myopic của R0 đáng tin.

Tham chiếu in kèm: no-def / myopic(fixed honest-push) / oracle.
KHÔNG chạy debate. Dùng: python experiments/representation_sufficiency_v2.py [--k 7]
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.state import split_state, mean_belief

N, Kans, K_DEF = 4, 4, 2
CANDS = [-1, 0, 1, 2, 3]
OUT = os.path.join(ROOT, "experiments", "out")
SIZES = [20, 30, 40, 50, 60]
REPS = 3


def honest_p(z):
    _, p = split_state(np.asarray(z, float), N, Kans)
    return p[1:]                                   # (N-1, Kans) — LOẠI adversary agent 0


def observable(z, level):
    ph = honest_p(z)                               # (3, 4)
    if level == "R0":
        return ph.mean(axis=0)                     # (4,)
    if level == "R1":
        return ph.reshape(-1)                      # (12,)
    if level == "R2":
        ent = -np.sum(ph * np.log(ph + 1e-12), axis=1)   # (3,)
        agree = ph.std(axis=0)                            # (4,)
        return np.concatenate([ph.reshape(-1), ent, agree])  # (19,)
    raise ValueError(level)


def action(c):
    u = np.zeros(Kans)
    if c >= 0:
        u[c] = 1.0
    return u


def honest_lead(z, x_adv):
    h = honest_p(z).mean(axis=0).astype(float)
    h[x_adv] = -np.inf
    return int(np.argmax(h))


def hmargin(bel, x, ch):
    return bel[ch] - bel[x]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def mcnemar(a, b):                                 # paired H0: a≡b. return (b>a, a>b, p 2-phía exact)
    from math import comb
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum()); n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


class KNN:
    def __init__(self, k): self.k = k
    def fit(self, X, Y):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
        self.Xn = (X - self.mu) / self.sd; self.Y = Y; return self
    def predict(self, X):
        Xn = (np.atleast_2d(X) - self.mu) / self.sd
        return np.array([self.Y[np.argsort(np.sqrt(((self.Xn - q) ** 2).sum(1)))[:self.k]].mean(0)
                         for q in Xn])


class Ridge:                                       # tuyến-tính, numpy-only (không phụ thuộc sklearn)
    def __init__(self, lam=1.0): self.lam = lam
    def fit(self, X, Y):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
        Xb = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        A = Xb.T @ Xb + self.lam * np.eye(Xb.shape[1]); A[-1, -1] -= self.lam   # không phạt bias
        self.W = np.linalg.solve(A, Xb.T @ Y); return self
    def predict(self, X):
        Xn = (np.atleast_2d(X) - self.mu) / self.sd
        return np.hstack([Xn, np.ones((len(Xn), 1))]) @ self.W


def subset(n, m, rep):
    idx = (np.linspace(0, n - 1, m).astype(int) + rep * 3) % n
    return np.unique(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", default=os.path.join(OUT, "ckpt_twindef_collect_mmlu_clean6.jsonl"))
    ap.add_argument("--eval", default=os.path.join(OUT, "ckpt_twindef_eval_mmlu_clean6.jsonl"))
    ap.add_argument("--k", type=int, default=7)
    args = ap.parse_args()
    col = [json.loads(l) for l in open(args.collect)]
    ev = [json.loads(l) for l in open(args.eval)]

    # tham chiếu realized
    nd, myo, orc = [], [], []
    for r in ev:
        x = r["x_adv"]; o = r["outs"]; ch = honest_lead(r["z_kdef"], x)
        nd.append(bool(o["-1"]["correct"])); myo.append(bool(o[str(ch)]["correct"]))
        oc = max(CANDS, key=lambda c: hmargin(np.asarray(o[str(c)]["bel"]), x, ch))
        orc.append(bool(o[str(oc)]["correct"]))
    a_nd, a_myo, a_or = np.mean(nd), np.mean(myo), np.mean(orc)

    factories = [("kNN", lambda: KNN(args.k)), ("Ridge", lambda: Ridge(1.0))]
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neural_network import MLPRegressor
        class GB:
            def fit(self, X, Y):
                self.m = MultiOutputRegressor(GradientBoostingRegressor(
                    n_estimators=150, max_depth=2, random_state=0)).fit(X, Y); return self
            def predict(self, X): return self.m.predict(np.atleast_2d(X))
        class MLP:
            def fit(self, X, Y):
                self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
                self.m = MLPRegressor(hidden_layer_sizes=(32,), max_iter=800,
                                      random_state=0).fit((X - self.mu) / self.sd, Y); return self
            def predict(self, X): return self.m.predict((np.atleast_2d(X) - self.mu) / self.sd)
        factories += [("GBoost", lambda: GB()), ("MLP", lambda: MLP())]
    except ImportError:
        print("(sklearn không có → chỉ k-NN + Ridge)")

    def eval_model(model, lvl):
        acc, corrs = [], []
        for r in ev:
            x = r["x_adv"]; o = r["outs"]; z = r["z_kdef"]; ch = honest_lead(z, x)
            feats = np.array([np.concatenate([observable(z, lvl), action(c)]) for c in CANDS])
            pred = model.predict(feats)
            ps = [hmargin(pred[i], x, ch) for i in range(len(CANDS))]
            rs = [hmargin(np.asarray(o[str(c)]["bel"]), x, ch) for c in CANDS]
            cs = CANDS[int(np.argmax(ps))]
            acc.append(bool(o[str(cs)]["correct"])); corrs.append(spearman(ps, rs))
        acc = np.array(acc, float); a = acc.mean()
        recov = (a - a_nd) / (a_or - a_nd) if a_or > a_nd else np.nan
        return recov, np.nanmean(corrs), acc          # acc = per-example (dùng bootstrap)

    print(f"\n===== CHỐT B v2: HONEST-ONLY + LEARNING CURVE (train≤{len(col)}, test={len(ev)}) =====")
    print(f"tham chiếu:  no-def {a_nd:.3f} | MYOPIC(fixed honest-push) {a_myo:.3f}"
          f" (recov {(a_myo-a_nd)/(a_or-a_nd):+.2f}) | oracle {a_or:.3f}")
    print(f"twin tuyến-tính (run trước): acc ~0.43, recov ~0.10\n")

    nd_arr, orc_arr = np.array(nd, float), np.array(orc, float)
    Xful = {lvl: np.array([np.concatenate([observable(np.array(r["traj"])[K_DEF], lvl), action(r["c"])])
                           for r in col]) for lvl in ("R0", "R1", "R2")}
    Yful = np.array([mean_belief(np.array(r["traj"])[-1], N, Kans) for r in col])

    full = {}                                          # (mname,lvl) -> (recov60, acc_arr, rc60)
    for mname, mk in factories:
        print(f"[{mname}] recov.head theo train-size m (trung bình {REPS} subset):")
        print(f"    {'obs':4} " + " ".join(f"m={m:<5}" for m in SIZES) + "   rank-corr@60")
        for lvl in ("R0", "R1", "R2"):
            row = []
            for m in SIZES:
                rs = []
                for rep in range(REPS):
                    idx = subset(len(col), m, rep)
                    if len(idx) < args.k + 1:
                        continue
                    rec, _, _ = eval_model(mk().fit(Xful[lvl][idx], Yful[idx]), lvl)
                    rs.append(rec)
                row.append(np.nanmean(rs) if rs else np.nan)
            rec60, rc60, acc60 = eval_model(mk().fit(Xful[lvl], Yful), lvl)
            full[(mname, lvl)] = (rec60, acc60, rc60)
            print(f"    {lvl:4} " + " ".join(f"{v:+6.2f} " for v in row) + f"   {rc60:+.3f}")
        print()

    # ---- BOOTSTRAP CI + GAP(𝒪) : tách adaptive-premium khỏi noise ----
    B = 2000
    def boot(fn):                                      # resample test-set (paired: cùng seed mọi observer)
        rng = np.random.default_rng(0); n = len(nd_arr)
        return np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    r_myo = (a_myo - a_nd) / (a_or - a_nd)
    print(f"---- CI@m=60 (bootstrap test, B={B}) : recov.head [2.5%,97.5%] ----")
    print(f"    ref: MYOPIC recov {r_myo:+.2f} | ORACLE +1.00")
    for mname, _mk in factories:
        parts = []
        for lvl in ("R0", "R1", "R2"):
            rec60, acc, _ = full[(mname, lvl)]
            def f(ix, acc=acc):
                d, o = nd_arr[ix].mean(), orc_arr[ix].mean()
                return (acc[ix].mean() - d) / (o - d) if o > d else np.nan
            ci = np.nanpercentile(boot(f), [2.5, 97.5])
            parts.append(f"{lvl} {rec60:+.2f}[{ci[0]:+.2f},{ci[1]:+.2f}]")
        print(f"    {mname:7} " + " ".join(parts))

    accs = [v[1] for v in full.values()]               # 𝒪 = mọi observer×level, full-data
    best = max(a.mean() for a in accs); gap_pt = a_or - best
    gci = np.percentile(boot(lambda ix: orc_arr[ix].mean() - max(a[ix].mean() for a in accs)), [2.5, 97.5])
    print(f"\n---- GAP(𝒪) = oracle − best-observer  (|𝒪|={len(accs)}) ----")
    print(f"    gap = {gap_pt:+.3f} acc  CI[{gci[0]:+.3f},{gci[1]:+.3f}]   (best-observer acc {best:.3f} vs oracle {a_or:.3f})")

    # ---- SIGNIFICANCE (McNemar exact) : chốt 3-tier claim ----
    best_key = max(full, key=lambda k: full[k][1].mean())
    best_arr = full[best_key][1]
    myo_arr = np.array(myo, bool); nd_b = np.array(nd, bool); orc_b = np.array(orc, bool)
    print(f"\n---- McNemar (exact 2-phía) : trạng thái 3-tier claim  (best-obs = {best_key[0]}/{best_key[1]}) ----")
    for tag, a, b in [("C1  oracle   vs no-def ", nd_b, orc_b),
                      ("    myopic   vs no-def ", nd_b, myo_arr),
                      ("C2  best-obs vs no-def ", nd_b, best_arr),
                      ("C3  best-obs vs myopic ", myo_arr, best_arr),
                      ("    best-obs vs oracle ", best_arr, orc_b)]:
        up, dn, p = mcnemar(a, b)
        flag = "SIG" if p < 0.05 else "n.s."
        print(f"    {tag}: {up:2d}↑/{dn:2d}↓  p={p:.3f}  [{flag}]")

    print("\n---- ĐỌC ----")
    print(" CI recov phủ [myopic..oracle] rộng ⇒ underpowered: KHÔNG được nói 'saturate'.")
    print(" gap-CI cả hai đầu > 0  ⇒ adaptive-premium tách được khỏi noise → 'suggests a Control–Observation Gap' đứng.")
    print(" gap-CI chồng 0         ⇒ chưa tách được → chỉ nói 'predictive limitation', KHÔNG suy ra gap.")
    print(" recov ĐANG lên theo m (chưa phẳng) ⇒ nhánh (A) THIẾU-DATA: scale collect trước khi kết luận gap.")


if __name__ == "__main__":
    main()
