"""Chốt B — REPRESENTATION SUFFICIENCY STUDY (KHÔNG phải "twin capacity ablation").

Câu hỏi: observable HIỆN TẠI có ĐỦ để dự báo can-thiệp-nào-tốt không? Biến-thiên ĐỘ GIÀU
của OBSERVABLE (không biến-thiên estimator — tránh regress "thử neural Koopman"), dùng predictor
PHI-THAM-SỐ (k-NN numpy; +GradientBoosting nếu có sklearn) để cô lập "thông tin trong observable"
khỏi "sức mạnh model". Regress TRỰC TIẾP (bỏ giả định dynamics tuyến tính của Koopman):

    (observable @ k_def, onehot(defense c))  →  final mean_belief (4-dim)

Train trên COLLECT ckpt (defense-c ngẫu nhiên, có full traj), test trên EVAL ckpt (topic khác →
KHÔNG leak). Mỗi mức richness đo:
  - RECOVERED HEADROOM = (acc − acc_nodef)/(acc_oracle − acc_nodef)  [số Figure-1]
  - Spearman(pred honest-margin, realized) TB theo topic       [chất lượng XẾP HẠNG can thiệp — sạch hơn]
  - CV R² học map (learnability diagnostic trên collect)

Richness (đều tính từ trajectory ĐÃ LƯU, 0 debate mới):
  R0 mean-belief(4)              — observable nghèo cố ý (vứt cấu trúc per-agent)
  R1 per-agent beliefs(16)       — thông tin twin đang dùng
  R2 R1 + confidence + agreement — entropy per-agent(4) + std cross-agent(4) = 24-dim

PLATEAU R0→R2 phẳng & dưới oracle ⇒ Observation Gap do OBSERVABLE, không do estimator.
⚠️ CẢNH BÁO n_collect nhỏ (~60): richer obs có thể fail vì THIẾU DATA (curse of dim), KHÔNG phải
   thiếu tin. Plateau chỉ CHẮC khi (a) CV R² không sụp do chiều, (b) collect lớn hơn. Script in cả hai.
⚠️ R3 (history/velocity: z@k_def − z@k_def-1, phá Markov) cần eval lưu THÊM state trước k_def →
   1-dòng patch run_twin_mpc_defense (lưu traj[K_DEF-1]) + rerun eval. Để RIÊNG, không trong bản này.

Dùng: python experiments/representation_sufficiency.py [--k 7]
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.state import split_state, mean_belief

N, Kans, K_DEF = 4, 4, 2
CANDS = [-1, 0, 1, 2, 3]
OUT = os.path.join(ROOT, "experiments", "out")


def per_agent_belief(z):
    _, p = split_state(np.asarray(z, float), N, Kans)     # (N, Kans)
    return p


def observable(z, level):
    """feature vector tại 1 state z (16-dim per-agent logits) theo mức richness."""
    p = per_agent_belief(z)                               # (N, Kans)
    if level == "R0":
        return p.mean(axis=0)                             # (4,)
    if level == "R1":
        return p.reshape(-1)                              # (16,)
    if level == "R2":
        ent = -np.sum(p * np.log(p + 1e-12), axis=1)      # per-agent entropy (N,)
        agree = p.std(axis=0)                             # cross-agent std per answer (Kans,)
        return np.concatenate([p.reshape(-1), ent, agree])  # (24,)
    raise ValueError(level)


def action(c):
    u = np.zeros(Kans)
    if c >= 0:
        u[c] = 1.0
    return u


def honest_lead(z, x_adv):
    p = per_agent_belief(z)
    h = p[1:].mean(axis=0).astype(float)
    h[x_adv] = -np.inf
    return int(np.argmax(h))


def spearman(a, b):
    """rank-corr thủ công (không scipy). a,b cùng độ dài."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


class KNN:
    """k-NN regression đa-đầu-ra, chuẩn-hoá feature bằng train stats (numpy thuần)."""
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
        self.Xn = (X - self.mu) / self.sd; self.Y = Y
        return self

    def predict(self, X):
        Xn = (np.atleast_2d(X) - self.mu) / self.sd
        out = []
        for q in Xn:
            d = np.sqrt(((self.Xn - q) ** 2).sum(1))
            idx = np.argsort(d)[:self.k]
            out.append(self.Y[idx].mean(0))
        return np.array(out)


def cv_r2(Xf, Yf, mk, folds=5):
    """R² k-fold (learnability). Trả R² trung bình đa-đầu-ra."""
    n = len(Xf); order = np.arange(n)          # KHÔNG shuffle (Date/random cấm ở harness; deterministic)
    sse = sst = 0.0
    for f in range(folds):
        te = order[f::folds]; tr = np.setdiff1d(order, te)
        if len(tr) < mk().k or len(te) == 0:
            continue
        pred = mk().fit(Xf[tr], Yf[tr]).predict(Xf[te])
        sse += ((Yf[te] - pred) ** 2).sum()
        sst += ((Yf[te] - Yf[tr].mean(0)) ** 2).sum()
    return 1.0 - sse / sst if sst > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", default=os.path.join(OUT, "ckpt_twindef_collect_mmlu_clean6.jsonl"))
    ap.add_argument("--eval", default=os.path.join(OUT, "ckpt_twindef_eval_mmlu_clean6.jsonl"))
    ap.add_argument("--k", type=int, default=7)
    args = ap.parse_args()
    col = [json.loads(l) for l in open(args.collect)]
    ev = [json.loads(l) for l in open(args.eval)]
    if not all("bel" in next(iter(r["outs"].values())) for r in ev):
        sys.exit("Eval ckpt thiếu 'bel' — cần bản mới.")

    # tham chiếu no-def / oracle (realized) từ eval
    def hmargin(bel, x, ch):
        return bel[ch] - bel[x]
    nd_acc, orc_acc = [], []
    for r in ev:
        x = r["x_adv"]; outs = r["outs"]; ch = honest_lead(r["z_kdef"], x)
        nd_acc.append(bool(outs["-1"]["correct"]))
        oc = max(CANDS, key=lambda c: hmargin(np.asarray(outs[str(c)]["bel"]), x, ch))
        orc_acc.append(bool(outs[str(oc)]["correct"]))
    a_nd, a_or = np.mean(nd_acc), np.mean(orc_acc)
    n = len(ev)

    # predictor factories: k-NN luôn; GBoost nếu có sklearn
    factories = [("kNN", lambda: KNN(args.k))]
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor

        class GB:
            def fit(self, X, Y):
                self.m = MultiOutputRegressor(
                    GradientBoostingRegressor(n_estimators=150, max_depth=2, random_state=0)).fit(X, Y)
                return self
            def predict(self, X):
                return self.m.predict(np.atleast_2d(X))
        GB.k = 5  # cho cv_r2 guard
        factories.append(("GBoost", lambda: GB()))
    except ImportError:
        print("(sklearn không có → chỉ chạy k-NN; cài sklearn để thêm GBoost)")

    print(f"\n============ CHỐT B: REPRESENTATION SUFFICIENCY (train={len(col)} collect,"
          f" test={n} eval) ============")
    print(f"tham chiếu (realized):  no-def acc {a_nd:.3f} | oracle acc {a_or:.3f}"
          f" | headroom = {a_or - a_nd:+.3f}")
    print(f"tham chiếu twin tuyến-tính (run trước): acc ~0.43, recovered ~10% headroom\n")
    print(f"{'model':7} {'obs':4} {'eval-acc':9} {'recov.head':11} {'rank-corr':10} {'CV R²':7}")
    print("-" * 52)

    for mname, mk in factories:
        for lvl in ("R0", "R1", "R2"):
            Xtr = np.array([np.concatenate([observable(np.array(r["traj"])[K_DEF], lvl), action(r["c"])])
                            for r in col])
            Ytr = np.array([mean_belief(np.array(r["traj"])[-1], N, Kans) for r in col])
            model = mk().fit(Xtr, Ytr)
            acc, corrs = [], []
            for r in ev:
                x = r["x_adv"]; outs = r["outs"]; z = r["z_kdef"]; ch = honest_lead(z, x)
                feats = np.array([np.concatenate([observable(z, lvl), action(c)]) for c in CANDS])
                pred = model.predict(feats)                       # (5, 4) belief dự đoán
                pscore = [hmargin(pred[i], x, ch) for i in range(len(CANDS))]
                rscore = [hmargin(np.asarray(outs[str(c)]["bel"]), x, ch) for c in CANDS]
                cstar = CANDS[int(np.argmax(pscore))]
                acc.append(bool(outs[str(cstar)]["correct"]))
                corrs.append(spearman(pscore, rscore))
            a = np.mean(acc)
            recov = (a - a_nd) / (a_or - a_nd) if a_or > a_nd else np.nan
            rc = np.nanmean(corrs)
            r2 = cv_r2(Xtr, Ytr, mk)
            print(f"{mname:7} {lvl:4} {a:9.3f} {recov:+11.2f} {rc:+10.3f} {r2:+7.2f}")

    print("\n---- ĐỌC ----")
    print(" recov.head → 1.0 = bằng oracle; ~0 = như no-def (observable vô dụng cho control).")
    print(" rank-corr  → khả năng XẾP HẠNG can thiệp; ~0 ⇒ observable KHÔNG chứa tin dự báo tác động.")
    print(" PLATEAU R0≈R1≈R2 & thấp, CV R² KHÔNG sụp theo chiều ⇒ Observation Gap là do OBSERVABLE.")
    print(" Nếu recov.head TĂNG đều R0→R2 ⇒ gap ĐANG đóng → cần thêm observable/data, chưa kết luận.")
    print(" ⚠️ n_collect nhỏ: nếu CV R² sụp mạnh ở R2 ⇒ có thể do thiếu data, cần scale collect rồi mới chốt.")


if __name__ == "__main__":
    main()
