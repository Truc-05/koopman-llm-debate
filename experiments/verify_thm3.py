"""VERIFY Theorem 3 (persistence-dominance) — bound thành VERIFIED inequality (KHÔNG gọi LLM).
Autonomous (u_t=0) transitions; fit operator NGAY trên observable g (g_t -> g_{t+1}) để decomposition tự nhất quán:
  Rp   = mean ‖g_{t+1}-g_t‖^2         (persistence residual = movement)
  Rin  = mean ‖g_{t+1}-Â g_t‖^2 (in-sample, fit trên g)     ⇒ S0 = Rp-Rin ≥0 (drift 'thấy được')
  Roos = LOO-by-question                                    ⇒ η  = Roos-Rin ≥0 (variance/overfit)
  SNR  = S0/η ;  skill_oos = 1-Roos/Rp = (S0-η)/Rp
Thm3 ⇔ S0 ≤ η (SNR ≤ 1) ⇒ skill_oos ≤ 0. Kiểm điều kiện VÀ kết luận. Cùng chiều 2 quan sát ⇒ theorem verified.
Autonomous = (traj c=-1) HOẶC (t<K_DEF). Tách init-transient t=0→1 (từ zero-init, không đại diện). Chỉ tag Kans=4.
Dùng: python experiments/verify_thm3.py
"""
import os, sys, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.state import mean_belief

N, K_DEF, RIDGE = 4, 2, 1e-3
MODELS = ["mistral_7b", "llama3_1_8b", "gemma2_9b", "qwen2_5_7b"]
DIRS = [os.path.join(ROOT, "experiments", d) for d in ["out", "out_qwen"]]

def auton(col, drop_init):
    """per-question list of (t, z_t, z_{t+1}) với u_t=0; drop_init bỏ transition t=0."""
    Q = []
    for r in col:
        Z = np.asarray(r["traj"], float); c = r["c"]; rows = []
        for t in range(len(Z) - 1):
            if (c == -1 or t < K_DEF) and not (drop_init and t == 0):
                rows.append((t, Z[t], Z[t + 1]))
        if rows: Q.append(rows)
    return Q

def fit(X, Y): return np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ Y)

def decompose(Q, gfun):
    G = lambda M: np.array([gfun(m) for m in M]) if gfun else M
    def gg(Q):
        X = np.array([G([x])[0] for q in Q for _, x, _ in q]); Y = np.array([G([y])[0] for q in Q for _, _, y in q]); return X, Y
    X, Y = gg([[r for r in q] for q in Q])
    Rp = float(np.mean(np.sum((Y - X) ** 2, 1)))
    A = fit(X, Y); Rin = float(np.mean(np.sum((Y - X @ A) ** 2, 1)))
    res, nt = 0.0, 0
    for i in range(len(Q)):
        Xtr = np.array([G([x])[0] for j, q in enumerate(Q) if j != i for _, x, _ in q])
        Ytr = np.array([G([y])[0] for j, q in enumerate(Q) if j != i for _, _, y in q])
        Ai = fit(Xtr, Ytr)
        Xt = np.array([G([x])[0] for _, x, _ in Q[i]]); Yt = np.array([G([y])[0] for _, _, y in Q[i]])
        res += float(np.sum(np.sum((Yt - Xt @ Ai) ** 2, 1))); nt += len(Q[i])
    Roos = res / nt; S0 = Rp - Rin; eta = Roos - Rin
    return dict(Rp=Rp, S0=S0, eta=eta, snr=(S0 / eta if eta > 1e-12 else float("inf")),
                skill=1 - Roos / (Rp + 1e-12), nq=len(Q), nt=nt)

def model_of(tag):
    for m in MODELS:
        if tag.endswith(m): return m
    return "?"

# gom collect (chỉ Kans=4 để đồng chiều), theo model
by_model = {}
for D in DIRS:
    for f in glob.glob(os.path.join(D, "ckpt_twindef_collect_*.jsonl")):
        col = [json.loads(l) for l in open(f)]
        if np.asarray(col[0]["traj"]).shape[1] != N * 4: continue         # chỉ Kans=4
        by_model.setdefault(model_of(os.path.basename(f)[len("ckpt_twindef_collect_"):-len(".jsonl")]), []).append(col)

gmb = lambda z: mean_belief(np.asarray(z), N, 4)
print("===== VERIFY Thm 3: autonomous predictable-drift S0 vs estimation-error η (Kans=4 tags) =====")
print("(điều kiện S0≤η ⇔ SNR≤1  ⇒  kết luận skill_oos≤0)\n")
for drop in [False, True]:
    tag = "GỒM init-transient" if not drop else "BỎ init-transient (t0→1)"
    print(f"--- {tag} ---")
    print(f"{'model':11s} {'obs':7s} {'nq/nt':9s} {'Rp':>9s} {'S0':>9s} {'η':>9s} {'SNR':>7s} {'skill_oos':>10s}  verdict")
    poolQ = []
    for m in MODELS:
        if m not in by_model: continue
        Q = [q for col in by_model[m] for q in auton(col, drop)]; poolQ += Q
        for lab, gf in [("belief", gmb), ("state", None)]:
            d = decompose(Q, gf)
            ok = "✅ Thm3" if (d["snr"] <= 1.0 and d["skill"] <= 1e-6) else ("SNR>1→skill>0 (cùng chiều)" if d["skill"] > 0 else "?")
            print(f"{m:11s} {lab:7s} {d['nq']:3d}/{d['nt']:<5d} {d['Rp']:9.4f} {d['S0']:9.4f} {d['eta']:9.4f} {d['snr']:7.2f} {d['skill']:+10.3f}  {ok}")
    for lab, gf in [("belief", gmb), ("state", None)]:
        d = decompose(poolQ, gf)
        ok = "✅ Thm3" if (d["snr"] <= 1.0 and d["skill"] <= 1e-6) else ("SNR>1→skill>0 (cùng chiều)" if d["skill"] > 0 else "?")
        print(f"{'POOLED':11s} {lab:7s} {d['nq']:3d}/{d['nt']:<5d} {d['Rp']:9.4f} {d['S0']:9.4f} {d['eta']:9.4f} {d['snr']:7.2f} {d['skill']:+10.3f}  {ok}")
    print()
print("[đọc] Trên observable QUYẾT ĐỊNH (belief/answer): SNR≤1 & skill≤0 ⇒ Thm3 verified — operator KHÔNG vượt")
print("      persistence. Nếu state có SNR>1/skill>0 do init-transient thì cột 'BỎ init' sẽ triệt tiêu ⇒ chứng tỏ")
print("      predictable-structure = burn-in nhân tạo, không phải debate dynamics. Cùng chiều 2 regime = định lý đúng.")
