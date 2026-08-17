"""DE-RISK chân B: Koopman RESPONSE operator ΔK_u có teeth không? (KHÔNG gọi LLM — reuse twindef ckpt paired.)
Mỗi câu eval có outs["-1"]=K_0 (no-def) và outs[c]=K_u (đẩy kênh c) ⇒ effect thật ΔK_u = bel_c - bel_{-1}, paired, n≈90/model.

Mọi 'method' = hàm f(z,c)->belief-vector dự đoán. Từ đó suy: (A) effect-prediction, (B) intervention-selection.
Methods:
  koopman   : rollout (K,B) fit trên COLLECT (twin_pred_belief) — đối tượng operator thật
  direct_lin: ridge [z16, onehot(c)] -> bel        (LOO-CV trên eval; non-Koopman causal baseline)
  direct_quad: ridge [z16, belief, belief^2, onehot(c)] -> bel   (non-Koopman phi tuyến)
  meaneff   : bel_c = climatology theo kênh c      (không dùng z)
  persist   : bel_c = mean_belief(z)               (bỏ qua c -> effect=0; NULL)
TEETH: koopman phải VƯỢT CẢ direct_lin & direct_quad (nếu direct ngang -> 'why Koopman?'). Thắng persistence/meaneff là KHÔNG đủ.

Dùng: python experiments/response_operator_test.py            (quét out/ + out_qwen/, pool theo model)
"""
import os, sys, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief, split_state

N, K_DEF = 4, 2
MODELS = ["mistral_7b", "llama3_1_8b", "gemma2_9b", "qwen2_5_7b"]
DIRS = [os.path.join(ROOT, "experiments", d) for d in ["out", "out_qwen"]]

# ---------- Koopman (K,B) fit trên collect + rollout ----------
def fit_koopman(col):
    D = np.asarray(col[0]["traj"]).shape[1]; Kans = D // N; T = np.asarray(col[0]["traj"]).shape[0] - 1
    d = PolynomialDictionary(degree=1); d.transform(np.asarray(col[0]["traj"])[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for r in col:
        Z = np.asarray(r["traj"], float)
        for t in range(len(Z) - 1):
            u = np.zeros(Kans)
            if r["c"] >= 0 and t >= K_DEF: u[r["c"]] = 1.0
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    Kop, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    def pred(z, c):
        psi = d.transform(np.asarray(z).reshape(1, -1))[0]
        for _ in range(K_DEF, T):
            u = np.zeros(Kans)
            if c >= 0: u[c] = 1.0
            psi = Kop @ psi + B @ u
        return mean_belief(np.real(psi[ss]), N, Kans)
    return pred, Kans

def honest_lead(z, x, Kans):
    _, p = split_state(np.asarray(z), N, Kans); h = p[1:].mean(0).astype(float); h[x] = -np.inf
    return int(np.argmax(h))

def feats(z, Kans, quad):
    z = np.asarray(z, float); b = mean_belief(z, N, Kans)
    f = np.concatenate([z, b])
    if quad: f = np.concatenate([f, b ** 2])
    return f

def ridge_fit(Xtr, Ytr, lam=1.0):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    return np.linalg.solve(A, Xtr.T @ Ytr)

# ---------- 1 model-dataset ----------
def run_tag(cc, ce):
    col = [json.loads(l) for l in open(cc)]; ev = [json.loads(l) for l in open(ce)]
    ev = [r for r in ev if r.get("z_kdef") is not None]
    kpred, Kans = fit_koopman(col)
    cands = list(range(Kans))                                    # kênh can thiệp thật (bỏ -1 khỏi 'chọn effect')
    Z = [np.asarray(r["z_kdef"], float) for r in ev]
    X = [r["x_adv"] for r in ev]; AST = [r["a_star"] for r in ev]
    OUT = [r["outs"] for r in ev]
    bel0 = [np.asarray(o["-1"]["bel"], float) for o in OUT]      # K_0
    belc = [{c: np.asarray(o[str(c)]["bel"], float) for c in cands} for o in OUT]

    # ---- direct baselines: LOO-CV trên eval, predict bel_c từ (z,c) ----
    def direct_pred_all(quad):
        rows_f, rows_y, rows_q = [], [], []
        for i, o in enumerate(OUT):
            for c in [-1] + cands:                                # dùng cả no-def làm dữ liệu học
                oh = np.zeros(Kans + 1); oh[c + 1] = 1.0
                rows_f.append(np.concatenate([feats(Z[i], Kans, quad), oh]))
                rows_y.append(np.asarray(o[str(c)]["bel"], float)); rows_q.append(i)
        F = np.array(rows_f); Yb = np.array(rows_y); Q = np.array(rows_q)
        pred = {}                                                # pred[(i,c)] = bel-vector
        for i in range(len(ev)):
            m = Q != i; W = ridge_fit(F[m], Yb[m])
            for c in [-1] + cands:
                oh = np.zeros(Kans + 1); oh[c + 1] = 1.0
                pred[(i, c)] = np.concatenate([feats(Z[i], Kans, quad), oh]) @ W
        return pred
    dlin = direct_pred_all(False); dquad = direct_pred_all(True)

    # climatology effect (LOO): mean effect theo kênh
    def meaneff_pred():
        pred = {}
        for i in range(len(ev)):
            for c in cands:
                eff = np.mean([belc[j][c] - bel0[j] for j in range(len(ev)) if j != i], axis=0)
                pred[(i, c)] = bel0[i] + eff                     # cần bel0[i]? -> dùng cho selection kém; hợp lý cho effect
        return pred
    meff = meaneff_pred()

    # ---- các hàm f(i,c)->bel ----
    def F_koop(i, c): return kpred(Z[i], c)
    def F_persist(i, c): return mean_belief(Z[i], N, Kans)
    METH = {"koopman": F_koop, "direct_lin": lambda i, c: dlin[(i, c)],
            "direct_quad": lambda i, c: dquad[(i, c)], "meaneff": lambda i, c: meff[(i, c)],
            "persist": F_persist}

    # ---- TASK A: effect-prediction (adv-bel effect) ----
    resA = {}
    for name, F in METH.items():
        pe, ae = [], []
        for i in range(len(ev)):
            b0p = F(i, -1) if name in ("koopman", "direct_lin", "direct_quad") else mean_belief(Z[i], N, Kans)
            for c in cands:
                pe.append(float(F(i, c)[X[i]] - b0p[X[i]]))       # predicted Δadv
                ae.append(float(belc[i][c][X[i]] - bel0[i][X[i]]))  # actual Δadv
        pe, ae = np.array(pe), np.array(ae)
        r = np.corrcoef(pe, ae)[0, 1] if pe.std() > 1e-9 else 0.0
        sse0 = np.sum(ae ** 2); sse = np.sum((ae - pe) ** 2)
        resA[name] = (float(r), float(1 - sse / (sse0 + 1e-12)))   # (Pearson, skill vs zero-effect)

    # ---- TASK B: intervention selection (honest-margin objective) ----
    def sel_outcomes(pick_c):
        acc, adv = [], []
        for i in range(len(ev)):
            c = pick_c(i); acc.append(int(np.argmax(belc[i][c]) == AST[i])); adv.append(float(belc[i][c][X[i]]))
        return np.array(acc), np.array(adv)
    def picker(F):
        def g(i):
            ch = honest_lead(Z[i], X[i], Kans)
            return max(cands, key=lambda c: F(i, c)[ch] - F(i, c)[X[i]])
        return g
    resB = {}
    for name, F in METH.items():
        a, v = sel_outcomes(picker(F)); resB[name] = (float(a.mean()), float(v.mean()))
    # oracle + no-def + random
    a, v = sel_outcomes(lambda i: max(cands, key=lambda c: belc[i][c][honest_lead(Z[i], X[i], Kans)] - belc[i][c][X[i]]))
    resB["ORACLE"] = (float(a.mean()), float(v.mean()))
    nd_acc = np.array([int(np.argmax(bel0[i]) == AST[i]) for i in range(len(ev))])
    resB["no_def"] = (float(nd_acc.mean()), float(np.mean([bel0[i][X[i]] for i in range(len(ev))])))
    ra = np.mean([[int(np.argmax(belc[i][c]) == AST[i]) for c in cands] for i in range(len(ev))])
    resB["random"] = (float(ra), float(np.mean([[belc[i][c][X[i]] for c in cands] for i in range(len(ev))])))
    return resA, resB, len(ev), Kans

def model_of(tag):
    for m in MODELS:
        if tag.endswith(m): return m
    return "?"

# ---------- main ----------
tags = {}
for D in DIRS:
    for f in glob.glob(os.path.join(D, "ckpt_twindef_collect_*.jsonl")):
        tag = os.path.basename(f)[len("ckpt_twindef_collect_"):-len(".jsonl")]
        ce = os.path.join(D, f"ckpt_twindef_eval_{tag}.jsonl")
        if os.path.exists(ce): tags[tag] = (f, ce)

by_model = {}
print("===== RESPONSE-OPERATOR TEST (ΔK_u) — per tag =====")
for tag, (cc, ce) in sorted(tags.items(), key=lambda kv: (model_of(kv[0]), kv[0])):
    try:
        A, B, n, K = run_tag(cc, ce)
    except Exception as e:
        print(f"[skip] {tag}: {e}"); continue
    by_model.setdefault(model_of(tag), []).append((A, B, n))
    print(f"\n[{tag} n={n}] TASK-A effect corr | TASK-B sel-acc:")
    print("   A corr:  " + "  ".join(f"{k}={A[k][0]:+.2f}" for k in ["koopman", "direct_lin", "direct_quad", "meaneff", "persist"]))
    print("   B acc:   " + "  ".join(f"{k}={B[k][0]:.2f}" for k in ["koopman", "direct_lin", "direct_quad", "ORACLE", "no_def", "random"]))

print("\n===== POOL theo model =====")
for m, L in by_model.items():
    keysA = ["koopman", "direct_lin", "direct_quad", "meaneff", "persist"]
    A = {k: np.mean([x[0][k][0] for x in L]) for k in keysA}
    Bacc = {k: np.mean([x[1][k][0] for x in L]) for k in ["koopman", "direct_lin", "direct_quad", "ORACLE", "no_def", "random"]}
    teeth = A["koopman"] > max(A["direct_lin"], A["direct_quad"]) + 1e-9 and \
            Bacc["koopman"] > max(Bacc["direct_lin"], Bacc["direct_quad"]) + 1e-9
    print(f"\n[{m}] ({len(L)} datasets)")
    print("  A effect-corr: " + "  ".join(f"{k}={A[k]:+.2f}" for k in keysA))
    print("  B sel-acc:     " + "  ".join(f"{k}={Bacc[k]:.2f}" for k in ["koopman", "direct_lin", "direct_quad", "ORACLE", "no_def", "random"]))
    print(f"  ⇒ koopman VƯỢT direct(lin&quad) ở CẢ effect-corr & sel-acc? {'✅ CÓ teeth' if teeth else '❌ direct ngang/hơn — why-Koopman'}")
print("\n[đọc] teeth chỉ khi koopman > direct_lin VÀ direct_quad ở CẢ 2 task. Thắng persist/meaneff = KHÔNG đủ.")
