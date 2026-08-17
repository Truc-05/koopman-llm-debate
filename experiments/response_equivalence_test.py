"""XÁC NHẬN CUỐI: Koopman ΔK_u ≈ direct-regression? (equivalence, KHÔNG chỉ CI-chồng-0). KHÔNG gọi LLM.
Paired cluster-bootstrap theo QUESTION (koopman & direct đánh giá trên cùng mẫu). 3 chênh lệch:
  Δ_effect = r_koop − r_direct   (Pearson effect-corr)
  Δ_select = Acc_koop − Acc_direct
  Δ_oracle = Acc_oracle − Acc_koop   (kiểm 'near-oracle')
TOST equivalence (biên đặt-trước): effect ±0.05, select ±0.03  ⇒ 90% CI ⊂ (−m,+m) thì KẾT LUẬN tương đương.
Thêm paired McNemar (koop vs direct, selection). Báo per-model + grand-pooled cluster-bootstrap.
Dùng: python experiments/response_equivalence_test.py
"""
import os, sys, json, glob
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief, split_state

N, K_DEF = 4, 2
MODELS = ["mistral_7b", "llama3_1_8b", "gemma2_9b", "qwen2_5_7b"]
DIRS = [os.path.join(ROOT, "experiments", d) for d in ["out", "out_qwen"]]
M_EFF, M_SEL = 0.05, 0.03            # biên equivalence
B = 5000

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
    Kop, Bm = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    def pred(z, c):
        psi = d.transform(np.asarray(z).reshape(1, -1))[0]
        for _ in range(K_DEF, T):
            u = np.zeros(Kans)
            if c >= 0: u[c] = 1.0
            psi = Kop @ psi + Bm @ u
        return mean_belief(np.real(psi[ss]), N, Kans)
    return pred, Kans

def honest_lead(z, x, Kans):
    _, p = split_state(np.asarray(z), N, Kans); h = p[1:].mean(0).astype(float); h[x] = -np.inf
    return int(np.argmax(h))

def feats(z, Kans, quad):
    z = np.asarray(z, float); b = mean_belief(z, N, Kans); f = np.concatenate([z, b])
    return np.concatenate([f, b ** 2]) if quad else f

def ridge_fit(X, Y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)

def collect_tag(cc, ce):
    """Trả list per-question: {rows_pk,rows_pd,rows_a (theo channel), koop,dlin,dquad,oracle,nodef (bool)}."""
    col = [json.loads(l) for l in open(cc)]; ev = [json.loads(l) for l in open(ce)]
    ev = [r for r in ev if r.get("z_kdef") is not None]
    kpred, Kans = fit_koopman(col); cands = list(range(Kans))
    Z = [np.asarray(r["z_kdef"], float) for r in ev]; Xa = [r["x_adv"] for r in ev]; AST = [r["a_star"] for r in ev]
    O = [r["outs"] for r in ev]
    bel0 = [np.asarray(o["-1"]["bel"], float) for o in O]
    belc = [{c: np.asarray(o[str(c)]["bel"], float) for c in cands} for o in O]
    # direct ridge (LOO-CV) predict bel_c
    def direct(quad):
        F, Yb, Q = [], [], []
        for i, o in enumerate(O):
            for c in [-1] + cands:
                oh = np.zeros(Kans + 1); oh[c + 1] = 1.0
                F.append(np.concatenate([feats(Z[i], Kans, quad), oh])); Yb.append(np.asarray(o[str(c)]["bel"], float)); Q.append(i)
        F, Yb, Q = np.array(F), np.array(Yb), np.array(Q); pr = {}
        for i in range(len(ev)):
            W = ridge_fit(F[Q != i], Yb[Q != i])
            for c in [-1] + cands:
                oh = np.zeros(Kans + 1); oh[c + 1] = 1.0
                pr[(i, c)] = np.concatenate([feats(Z[i], Kans, quad), oh]) @ W
        return pr
    dlin, dquad = direct(False), direct(True)
    def pick(Fbel, i):
        ch = honest_lead(Z[i], Xa[i], Kans); return max(cands, key=lambda c: Fbel(i, c)[ch] - Fbel(i, c)[Xa[i]])
    Fk = lambda i, c: kpred(Z[i], c); Fl = lambda i, c: dlin[(i, c)]; Fq = lambda i, c: dquad[(i, c)]
    rec = []
    for i in range(len(ev)):
        x = Xa[i]
        pk = [float(Fk(i, c)[x] - Fk(i, -1)[x]) for c in cands]
        pd = [float(Fl(i, c)[x] - Fl(i, -1)[x]) for c in cands]
        a = [float(belc[i][c][x] - bel0[i][x]) for c in cands]
        ck, cl, cq = pick(Fk, i), pick(Fl, i), pick(Fq, i)
        co = max(cands, key=lambda c: belc[i][c][honest_lead(Z[i], x, Kans)] - belc[i][c][x])
        rec.append(dict(pk=pk, pd=pd, a=a,
                        koop=int(np.argmax(belc[i][ck]) == AST[i]), dlin=int(np.argmax(belc[i][cl]) == AST[i]),
                        dquad=int(np.argmax(belc[i][cq]) == AST[i]), oracle=int(np.argmax(belc[i][co]) == AST[i]),
                        nodef=int(np.argmax(bel0[i]) == AST[i])))
    return rec

def pear(pred, act):
    pred, act = np.asarray(pred), np.asarray(act)
    return float(np.corrcoef(pred, act)[0, 1]) if pred.std() > 1e-9 and act.std() > 1e-9 else 0.0

def analyze(name, recs):
    n = len(recs)
    PK = [np.array(r["pk"]) for r in recs]; PD = [np.array(r["pd"]) for r in recs]; A = [np.array(r["a"]) for r in recs]
    koop = np.array([r["koop"] for r in recs]); dlin = np.array([r["dlin"] for r in recs])
    orac = np.array([r["oracle"] for r in recs]); nod = np.array([r["nodef"] for r in recs])
    def stat(idx):
        pk = np.concatenate([PK[i] for i in idx]); pd = np.concatenate([PD[i] for i in idx]); a = np.concatenate([A[i] for i in idx])
        de = pear(pk, a) - pear(pd, a)
        ds = koop[idx].mean() - dlin[idx].mean()
        do = orac[idx].mean() - koop[idx].mean()
        return de, ds, do
    de0, ds0, do0 = stat(np.arange(n))
    rng = np.random.default_rng(11)
    BS = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    def ci(col, lo, hi): return float(np.percentile(BS[:, col], lo)), float(np.percentile(BS[:, col], hi))
    # McNemar koop vs dlin (paired selection)
    b01 = int(np.sum((koop == 0) & (dlin == 1))); b10 = int(np.sum((koop == 1) & (dlin == 0))); md = b01 + b10
    pmc = 1.0 if md == 0 else min(1.0, 2 * sum(comb(md, i) for i in range(min(b01, b10) + 1)) * 0.5 ** md)
    acc_k, acc_o, acc_nd = float(koop.mean()), float(orac.mean()), float(nod.mean())
    head = (acc_k - acc_nd) / (acc_o - acc_nd) if acc_o > acc_nd else float("nan")
    print(f"\n===== [{name}] n_q={n} =====")
    # effect
    l90, h90 = ci(0, 5, 95); l95, h95 = ci(0, 2.5, 97.5)
    eq = "✅ EQUIVALENT" if (l90 > -M_EFF and h90 < M_EFF) else ("↔ no-adv (CI∋0)" if l95 < 0 < h95 else "direct KHÁC")
    print(f"  Δ_effect (r_koop−r_direct) = {de0:+.3f}  95%CI[{l95:+.3f},{h95:+.3f}]  90%CI[{l90:+.3f},{h90:+.3f}]  TOST±{M_EFF} ⇒ {eq}")
    # select
    l90, h90 = ci(1, 5, 95); l95, h95 = ci(1, 2.5, 97.5)
    eq = "✅ EQUIVALENT" if (l90 > -M_SEL and h90 < M_SEL) else ("↔ no-adv (CI∋0)" if l95 < 0 < h95 else "direct KHÁC")
    print(f"  Δ_select (Acc_koop−Acc_direct) = {ds0:+.3f}  95%CI[{l95:+.3f},{h95:+.3f}]  90%CI[{l90:+.3f},{h90:+.3f}]  TOST±{M_SEL} ⇒ {eq}")
    print(f"    McNemar koop-vs-direct: +{b10}/−{b01}  p={pmc:.3f}")
    # oracle
    l95, h95 = ci(2, 2.5, 97.5)
    nearor = "SÁT oracle (CI∋0)" if l95 < 0 < h95 else f"CÒN GAP → 'recovers {head*100:.0f}% headroom'"
    print(f"  Δ_oracle (Acc_oracle−Acc_koop) = {do0:+.3f}  95%CI[{l95:+.3f},{h95:+.3f}]  ⇒ {nearor}")
    print(f"    acc: koop={acc_k:.2f} oracle={acc_o:.2f} no_def={acc_nd:.2f}  headroom-recovered={head*100:.0f}%")

def model_of(tag):
    for m in MODELS:
        if tag.endswith(m): return m
    return "?"

tags = {}
for D in DIRS:
    for f in glob.glob(os.path.join(D, "ckpt_twindef_collect_*.jsonl")):
        tag = os.path.basename(f)[len("ckpt_twindef_collect_"):-len(".jsonl")]
        ce = os.path.join(D, f"ckpt_twindef_eval_{tag}.jsonl")
        if os.path.exists(ce): tags[tag] = (f, ce)

by_model, allrec = {}, []
print("=== gom per-question (Koopman fit collect / direct LOO-CV eval) ===")
for tag, (cc, ce) in sorted(tags.items(), key=lambda kv: (model_of(kv[0]), kv[0])):
    try:
        rec = collect_tag(cc, ce)
    except Exception as e:
        print(f"[skip] {tag}: {e}"); continue
    by_model.setdefault(model_of(tag), []).extend(rec); allrec.extend(rec)
    print(f"  {tag}: {len(rec)} q")

print("\n########## PER-MODEL (cluster-bootstrap theo question) ##########")
for m in MODELS:
    if m in by_model: analyze(m, by_model[m])
print("\n########## GRAND-POOLED ##########")
analyze("ALL MODELS pooled", allrec)
print(f"\n[đọc] TOST: 90%CI ⊂ (−m,+m) ⇒ EQUIVALENT (Koopman collapse về static response). CI∋0 mà không đạt biên ⇒ 'no consistent advantage'."
      f" Δ_select<0 & CI toàn âm ⇒ direct THẮNG ⇒ 'dynamic factorization adds no value'. Δ_oracle CI∌0 ⇒ dùng 'recovers X% headroom' KHÔNG 'near-oracle'.")
