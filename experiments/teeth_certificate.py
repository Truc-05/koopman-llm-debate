"""FEASIBILITY: operator (A,B) có 'RĂNG' không? — per-question manipulability CERTIFICATE.
Câu hỏi: điểm controllability/reachability suy từ operator có DỰ ĐOÁN được câu nào bị lái (manipulable)
KHÔNG, và có VƯỢT baseline tầm thường (persistence) không? (operator<persistence = chết, đã thấy ở fidelity.)
KHÔNG chạy debate mới — chỉ đọc ckpt collect (fit A,B) + eval (z_kdef, outs). numpy-only.

Dùng (user tự chạy):
  python experiments/teeth_certificate.py                 # quét cả out/ và out_qwen/
  python experiments/teeth_certificate.py out             # chỉ 1 thư mục
Đọc: nếu Δ(operator−persistence) CI[95%] > 0 ở ≥1 test ⇒ operator có răng (mở rộng + viết certificate-theorem).
     nếu CI chồng 0 khắp nơi ⇒ xác nhận operator không thêm gì ⇒ dừng, khỏi tốn công.
"""
import sys, os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief, split_state

N, K_DEF = 4, 2                                  # n_agents, round bắt đầu defense (khớp pipeline)
MODELS = ["mistral_7b", "llama3_1_8b", "gemma2_9b", "qwen2_5_7b"]
DIRS = sys.argv[1:] or ["experiments/out", "experiments/out_qwen"]
DIRS = [os.path.join(ROOT, d) if not os.path.isabs(d) else d for d in DIRS]

# ---------- stats numpy-only ----------
def _rank(v):
    v = np.asarray(v, float); order = v.argsort(); r = np.empty(len(v)); r[order] = np.arange(len(v))
    # average ties
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum - 1) / 2.0
    return avg[inv]

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])

def auc(score, label):                            # AUC score dự đoán label∈{0,1} (Mann-Whitney)
    score, label = np.asarray(score, float), np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _rank(score);
    return float((r[label == 1].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))

def boot_ci(fn, n_boot=2000, seed_arrs=None):     # CI cho 1 thống kê vô hướng, resample index
    m = len(seed_arrs[0]); idx0 = np.arange(m)
    base = fn(idx0)
    # deterministic bootstrap (không dùng Math.random ở python vẫn OK, nhưng cố định seed để lặp lại)
    rng = np.random.default_rng(12345)
    vals = np.array([fn(rng.integers(0, m, m)) for _ in range(n_boot)])
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return base, lo, hi

# ---------- fit (A,B) + rollout đúng pipeline ----------
def fit_ops(col):
    D = np.asarray(col[0]["traj"]).shape[1]; Kans = D // N
    T = np.asarray(col[0]["traj"]).shape[0] - 1
    d = PolynomialDictionary(degree=1); d.transform(np.asarray(col[0]["traj"])[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for r in col:
        Z = np.asarray(r["traj"], float)
        for t in range(len(Z) - 1):
            u = np.zeros(Kans)
            if r["c"] >= 0 and t >= K_DEF:
                u[r["c"]] = 1.0
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    Kop, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    return Kop, B, d, ss, Kans, T

def pred_belief(z_kdef, c, Kop, B, d, ss, Kans, T):
    psi = d.transform(np.asarray(z_kdef).reshape(1, -1))[0]
    for _ in range(K_DEF, T):
        u = np.zeros(Kans)
        if c >= 0:
            u[c] = 1.0
        psi = Kop @ psi + B @ u
    return mean_belief(np.real(psi[ss]), N, Kans)

def honest_lead(z, x_adv, Kans):
    _, p = split_state(np.asarray(z), N, Kans)
    h = p[1:].mean(axis=0).astype(float); h[x_adv] = -np.inf
    return int(np.argmax(h))

# ---------- gom signal per-question mọi tag ----------
def collect_signals(cc, ce):
    col = [json.loads(l) for l in open(cc)]
    ev = [json.loads(l) for l in open(ce)]
    Kop, B, d, ss, Kans, T = fit_ops(col)
    cands = [-1] + list(range(Kans))
    rows = []
    for r in ev:
        z, x, a_star, outs = r["z_kdef"], r["x_adv"], r["a_star"], r["outs"]
        if z is None:
            continue
        c_hon = honest_lead(z, x, Kans)
        sc = lambda bel: float(bel[c_hon] - bel[x])           # honest-margin objective (label-free)
        op_nodef = float(pred_belief(z, -1, Kop, B, d, ss, Kans, T)[x])
        persist = float(mean_belief(np.asarray(z), N, Kans)[x])
        op_best = max(sc(pred_belief(z, c, Kop, B, d, ss, Kans, T)) for c in cands)
        act_nodef = float(outs["-1"]["adv_bel"])
        act_best = max(sc(np.asarray(outs[str(c)]["bel"])) for c in cands)
        manip = 0 if outs["-1"]["correct"] else 1              # adversary thắng (no-def) = bị lái
        rows.append((op_nodef, persist, act_nodef, op_best, act_best, manip))
    return np.array(rows, float) if rows else np.zeros((0, 6))

def model_of(tag):
    for m in MODELS:
        if tag.endswith(m):
            return m
    return "?"

# ---------- main ----------
tags = {}
for D in DIRS:
    for f in glob.glob(os.path.join(D, "ckpt_twindef_collect_*.jsonl")):
        tag = os.path.basename(f)[len("ckpt_twindef_collect_"):-len(".jsonl")]
        ce = os.path.join(D, f"ckpt_twindef_eval_{tag}.jsonl")
        if os.path.exists(ce):
            tags[(D, tag)] = (f, ce)

print(f"===== TEETH TEST: operator certificate vs persistence  ({len(tags)} tag) =====")
print("cols: op_nodef persist | act_nodef | op_best act_best | manip(0/1)\n")

ALL = {}
for (D, tag), (cc, ce) in sorted(tags.items(), key=lambda kv: (model_of(kv[0][1]), kv[0][1])):
    try:
        S = collect_signals(cc, ce)
    except Exception as e:
        print(f"[skip] {tag}: {e}")
        continue
    if len(S) < 8:
        continue
    m = model_of(tag)
    ALL.setdefault(m, []).append(S)
    ALL.setdefault("__POOL__", []).append(S)
    op_n, per, act_n, op_b, act_b, man = S.T
    r_op = spearman(op_n, act_n); r_pe = spearman(per, act_n)
    print(f"[{tag:28s}] n={len(S):2d}  reach: ρ_op={r_op:+.2f} ρ_persist={r_pe:+.2f} Δ={r_op-r_pe:+.2f}"
          f" | cert: ρ(op_best,act_best)={spearman(op_b,act_b):+.2f}"
          f" | manip-AUC op={auc(op_n,man):.2f} persist={auc(per,man):.2f}")

def report_pool(name, mats):
    S = np.vstack(mats); op_n, per, act_n, op_b, act_b, man = S.T
    n = len(S)
    # Δ Spearman (reach certificate) operator - persistence, bootstrap CI
    def drho(idx):
        return spearman(op_n[idx], act_n[idx]) - spearman(per[idx], act_n[idx])
    d, lo, hi = boot_ci(drho, seed_arrs=[op_n])
    # Δ AUC (manipulability classify) operator - persistence
    def dauc(idx):
        return auc(op_n[idx], man[idx]) - auc(per[idx], man[idx])
    da, alo, ahi = boot_ci(dauc, seed_arrs=[op_n])
    teeth = "✅ RĂNG" if lo > 0 or alo > 0 else "❌ không thêm"
    print(f"\n== POOL [{name}] n={n} ==")
    print(f"  reach ρ: op={spearman(op_n,act_n):+.2f} persist={spearman(per,act_n):+.2f}"
          f"  Δ={d:+.3f} CI[{lo:+.3f},{hi:+.3f}]")
    print(f"  manip AUC: op={auc(op_n,man):.3f} persist={auc(per,man):.3f}"
          f"  Δ={da:+.3f} CI[{alo:+.3f},{ahi:+.3f}]")
    print(f"  defend-cert ρ(op_best,act_best)={spearman(op_b,act_b):+.2f}  ⇒ {teeth}")

for m in MODELS:
    if m in ALL:
        report_pool(m, ALL[m])
if "__POOL__" in ALL:
    report_pool("ALL MODELS", ALL["__POOL__"])
print("\n[đọc] Δ>0 & CI-lower>0 = operator VƯỢT persistence ⇒ certificate có răng (mở rộng+viết theorem).")
print("      CI chồng 0 khắp = operator ≈ persistence ⇒ xác nhận không răng ⇒ dừng.")
