"""Controlled-Response Benchmark (#1) — chứng minh Koopman đóng góp qua B (đáp ứng can thiệp),
KHÔNG qua A (autonomous evolution). Đọc ckpt_twindef_collect: mỗi debate có traj + defense c
(control u_t = one-hot(c) khi t>=K_DEF). So OOS (5-fold cross-fit) belief-MSE của mọi model theo
2 chế độ: AUTONOMOUS (bỏ u) vs CONTROLLED (dùng u).

CHỐT ĐỌC (đã bàn):
 - degree-1 ⇒ Koopman-d1 ≡ Linear-ARX (CÙNG model). KHÔNG claim "Koopman>AR".
 - V1 input-helps : Koopman-d1(ctrl) < persistence & DeGroot & Linear-AR(auto)  ⇒ B học được ⇒ why-control.
 - V2 linear-đủ  : Koopman-d1(ctrl) ≈ MLP-ctrl(nonlinear)                        ⇒ linearity suffices ⇒ why-Koopman.
 - V3 lift-thừa  : Koopman-d2(ctrl) ≈/tệ hơn d1                                    ⇒ degree-1 đủ.
 - persistence rollout bé tí ⇒ belief bất động ⇒ autonomous tầm thường (con ngựa chết) ⇒ chỉ controlled đáng kể.

KHÔNG gọi LLM. Dùng: python experiments/controlled_response_benchmark.py [tag]
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
N = Kans = 4
K_DEF = 2
REG = 1e-6
NFOLD = 5

recs = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_twindef_collect_{tag}.jsonl"))]
trajs = [np.asarray(r["traj"], float) for r in recs]
cs = [int(r["c"]) for r in recs]
n = len(trajs); T = trajs[0].shape[0] - 1
fold = np.arange(n) % NFOLD


def u_of(i, t):
    u = np.zeros(Kans)
    if cs[i] >= 0 and t >= K_DEF:
        u[cs[i]] = 1.0
    return u


def agent_mean(z):                                     # đẩy mỗi agent về trung bình các agent (consensus)
    p = z.reshape(N, Kans)
    return np.repeat(p.mean(axis=0, keepdims=True), N, axis=0).reshape(-1)


def snapshots(idx):
    X, Y, U = [], [], []
    for i in idx:
        Z = trajs[i]
        for t in range(T):
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u_of(i, t))
    return np.array(X), np.array(Y), np.array(U)


# ---- fitters: trả dict(onestep(x,u)->bel4, rollout(z0,us)->bel4) ----
def fit_persistence(Xtr, Ytr, Utr):
    return dict(onestep=lambda x, u: mean_belief(x, N, Kans),
                rollout=lambda z0, us: mean_belief(z0, N, Kans))


def fit_degroot(Xtr, Ytr, Utr):                        # z' = (1-a)z + a·agent_mean(z), fit a (LS)
    D = np.array([agent_mean(x) for x in Xtr]) - Xtr; R = Ytr - Xtr
    a = float(np.clip((D * R).sum() / ((D * D).sum() + 1e-12), 0, 1))
    step = lambda z: (1 - a) * z + a * agent_mean(z)
    def roll(z0, us):
        z = z0.copy()
        for _ in us:
            z = step(z)
        return mean_belief(z, N, Kans)
    return dict(onestep=lambda x, u: mean_belief(step(x), N, Kans), rollout=roll, a=a)


def fit_koop(Xtr, Ytr, Utr, degree, use_u):
    d = PolynomialDictionary(degree=degree); d.transform(Xtr[:1]); ss = d.state_slice
    U = Utr.T if use_u else np.zeros_like(Utr.T)
    Kop, B = fit_edmdc(d.transform(Xtr).T, d.transform(Ytr).T, U, reg=REG)
    bu = (lambda u: B @ u) if use_u else (lambda u: 0.0)
    def onestep(x, u):
        psi = d.transform(x.reshape(1, -1))[0]
        return mean_belief(np.real((Kop @ psi + bu(u))[ss]), N, Kans)
    def roll(z0, us):
        psi = d.transform(z0.reshape(1, -1))[0]
        for u in us:
            psi = Kop @ psi + bu(u)
        return mean_belief(np.real(psi[ss]), N, Kans)
    return dict(onestep=onestep, rollout=roll)


def fit_mlp(Xtr, Ytr, Utr, use_u, hidden=(32,), es=False):
    from sklearn.neural_network import MLPRegressor
    In = np.hstack([Xtr, Utr]) if use_u else Xtr
    mu = In.mean(0); sd = In.std(0) + 1e-9
    m = MLPRegressor(hidden_layer_sizes=hidden, max_iter=3000, random_state=0,
                     early_stopping=es).fit((In - mu) / sd, Ytr)
    feat = lambda x, u: (((np.hstack([x, u]) if use_u else x) - mu) / sd).reshape(1, -1)
    def roll(z0, us):
        z = z0.copy()
        for u in us:
            z = m.predict(feat(z, u))[0]
        return mean_belief(z, N, Kans)
    return dict(onestep=lambda x, u: mean_belief(m.predict(feat(x, u))[0], N, Kans), rollout=roll)


MODELS = [                                             # (tên, chế-độ, fitter)
    ("persistence", "auton", fit_persistence),
    ("DeGroot",     "auton", fit_degroot),
    ("Linear-AR",   "auton", lambda X, Y, U: fit_koop(X, Y, U, 1, False)),
    ("MLP-auto",    "auton", lambda X, Y, U: fit_mlp(X, Y, U, False)),
    ("Koopman-d1*",  "ctrl",  lambda X, Y, U: fit_koop(X, Y, U, 1, True)),   # * ≡ Linear-ARX
    ("Koopman-d2",   "ctrl",  lambda X, Y, U: fit_koop(X, Y, U, 2, True)),
    ("MLP-ctrl(16)", "ctrl",  lambda X, Y, U: fit_mlp(X, Y, U, True, (16,))),
    ("MLP-ctrl(32)", "ctrl",  lambda X, Y, U: fit_mlp(X, Y, U, True, (32,))),
    ("MLP-ctrl(64)", "ctrl",  lambda X, Y, U: fit_mlp(X, Y, U, True, (64,))),
    ("MLP-ctrl(es)", "ctrl",  lambda X, Y, U: fit_mlp(X, Y, U, True, (32,), True)),  # early-stopping (chống overfit)
]

res = {name: {"one": [], "roll": [], "lead": []} for name, _, _ in MODELS}
for f in range(NFOLD):
    tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
    Xtr, Ytr, Utr = snapshots(tr)
    fitted = {name: fn(Xtr, Ytr, Utr) for name, _, fn in MODELS}
    for i in te:
        Z = trajs[i]
        for t in range(T):
            u = u_of(i, t); yb = mean_belief(Z[t + 1], N, Kans)
            for name, _, _ in MODELS:
                res[name]["one"].append(np.mean((fitted[name]["onestep"](Z[t], u) - yb) ** 2))
        us = [u_of(i, t) for t in range(K_DEF, T)]; fb = mean_belief(Z[-1], N, Kans)
        for name, _, _ in MODELS:
            pb = fitted[name]["rollout"](Z[K_DEF], us)
            res[name]["roll"].append(np.mean((pb - fb) ** 2))
            res[name]["lead"].append(int(np.argmax(pb) == np.argmax(fb)))

m = {name: (np.mean(res[name]["one"]), np.mean(res[name]["roll"]), np.mean(res[name]["lead"])) for name, _, _ in MODELS}

print(f"\n===== CONTROLLED-RESPONSE BENCHMARK  (tag={tag}, n={n} debate, T={T}, {NFOLD}-fold OOS) =====")
print(" u = one-hot(defense c) khi t>=K_DEF. rollout: cuộn từ round K_DEF tới cuối. Lỗi đo trên mean-belief.\n")
print(f"    {'model':13} {'chế-độ':7} {'1step-MSE':>10} {'rollout-MSE':>12} {'lead@roll':>10}")
for name, reg, _ in MODELS:
    o, r, l = m[name]
    print(f"    {name:13} {reg:7} {o:10.4f} {r:12.4f} {l:10.3f}")
print("    (* Koopman-d1 ≡ Linear-ARX: cùng model — cột controlled của họ tuyến-tính)")

kd1 = m["Koopman-d1*"]; kd2 = m["Koopman-d2"]
mlp_names = [nm for nm, rg, _ in MODELS if nm.startswith("MLP-ctrl")]
best_mlp = min(mlp_names, key=lambda nm: m[nm][1]); bm = m[best_mlp][1]
auton_roll = {k: m[k][1] for k in ("persistence", "DeGroot", "Linear-AR")}
print("\n---- VERDICT ----")
print(f" persistence rollout-MSE = {m['persistence'][1]:.4f}"
      f"  ⇒ {'belief BẤT ĐỘNG → autonomous tầm thường (con ngựa chết)' if m['persistence'][1] < 0.02 else 'belief có dịch chuyển'}")
beat = all(kd1[1] < v for v in auton_roll.values())
print(f" V1 input-helps  : Koopman-d1(ctrl) rollout={kd1[1]:.4f} vs " +
      ", ".join(f"{k}={v:.4f}" for k, v in auton_roll.items()) +
      f"  ⇒ {'B HỌC ĐƯỢC (thắng mọi autonomous) ✓' if beat else 'B CHƯA thắng autonomous ✗ — controlled không ăn tiền'}")
ratio = kd1[1] / (bm + 1e-12)
print(f" V2 linear-đủ    : Koopman-d1={kd1[1]:.4f} vs BEST-of-{len(mlp_names)} nonlinear ({best_mlp})={bm:.4f}  (tỉ số {ratio:.2f})"
      f"  ⇒ {'linear ≥ best-nonlinear → LINEARITY SUFFICES ✓ (bulletproof: đã steelman nonlinear)' if ratio <= 1.05 else ('linear ≈ nonlinear (≤25%) → largely suffices' if ratio <= 1.25 else 'nonlinear hơn RÕ → HẠ GIỌNG V2')}")
rk = np.array(res["Koopman-d1*"]["roll"]); rb = np.array(res[best_mlp]["roll"]); diff = rk - rb  # >0 = Koopman TỆ hơn
_rng = np.random.default_rng(0); _nD = len(diff)
_bd = np.array([diff[_rng.integers(0, _nD, _nD)].mean() for _ in range(5000)])
_lo, _hi = np.percentile(_bd, [2.5, 97.5])
print(f"    ├─ paired Δ(Koopman-d1 − {best_mlp}) rollout-MSE = {diff.mean():+.4f}  CI[{_lo:+.4f},{_hi:+.4f}]"
      f"  ⇒ {'SIG: nonlinear thật hơn → giữ ~10% gap' if _lo > 0 else 'n.s.: 10% = NOISE → viết mạnh -linear suffices, no sig diff-'}")
print(f" V3 lift-thừa    : Koopman-d2={kd2[1]:.4f} vs d1={kd1[1]:.4f}"
      f"  ⇒ {'degree-1 đủ (d2 không hơn) → đừng claim Koopman>AR ✓' if kd2[1] >= kd1[1] * 0.95 else 'd2 HƠN d1 → lift ăn tiền, có thể claim Koopman-native'}")
