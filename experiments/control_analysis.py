"""Module #1 — PHÂN TÍCH CONTROL-THEORETIC của operator (A,B) đã nhận dạng (offline, KHÔNG debate mới).
Biến bài từ "đo accuracy" → "phân tích hệ" (venue-fit Cybernetics). Fit (K,B) degree-1 trên ckpt collect
(u = one-hot defense khi t>=K_DEF), rồi báo: (1) spectrum/stability, (2) controllability (rank + Gramian),
(3) reachability per-target. LQR/MPC model-based đã có = twin-MPC (0.49 vs oracle 0.60) — trích lại, không lặp.
Dùng: python experiments/control_analysis.py [tag]     KHÔNG tự chạy — user tự chạy.
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
K_DEF = 2
col = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_twindef_collect_{tag}.jsonl"))]
N = 4                                        # n_agents (debate)
D = np.asarray(col[0]["traj"]).shape[1]      # = N*Kans (16 cho Kans=4, 20 cho Kans=5)
Kans = D // N

d = PolynomialDictionary(degree=1); d.transform(np.asarray(col[0]["traj"])[:1]); ss = d.state_slice
X, Y, U = [], [], []
for r in col:
    Z = np.asarray(r["traj"], float)
    for t in range(len(Z) - 1):
        u = np.zeros(Kans)
        if r["c"] >= 0 and t >= K_DEF:
            u[r["c"]] = 1.0
        X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
K, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
T = np.asarray(col[0]["traj"]).shape[0] - 1

A = np.real(K[ss, ss])                                 # 16×16 state-transition block
Bs = np.real(B[ss, :])                                 # 16×Kans control block

print(f"\n===== CONTROL ANALYSIS of identified (A,B)  (tag={tag}, dim={D}, n_snap={len(X)}) =====")

# (1) SPECTRUM / STABILITY
lam = np.linalg.eigvals(A); rho = np.abs(lam)
n_in = int((rho <= 1.0 + 1e-9).sum())
print(f"\n[1] SPECTRUM  |λ|max={rho.max():.3f}  ({n_in}/{D} trong đĩa đơn vị)  "
      f"⇒ {'ỔN ĐỊNH (không nổ)' if rho.max() <= 1.0 + 1e-6 else 'CÓ mode nổ — cần reg/svd-rank'}")
print(f"    top-5 |λ|: " + ", ".join(f"{v:.3f}" for v in np.sort(rho)[::-1][:5]))

# (2) CONTROLLABILITY: rank ma trận controllability + Gramian hữu hạn horizon
Ck = Bs.copy(); blocks = [Bs]; Ak = np.eye(D)
for _ in range(1, D):
    Ak = A @ Ak; blocks.append(Ak @ Bs)
Cmat = np.hstack(blocks)
rank = np.linalg.matrix_rank(Cmat, tol=1e-6)
W = np.zeros((D, D)); Ak = np.eye(D)
for _ in range(T):
    W += Ak @ Bs @ Bs.T @ Ak.T; Ak = A @ Ak
w = np.linalg.eigvalsh(W); wmin, wmax = w.min(), w.max()
print(f"\n[2] CONTROLLABILITY  rank(𝒞)={rank}/{D}  ⇒ {'ĐỦ HẠNG (điều-khiển-được hoàn toàn)' if rank == D else 'thiếu hạng — subspace điều khiển được = '+str(rank)}")
cond = wmax / max(wmin, 1e-15)
print(f"    Gramian W_c (horizon {T}): λmin={wmin:.2e}  λmax={wmax:.2e}  eccentricity(cond)={cond:.1e}"
      f"  ⇒ {'gần ĐẲNG hướng' if cond < 1e3 else 'ANISOTROPIC mạnh — vài hướng gần UNcontrollable (λmin≪λmax)'}")

# (3) REACHABILITY per-target: đẩy từ uniform z0, belief[x] cuối
z0 = np.zeros(D); reach = []
for x in range(Kans):
    psi = d.transform(z0.reshape(1, -1))[0]; u = np.zeros(Kans); u[x] = 1.0
    for _ in range(T):
        psi = K @ psi + B @ u
    reach.append(mean_belief(np.real(psi[ss]), N, Kans)[x])
print(f"\n[3] REACHABILITY  belief[x] đạt được khi đẩy target x (từ uniform, {T} vòng):")
print("    " + "  ".join(f"x{ x}={b:.2f}" for x, b in enumerate(reach)) +
      f"   ⇒ controllability index (TB) = {np.mean(reach):.2f}  (gần 1 = mọi đáp án ép được)")

import collections
cdist = collections.Counter(int(r["c"]) for r in col)
bn = [np.linalg.norm(Bs[:, x]) for x in range(Kans)]
print(f"\n[4] DIAGNOSTIC x-thấp = thật hay artifact?")
print(f"    defense c count: " + "  ".join(f"c{c}={cdist.get(c, 0)}" for c in [-1, *range(Kans)]))
print(f"    ‖B[:,x]‖:        " + "  ".join(f"x{x}={bn[x]:.2f}" for x in range(Kans)))
print(f"    reach[x]:        " + "  ".join(f"x{x}={reach[x]:.2f}" for x in range(Kans)))
print(f"    ⇒ count thấp/‖B‖ nhỏ mà reach thấp = ARTIFACT nhận dạng (thu thêm/cân bằng c); "
      f"count đủ mà vẫn thấp = positional-bias THẬT của model.")

print(f"\n[note] Model-based controller (twin-MPC dùng chính (A,B)) đã đo: acc 0.49 vs oracle 0.60"
      f" ⇒ (A,B) đủ để THIẾT KẾ controller, closed-loop dưới oracle = open control-design.")
