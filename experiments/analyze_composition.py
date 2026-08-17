"""#1+#2 ANALYZE composition (KHÔNG gọi LLM). Train Koopman (A,B) trên collect single CŨ, test trên compositions.
#1 Unseen composition: dự đoán bel cuối của seq12/seq21 bằng Koopman-rollout theo schedule vs baseline
   'additive' & 'last-single' (= extrapolation của direct-single). Koopman thắng ⇒ 'why Koopman' được giải.
#2 Order/commutator: order-effect thật ‖bel(seq12)-bel(seq21)‖; Koopman có dự đoán được câu nào order-sensitive?
Additivity + phase classification. Đọc composition ckpt + collect (fit A,B, pool Kans=4 theo model).
Dùng: python experiments/analyze_composition.py [--model qwen2.5:7b]
"""
import os, sys, json, glob, argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief
N, K_DEF = 4, 2
DIRS = [os.path.join(ROOT, "experiments", d) for d in ["out", "out_qwen"]]

def fit_koopman(cols):
    D = np.asarray(cols[0][0]["traj"]).shape[1]; Kans = D // N; T = np.asarray(cols[0][0]["traj"]).shape[0] - 1
    d = PolynomialDictionary(degree=1); d.transform(np.asarray(cols[0][0]["traj"])[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for col in cols:
        for r in col:
            Z = np.asarray(r["traj"], float)
            for t in range(len(Z) - 1):
                u = np.zeros(Kans)
                if r["c"] >= 0 and t >= K_DEF: u[r["c"]] = 1.0
                X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    Kop, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    def roll(z_kdef, chans):                              # chans = list kênh cho vòng K_DEF..T-1
        psi = d.transform(np.asarray(z_kdef).reshape(1, -1))[0]
        for c in chans:
            u = np.zeros(Kans)
            if c >= 0: u[c] = 1.0
            psi = Kop @ psi + B @ u
        return mean_belief(np.real(psi[ss]), N, Kans)
    return roll, Kans, T

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="llama3.1:8b"); a = ap.parse_args()
    import re; mslug = re.sub(r"[^0-9a-zA-Z]+", "_", a.model).strip("_")
    # collect Kans=4 cùng model để fit
    cols = []
    for D in DIRS:
        for f in glob.glob(os.path.join(D, f"ckpt_twindef_collect_*_{mslug}.jsonl")):
            col = [json.loads(l) for l in open(f)]
            if np.asarray(col[0]["traj"]).shape[1] == N * 4: cols.append(col)
    comp_files = []
    for D in DIRS:
        comp_files += glob.glob(os.path.join(D, f"ckpt_composition_*_{mslug}.jsonl"))
    if not cols or not comp_files:
        sys.exit(f"Thiếu data: collect={len(cols)} composition={len(comp_files)} cho model {mslug}. Chạy collect_composition trước.")
    roll, Kans, T = fit_koopman(cols); half = (T - K_DEF) // 2
    seen = {}
    for f in comp_files:
        for l in open(f):
            r = json.loads(l); seen.setdefault(r["ti"], r)      # dedupe theo ti (phòng append trùng)
    recs = list(seen.values())
    print(f"===== COMPOSITION ANALYSIS ({mslug}, n_q={len(recs)} unique questions, fit trên {len(cols)} collect-set) =====")

    fb = lambda traj: mean_belief(np.asarray(traj)[-1], N, Kans)
    zk = lambda traj: np.asarray(traj)[K_DEF]
    # per-record errors (ĐƠN VỊ bootstrap = QUESTION/composition-pair, KHÔNG phải sequence)
    E, order_act, order_koop, superr = [], [], [], []
    for r in recs:
        c1, c2 = r["c1"], r["c2"]; C = r["conds"]
        b0 = fb(C["nodef"]); b1 = fb(C["s1"]); b2 = fb(C["s2"])
        e = {"koopman": [], "additive": [], "last": []}
        for kind, chans, last in [("seq12", [c1] * half + [c2] * (T - K_DEF - half), b2),
                                  ("seq21", [c2] * half + [c1] * (T - K_DEF - half), b1)]:
            act = fb(C[kind])
            e["koopman"].append(np.linalg.norm(roll(zk(C[kind]), chans) - act))
            e["additive"].append(np.linalg.norm((b1 + b2 - b0) - act))       # direct-single superposition
            e["last"].append(np.linalg.norm(last - act))
        E.append(e)
        a12, a21 = fb(C["seq12"]), fb(C["seq21"])
        order_act.append(np.linalg.norm(a12 - a21))
        k12 = roll(zk(C["seq12"]), [c1] * half + [c2] * (T - K_DEF - half))
        k21 = roll(zk(C["seq21"]), [c2] * half + [c1] * (T - K_DEF - half))
        order_koop.append(np.linalg.norm(k12 - k21))
        superr.append(np.linalg.norm((a12 - b0) - ((b1 - b0) + (b2 - b0))))

    nq = len(E); meanm = lambda m: float(np.mean([x for e in E for x in e[m]]))
    print(f"\n[#1 UNSEEN COMPOSITION] pred-error bel cuối (thấp=tốt) — n_q={nq}, {2*nq} seq, bootstrap theo QUESTION:")
    for k in ["koopman", "additive", "last"]:
        print(f"    {k:9s} = {meanm(k):.3f}")
    best = min(["additive", "last"], key=meanm)
    margin = max(0.03, 0.05 * meanm(best))                            # TOST: ±0.03 hoặc 5% best-trivial
    rng = np.random.default_rng(5)
    def dmean(sub, base):
        bk = [x for e in sub for x in e["koopman"]]; bb = [x for e in sub for x in e[base]]
        return float(np.mean(bb) - np.mean(bk))
    def clboot(base):                                                # cluster bootstrap theo record
        return np.array([dmean([E[i] for i in rng.integers(0, nq, nq)], base) for _ in range(5000)])
    for b in ["additive", "last"]:
        d0 = dmean(E, b); bs = clboot(b); l95, h95 = np.percentile(bs, [2.5, 97.5]); l90, h90 = np.percentile(bs, [5, 95])
        print(f"    Δ({b}−koopman)={d0:+.3f}  95%CI[{l95:+.3f},{h95:+.3f}]  90%CI[{l90:+.3f},{h90:+.3f}]" + ("   ← best-trivial" if b == best else ""))
    d0 = dmean(E, best); bs = clboot(best); l95, h95 = np.percentile(bs, [2.5, 97.5]); l90, h90 = np.percentile(bs, [5, 95])
    comp_win = l95 > 0
    if comp_win: verdict = f"✅ KOOPMAN THẮNG best-trivial '{best}'"
    elif h95 < 0: verdict = f"❌ Koopman THUA '{best}' — recency/static đủ"
    elif l90 > -margin and h90 < margin: verdict = f"↔ TOST-EQUIVALENT '{best}' (90%CI ⊂ ±{margin:.3f}) ⇒ Koopman KHÔNG lợi thế"
    else: verdict = f"⚠️ chưa kết luận (CI∋0 & chưa đạt TOST ±{margin:.3f})"
    print(f"    TOST margin=±{margin:.3f}  ⇒ {verdict}")

    oa = float(np.mean(order_act)); ok = float(np.mean(order_koop))
    scale = np.mean([np.linalg.norm(fb(r['conds']['s1']) - fb(r['conds']['s2'])) for r in recs])
    rho = np.corrcoef(order_act, order_koop)[0, 1] if np.std(order_koop) > 1e-9 else 0.0
    print(f"\n[#2 ORDER/COMMUTATOR] order-effect thật ‖bel(seq12)−bel(seq21)‖ = {oa:.3f}  (scale ‖s1−s2‖={scale:.3f}, ratio={oa/scale:.2f})")
    koop_catches = rho > 0.3
    ordmsg = ("order-SENSITIVE & Koopman BẮT ĐƯỢC (Koopman-native win)" if oa / scale > 0.5 and koop_catches else
              (f"order LỚN (ratio {oa/scale:.2f}) nhưng do RECENCY & Koopman KHÔNG bắt (corr {rho:+.2f}) ⇒ static-recency đủ" if oa / scale > 0.5 else
               "order nhỏ ⇒ static đủ"))
    print(f"    Koopman dự đoán order-effect: mean={ok:.3f}, corr(thật,koopman)={rho:+.2f}  ⇒ {ordmsg}")
    sm = float(np.mean(superr))
    print(f"\n[ADDITIVITY] superposition error ‖Δcomp − (Δu1+Δu2)‖ = {sm:.3f}  (so scale {scale:.3f}) ⇒ {'PHI-cộng-tính (non-additive)' if sm/scale > 0.5 else 'gần cộng-tính'}")

    # phase classification
    noncomm = oa / scale > 0.5 and koop_catches
    phase = ("NONCOMMUTATIVE nonlinear response (Koopman-native)" if noncomm else
             ("COMPOSITIONAL linear dynamics (Koopman > best-trivial)" if comp_win else
              "STATIC low-rank response (collapse — recency/direct đủ, Koopman KHÔNG thắng)"))
    print(f"\n[PHASE-MAP] ⇒ **{phase}**")
    print("[đọc] Koopman thắng #1 hoặc order-sensitive #2 ⇒ bài lên hạng, T-CYB hợp. Ngược lại ⇒ kết luận chắc:")
    print("      MAD intervention dynamics collapse thành static low-rank response NGAY CẢ dưới composition.")

if __name__ == "__main__":
    main()
