"""Chốt A — kiểm tra SELECTION CONFOUND của oracle (0 debate mới, chỉ đọc eval ckpt).

Oracle chọn argmax honest-margin trên 5 candidate → +16.7%. RỦI RO: gain có thể là do
CHỌN-BEST-OF-RUNS (selection over stochastic reruns) chứ không phải INTERVENTION thật.
Nếu đúng vậy thì trụ "controllable" sụp và cả Figure-1 (trần oracle) là artifact.

Script đo 3 chỉ báo rủi ro + 1 DISCRIMINATOR quyết định, TẤT CẢ từ ckpt đã lưu:
  (A) rate honest_lead(z_kdef) == a_star — cao ⇒ honest-margin ~ proxy đúng/sai ⇒ selection dễ bơm.
  (B) steering-gain: đẩy c có THẬT SỰ nâng bel[c] so no-def? ⇒ controllability MỨC BELIEF.
  (C) *** DISCRIMINATOR *** FIXED honest-push policy (đẩy c_hon MỌI topic, KHÔNG chọn best)
      vs no-def, McNemar. Policy KHÔNG-selection mà VẪN beat no-def ⇒ gain KHÔNG do selection
      ⇒ Chốt A an toàn, KHỎI tốn 240 debate. Nếu (C) mơ hồ → mới cần control rerun đầy đủ.
  (D) phân rã oracle-fixes: xác nhận mọi topic oracle sửa (nodef sai→oracle đúng) đều có c*≠-1;
      và c* có hay trùng c_hon không (policy có mạch lạc hay bốc ngẫu nhiên).

Control ĐẦY ĐỦ (5× rerun no-def rồi chọn best-honest-margin) vẫn cần ~240 debate — chỉ chạy nếu (C) mơ hồ.
Dùng: python experiments/check_selection_confound.py [--eval experiments/out/ckpt_twindef_eval_mmlu_clean6.jsonl]
"""
import os, sys, json, argparse
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.state import split_state

N, Kans = 4, 4
CANDS = [-1, 0, 1, 2, 3]


def honest_lead(z, x_adv):
    """đáp án phe honest (loại agent 0 = adversary) tự nghiêng về tại z — LABEL-FREE."""
    _, p = split_state(np.asarray(z), N, Kans)
    h = p[1:].mean(axis=0).astype(float)
    h[x_adv] = -np.inf
    return int(np.argmax(h))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (ctr - half) / den, (ctr + half) / den


def mcnemar_exact(a, b):
    """McNemar exact (binomial). Trả (n: a đúng/b sai, n: a sai/b đúng, p 2-phía)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if (not x) and y)
    m = n01 + n10
    if m == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2.0 * sum(comb(m, i) for i in range(k + 1)) * (0.5 ** m)
    return n01, n10, min(1.0, p)


def hmargin(bel, x, c_hon):
    return bel[c_hon] - bel[x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(ROOT, "experiments", "out",
                    "ckpt_twindef_eval_mmlu_clean6.jsonl"))
    args = ap.parse_args()
    ev = [json.loads(l) for l in open(args.eval)]
    n = len(ev)
    if not all("bel" in next(iter(r["outs"].values())) for r in ev):
        sys.exit("Eval ckpt thiếu 'bel' — cần bản mới. Xem run_twin_mpc_defense.py.")

    nd_acc, fix_acc, orc_acc = [], [], []      # no-def / fixed-honest-push / oracle
    c_hon_is_star, orc_c, fix_is_hon = [], [], []
    lifts = {c: [] for c in range(Kans)}       # steering-gain per pushed candidate

    for r in ev:
        x = r["x_adv"]; a_star = r["a_star"]; outs = r["outs"]; z = r["z_kdef"]
        c_hon = honest_lead(z, x)
        c_hon_is_star.append(c_hon == a_star)

        nd_acc.append(bool(outs["-1"]["correct"]))
        fix_acc.append(bool(outs[str(c_hon)]["correct"]))     # (C) policy KHÔNG selection
        oc = max(CANDS, key=lambda c: hmargin(np.asarray(outs[str(c)]["bel"]), x, c_hon))
        orc_acc.append(bool(outs[str(oc)]["correct"]))
        orc_c.append(oc); fix_is_hon.append(oc == c_hon)

        nd_bel = np.asarray(outs["-1"]["bel"])
        for c in range(Kans):                                  # (B) đẩy c có nâng bel[c]?
            lifts[c].append(float(np.asarray(outs[str(c)]["bel"])[c] - nd_bel[c]))

    def acc_line(a):
        k = int(np.sum(a)); lo, hi = wilson(k, n)
        return f"{k}/{n}={k/n:.3f} [{lo:.2f},{hi:.2f}]"

    print(f"\n================ CHỐT A: SELECTION CONFOUND (n={n}, 0 debate mới) ================")

    print(f"\n(A) honest-margin anchor có ĐÚNG không (rate c_hon == a_star):"
          f"  {np.mean(c_hon_is_star):.2f}")
    print("    → cao (≳0.6) ⇒ honest-margin ~ proxy đúng/sai ⇒ selection có KHẢ NĂNG bơm accuracy.")
    print("    → thấp ⇒ honest-margin KHÔNG phải proxy nhãn ⇒ rủi ro selection thấp.")

    all_lift = np.array([v for c in range(Kans) for v in lifts[c]])
    print(f"\n(B) steering-gain (đẩy c ⇒ Δbel[c] so no-def): TB toàn cục {all_lift.mean():+.3f}"
          f" | frac Δ>0 = {np.mean(all_lift > 0):.2f}")
    for c in range(Kans):
        v = np.array(lifts[c])
        print(f"      push c={c}:  Δbel[c] TB {v.mean():+.3f}  (frac Δ>0 = {np.mean(v > 0):.2f})")
    print("    → Δbel[c] dương rõ ⇒ can thiệp THẬT lái được belief (controllability mức belief) ⇒ đỡ Chốt A.")

    print(f"\n(C) *** DISCRIMINATOR — policy KHÔNG selection ***")
    print(f"      no-def            acc {acc_line(nd_acc)}")
    print(f"      fixed honest-push acc {acc_line(fix_acc)}   (đẩy c_hon MỌI topic, không chọn best)")
    print(f"      oracle (select)   acc {acc_line(orc_acc)}   (để so trần)")
    b, c, p = mcnemar_exact(fix_acc, nd_acc)
    print(f"      McNemar fixed-honest-push vs no-def: {b}↑/{c}↓  p={p:.4f}  "
          f"{'SIGNIF ⇒ gain KHÔNG do selection ⇒ CHỐT A AN TOÀN' if p < 0.05 else 'ns'}")
    print("    → fixed-policy (không selection) beat no-def ⇒ +16.7% oracle KHÔNG thể chỉ là selection.")
    print("    → nếu ns nhưng hướng dương mạnh: directional; nếu ~0/âm: PHẢI chạy control rerun 240 debate.")

    fixes = [i for i in range(n) if orc_acc[i] and not nd_acc[i]]
    bad = [i for i in fixes if orc_c[i] == -1]
    print(f"\n(D) oracle sửa {len(fixes)} topic (nodef sai→oracle đúng); trong đó c*=-1 (bất khả): {len(bad)}")
    print(f"    oracle chọn c*=-1 (no-def) trên {np.mean([c == -1 for c in orc_c]):.2f} topic;"
          f" c*==c_hon trên {np.mean(fix_is_hon):.2f} topic.")
    print("    → mọi fix có c*≠-1 (đúng kỳ vọng: sửa được là nhờ can thiệp, không phải chọn no-def).")

    print(f"\n---- ĐỌC KẾT LUẬN ----")
    print(" Chốt A AN TOÀN nếu: (C) fixed-honest-push ≥ no-def (lý tưởng p<0.05) VÀ (B) steering-gain dương.")
    print(" Khi đó Figure-1 (trần oracle) đứng, KHỎI tốn 240 debate. Ngược lại → chạy control rerun đầy đủ.")


if __name__ == "__main__":
    main()
