"""Phân tích campaign can thiệp (xem beat4_intervention_spec.md §4).

Đọc interv_<tag>.json → mỗi topic-trigger có 4 arm, mỗi arm R continuation.
Đơn vị = topic; outcome = acc_arm,i (tỉ lệ đúng trên R continuation).

- Wilcoxon signed-rank ghép cặp cho H1 (koopman−none), H2 (koopman−generic),
  phụ (generic−sham, sham−none). Hiệu chỉnh Holm cho {H1,H2}.
- McNemar (majority-vote/topic) đối chứng.
- Bootstrap CI 95% cho Δacc (ghép cặp theo topic).

Dùng: python experiments/analyze_intervention.py interv_mmlu_160
"""
import sys, os, json
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "interv_mmlu_160"
data = json.load(open(os.path.join(OUT, f"{tag}.json")))
ARMS = data["arms"]

trig = [r for r in data["records"] if r["triggered"] and r["arms"]]
if not trig:
    sys.exit("Chưa có topic trigger nào có đủ arm.")

# acc[arm] = mảng theo topic; maj[arm] = đúng-đa-số theo topic (nhị phân)
acc = {a: np.array([np.mean([o["correct"] for o in r["arms"][a]["outcomes"]]) for r in trig])
       for a in ARMS}
maj = {a: np.array([np.mean([o["correct"] for o in r["arms"][a]["outcomes"]]) > 0.5 for r in trig])
       for a in ARMS}
n = len(trig)

print(f"tag={tag}  n_topic_trigger={n}  R={data['R']}  θ={data['theta']:.3f}")
print("\nacc trung bình theo arm (mean per-topic acc):")
for a in ARMS:
    print(f"  {a:<10} {acc[a].mean():.3f}  (maj-vote acc {maj[a].mean():.3f})")


def boot_ci(delta, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(B, len(delta)))
    means = delta[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])


def compare(a1, a2, label):
    delta = acc[a1] - acc[a2]
    lo, hi = boot_ci(delta)
    # Wilcoxon signed-rank (bỏ cặp bằng 0); nếu toàn 0 thì p=1
    nz = delta[delta != 0]
    if len(nz) == 0:
        p_w = 1.0
    else:
        p_w = stats.wilcoxon(nz, alternative="two-sided").pvalue
    # McNemar trên majority-vote: b=a1 đúng & a2 sai, c=a1 sai & a2 đúng
    b = int(np.sum(maj[a1] & ~maj[a2])); c = int(np.sum(~maj[a1] & maj[a2]))
    p_mc = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return {"label": label, "a1": a1, "a2": a2, "d": delta.mean(),
            "ci": (lo, hi), "p_w": p_w, "b": b, "c": c, "p_mc": p_mc}


tests = [compare("A_koopman", "A_none", "H1 koopman>none (hiệu ứng tổng)"),
         compare("A_koopman", "A_generic", "H2 koopman>generic (điều hướng phổ)"),
         compare("A_generic", "A_sham", "phụ: generic>sham (chỉ thị re-derive)"),
         compare("A_sham", "A_none", "phụ: sham>none (hiệu ứng ngắt nhịp)")]

# Holm cho 2 giả thuyết chính H1,H2
prim = tests[:2]
order = sorted(range(2), key=lambda i: prim[i]["p_w"])
holm = {}
for rank, i in enumerate(order):
    holm[i] = min(1.0, prim[i]["p_w"] * (2 - rank))

print("\n{:<40}{:>9}{:>20}{:>10}{:>12}".format("giả thuyết", "Δacc", "CI95", "p(Wilc)", "p(McN b/c)"))
for i, t in enumerate(tests):
    hp = f"  Holm={holm[i]:.3f}" if i in holm else ""
    print("{:<40}{:>+9.3f}   [{:+.3f},{:+.3f}]  {:>8.4f}   b={} c={} p={:.3f}{}".format(
        t["label"], t["d"], t["ci"][0], t["ci"][1], t["p_w"], t["b"], t["c"], t["p_mc"], hp))

print("\nĐọc: H1 dương & Holm<0.05 & Δ≥0.10 => tuyên bố nhịp 4. "
      "H2 dương => điều hướng Koopman có giá trị riêng (luận điểm cốt lõi).")

# ước σ_δ cho power Stage 2
sd = (acc["A_koopman"] - acc["A_none"]).std(ddof=1)
print(f"\nσ_δ(koopman−none) = {sd:.3f}  -> n cho Δ=0.10: "
      f"{int(np.ceil(7.85*sd**2/0.10**2))}, Δ=0.15: {int(np.ceil(7.85*sd**2/0.15**2))} topic")
