"""Nhịp 4 — chạy campaign can thiệp fork ghép cặp (xem beat4_intervention_spec.md).

Với mỗi topic MỚI: chạy tới vòng k (=3) MỘT lần → tính gated-τ@k bằng operator ĐÓNG BĂNG
từ r5 → nếu trigger (gated < θ) thì FORK từ đúng state đó thành 4 arm, mỗi arm R continuation:
  A_koopman : moderator hướng agent xét lại option = "challenger" operator dự báo tăng mạnh nhất
  A_generic : cùng template nhưng option NGẪU NHIÊN (không dùng phổ)  -> tách giá trị điều hướng
  A_sham    : thông điệp trung tính (chỉ "reason carefully")           -> tách hiệu ứng ngắt nhịp
  A_none    : không moderator (system_prompt rỗng = khớp calibration)   -> baseline
Ghép cặp: mọi arm chia sẻ y hệt vòng 0..k-1 -> khử phương sai độ-khó-câu-hỏi.

Detector (φ₁, W, K, median_disp_k, θ) fit MỘT lần trên trajs_mmlu_40_r5 rồi ĐÓNG BĂNG.
Trigger áp lên topic mới = out-of-sample tự nhiên.

Checkpoint từng topic -> resume bằng cách chạy lại đúng lệnh. Fail-fast nếu content rỗng.

Dùng:
  python experiments/run_intervention.py --topics experiments/topics/mmlu_160.json \
      --calib mmlu_40_r5 --R 4 --k 3 --precision 0.70 [--max-trigger 15]
"""
import os, sys, json, copy, argparse
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.orchestrator import Agent
from src.debate.state import build_state, split_state, mean_belief
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import spectral_decomposition, eigenfunctions
from src.criteria.truth_alignment import truth_alignment_index
from src.debate.observables_truth import h_star_soft, along_trajectory

OUT = os.path.join(ROOT, "experiments", "out")
ARMS = ["A_koopman", "A_generic", "A_sham", "A_none"]


# ----------------------------- detector đóng băng -----------------------------
def build_detector(calib_tag, k, degree, reg, precision_target):
    """Fit pooled Koopman trên calibration trajs, trả detector đóng băng + θ."""
    npz = np.load(os.path.join(OUT, f"trajs_{calib_tag}.npz"))
    trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
    res = json.load(open(os.path.join(OUT, f"results_{calib_tag}.json")))
    y = np.array([bool(r["correct"]) for r in res["runs"]])

    d = PolynomialDictionary(degree=degree)
    Z_t = np.concatenate([t[:-1] for t in trajs], 0)
    Z_tp1 = np.concatenate([t[1:] for t in trajs], 0)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=reg)
    _, _, W = spectral_decomposition(K)          # φ₁ = eigenfunctions(d, W, ·)[:,0]
    d.transform(Z_t[:1])                          # đảm bảo state_slice được set

    median_disp = float(np.median([np.linalg.norm(t[k] - t[0]) for t in trajs]))
    return d, W, K, median_disp, y, trajs


def gated_tau_at_k(d, W, traj_prefix, k, N, Kans, a_star, median_disp):
    """traj_prefix: (k+1, NK) gồm z_0..z_k. Trả (gated, tau, disp)."""
    Zi = traj_prefix[:k]                                   # k trạng thái nguồn
    h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=Kans, a_star=a_star)
    phi1 = eigenfunctions(d, W, Zi)[:, 0]
    tau = float(truth_alignment_index(h, phi1))
    disp = float(np.linalg.norm(traj_prefix[k] - traj_prefix[0]))
    gated = tau * min(1.0, disp / median_disp)
    return gated, tau, disp


def choose_theta(gated_calib, y, precision_target):
    """θ lớn nhất sao cho P(đúng | gated < θ) ≤ 1 - precision_target."""
    max_correct = 1.0 - precision_target
    best = None
    for th in np.unique(gated_calib):
        below = gated_calib < th
        if below.sum() >= 5 and y[below].mean() <= max_correct:
            best = float(th)
    if best is None:                    # fallback: phân vị 40% (thận trọng)
        best = float(np.quantile(gated_calib, 0.40))
    return best


# ------------------------- Koopman challenger (label-free) -------------------------
def koopman_challenger(K, d, z_k, N, Kans, horizon):
    """Option operator dự báo TĂNG mạnh nhất (loại option đang dẫn). Không dùng nhãn."""
    psi = d.transform(z_k.reshape(1, -1))[0]
    for _ in range(horizon):
        psi = K @ psi
    z_pred = np.real(psi[d.state_slice])
    pred_belief = mean_belief(z_pred, N, Kans)
    cur_belief = mean_belief(z_k, N, Kans)
    delta = pred_belief - cur_belief
    leader = int(np.argmax(cur_belief))
    delta[leader] = -np.inf
    return int(np.argmax(delta))


def moderator_prompt(arm, target_idx, answers):
    if arm == "A_none":
        return ""
    if arm == "A_sham":
        return ("You are the debate moderator. Continue the discussion and reason "
                "carefully and rigorously about the question.")
    # koopman & generic: cùng template, khác cách chọn target_idx
    return ("You are the debate moderator. The discussion may be converging "
            "prematurely or stalling. Give option "
            f"{target_idx} (\"{answers[target_idx]}\") fair, independent "
            "consideration, then RE-DERIVE your answer from scratch on the merits. "
            "Do not simply agree with the group's current leaning.")


# ----------------------------- debate primitives -----------------------------
def make_agents(cfg, api_key, system_prompt=""):
    return [Agent(i, model=cfg["model"], temperature=cfg["temperature"],
                  max_tokens=cfg["max_tokens"], base_url=cfg["base_url"],
                  api_key=api_key or "none", retries=cfg.get("retries", 5),
                  transcript_window=cfg.get("transcript_window", 4),
                  timeout=cfg.get("timeout", 120), system_prompt=system_prompt)
            for i in range(cfg["n_agents"])]


def play_rounds(agents, question, answers, transcript, current_logits, rounds,
                request_interval):
    """Chạy các vòng trong `rounds`, append vào transcript (mutate). Trả (logits, states)."""
    states = []
    import time
    for round_idx in rounds:
        round_logits = list(current_logits)
        for k_i, idx in enumerate(range(len(agents))):
            if request_interval > 0:
                time.sleep(request_interval)
            logits, text = agents[idx].act(question, answers, transcript)
            round_logits[idx] = logits
            transcript.append({"agent_id": idx, "round": round_idx, "text": text})
        current_logits = round_logits
        states.append(build_state(current_logits))
    return current_logits, states


def n_empty(transcript):
    return sum(1 for t in transcript if not t["text"].strip())


# --------------------------------- main ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--calib", default="mmlu_40_r5", help="tag calibration để fit detector")
    ap.add_argument("--R", type=int, default=4, help="continuation mỗi arm")
    ap.add_argument("--k", type=int, default=3, help="vòng trigger")
    ap.add_argument("--precision", type=float, default=0.70, help="precision mục tiêu của trigger")
    ap.add_argument("--horizon", type=int, default=3, help="bước roll-forward chọn challenger")
    ap.add_argument("--max-trigger", type=int, default=0, help=">0: dừng sau N topic trigger (pilot)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "configs", "debate.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "dictionary.yaml")) as f:
        dcfg = yaml.safe_load(f)
    key_env = cfg.get("api_key_env", "GROQ_API_KEY")
    api_key = os.environ.get(key_env, "")
    if not api_key and "localhost" not in cfg["base_url"]:
        sys.exit(f"{key_env} chưa set — export rồi chạy lại.")

    N, T = cfg["n_agents"], cfg["n_rounds"]
    ri = cfg.get("request_interval", 3.0)
    topics = json.load(open(args.topics))
    Kans = len(topics[0]["answers"])
    k = args.k

    # ---- detector đóng băng ----
    d, W, Kop, median_disp, y_cal, trajs_cal = build_detector(
        args.calib, k, dcfg["degree"], dcfg["edmd"]["reg"], args.precision)
    # a_star không có trong results_*.json -> lấy từ transcripts_*.json
    tr_cal = json.load(open(os.path.join(OUT, f"transcripts_{args.calib}.json")))
    gated_cal = np.array([
        gated_tau_at_k(d, W, trajs_cal[i][:k + 1], k, N, Kans,
                       int(tr_cal[i]["a_star"]), median_disp)[0]
        for i in range(len(trajs_cal))])
    theta = choose_theta(gated_cal, y_cal, args.precision)
    below = gated_cal < theta
    print(f"== detector (đóng băng từ {args.calib}) ==")
    print(f"  median_disp_{k} = {median_disp:.2f}")
    print(f"  θ = {theta:.4f} | trigger-rate calib = {below.mean():.2f} | "
          f"P(đúng|trigger) calib = {y_cal[below].mean():.3f} "
          f"(mục tiêu ≤ {1-args.precision:.2f})")

    tag = os.path.splitext(os.path.basename(args.topics))[0]
    ckpt = os.path.join(OUT, f"ckpt_interv_{tag}.jsonl")
    rng = np.random.default_rng(cfg.get("seed", 0) + args.seed)
    shuffle = cfg.get("shuffle_answers", True)

    records, n_trig = [], 0
    if os.path.exists(ckpt):
        for line in open(ckpt):
            records.append(json.loads(line))
        n_trig = sum(1 for r in records if r["triggered"])
        print(f"RESUME: {len(records)} topic đã xong ({n_trig} trigger).")

    for ti, topic in enumerate(topics):
        perm = rng.permutation(Kans) if shuffle else np.arange(Kans)
        if ti < len(records):
            continue
        if args.max_trigger and n_trig >= args.max_trigger:
            print(f"Đã đủ {n_trig} topic trigger — dừng (pilot).")
            break

        answers = [topic["answers"][j] for j in perm]
        a_star = int(np.where(perm == topic["a_star"])[0][0])
        print(f"\n--- [{ti+1}/{len(topics)}] {topic['question'][:70]}")

        # 1) chạy chung vòng 0..k-1
        agents = make_agents(cfg, api_key)
        transcript, logits0 = [], [np.zeros(Kans) for _ in range(N)]
        traj = [build_state(logits0)]
        cur, states = play_rounds(agents, topic["question"], answers, transcript,
                                  logits0, range(0, k), ri)
        traj += states                                   # z_0..z_k
        if n_empty(transcript) * 4 > len(transcript):
            sys.exit(f"DỪNG: quá nhiều reply rỗng ở vòng đầu topic {ti+1} (server?). "
                     f"Sửa server rồi chạy lại — sẽ resume từ topic {ti+1}.")

        # 2) trigger?
        gated, tau, disp = gated_tau_at_k(d, W, np.stack(traj), k, N, Kans, a_star,
                                          median_disp)
        triggered = bool(gated < theta)
        rec = {"topic_idx": ti, "question": topic["question"],
               "subject": topic.get("subject", ""), "a_star": a_star,
               "perm": perm.tolist(), "gated_tau_k": gated, "tau_k": tau,
               "disp_k": disp, "triggered": triggered,
               "shared_transcript": transcript, "z_k_logits": [c.tolist() for c in cur],
               "arms": {}}
        print(f"    gated-τ@{k}={gated:.3f} (θ={theta:.3f}) -> "
              f"{'TRIGGER' if triggered else 'bỏ qua'}")

        # 3) fork 4 arm nếu trigger
        if triggered:
            n_trig += 1
            z_k = np.stack(traj)[k]
            ch_koop = koopman_challenger(Kop, d, z_k, N, Kans, args.horizon)
            leader = int(np.argmax(mean_belief(z_k, N, Kans)))
            non_leader = [a for a in range(Kans) if a != leader]
            for arm in ARMS:
                if arm == "A_koopman":
                    tgt = ch_koop
                elif arm == "A_generic":
                    tgt = int(rng.choice(non_leader))
                else:
                    tgt = -1
                sysp = moderator_prompt(arm, tgt, answers)
                outs = []
                for r in range(args.R):
                    ag = make_agents(cfg, api_key, system_prompt=sysp)
                    tsc = copy.deepcopy(transcript)
                    curc = [c.copy() for c in cur]
                    _, st = play_rounds(ag, topic["question"], answers, tsc, curc,
                                        range(k, T), ri)
                    if n_empty(tsc) * 4 > len(tsc):
                        sys.exit(f"DỪNG: reply rỗng ở arm {arm} topic {ti+1} — sửa server, "
                                 f"chạy lại (KHÔNG lưu topic hỏng, resume từ topic {ti+1}).")
                    pfin = mean_belief(st[-1], N, Kans)
                    outs.append({"correct": bool(np.argmax(pfin) == a_star),
                                 "final_belief": pfin.tolist()})
                rec["arms"][arm] = {"target": tgt, "system_prompt": sysp, "outcomes": outs}
                acc = np.mean([o["correct"] for o in outs])
                print(f"      {arm:<10} target={tgt:>2} acc={acc:.2f} ({sum(o['correct'] for o in outs)}/{args.R})")

        with open(ckpt, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records.append(rec)

    # ---- lưu tổng ----
    out = os.path.join(OUT, f"interv_{tag}.json")
    json.dump({"tag": tag, "calib": args.calib, "k": k, "R": args.R,
               "theta": theta, "median_disp_k": median_disp, "arms": ARMS,
               "n_topics": len(records), "n_triggered": n_trig,
               "records": records}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> {out}  | {n_trig} topic trigger / {len(records)} topic")


if __name__ == "__main__":
    main()
