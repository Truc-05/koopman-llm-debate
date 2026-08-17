"""Real pipeline: run LLM debates (Ollama local / Groq / HF — configs/debate.yaml),
save trajectories + transcripts, fit EDMD, print spectrum + truth alignment.

Smoke test (3 câu mẫu):
    python experiments/run_debate_groq.py
Benchmark MMLU (tải trước bằng prepare_mmlu.py):
    python experiments/prepare_mmlu.py --n 40
    python experiments/run_debate_groq.py --topics experiments/topics/mmlu_40.json

Mỗi topic chạy `--repeats` lần với thứ tự đáp án XÁO NGẪU NHIÊN mỗi lần
(chống position-bias + tạo anchor đa dạng). Transcript lưu ra JSON để soi
định tính (debate đồng thuận sai là herding thật hay model gán nhầm index).
"""
import argparse
import os
import sys
import json

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.debate.orchestrator import Agent, DebateOrchestrator
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import (spectral_decomposition, spectral_gap,
                                  eigenfunctions, is_unique_consensus)
from src.criteria.truth_alignment import truth_alignment_index
from src.debate.observables_truth import h_star_soft, along_trajectory

OUT = os.path.join(ROOT, "experiments", "out")

# Smoke-test topics (không phải benchmark). Benchmark thật: --topics <file.json>
SMOKE_TOPICS = [
    {"question": "Which planet has the strongest surface gravity?",
     "answers": ["Jupiter", "Neptune", "Saturn", "Earth"], "a_star": 0},
    {"question": "Which sorting algorithm has the best worst-case time complexity?",
     "answers": ["Quick sort", "Merge sort", "Bubble sort", "Insertion sort"], "a_star": 1},
    {"question": "What is the primary cause of ocean tides?",
     "answers": ["Wind", "Ocean currents", "The Moon's gravity", "Earth's rotation alone"], "a_star": 2},
]


def load_topics(path, limit=None):
    with open(path) as f:
        topics = json.load(f)
    n_answers = len(topics[0]["answers"])
    topics = [t for t in topics if len(t["answers"]) == n_answers]
    return topics[:limit] if limit else topics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default=None,
                    help="JSON file [{question, answers, a_star}]; mặc định: 3 câu smoke test")
    ap.add_argument("--n", type=int, default=None, help="giới hạn số topic lấy từ file")
    ap.add_argument("--repeats", type=int, default=None,
                    help="số lần lặp mỗi topic (mặc định: 1 với --topics, config với smoke)")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "configs", "debate.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "dictionary.yaml")) as f:
        dcfg = yaml.safe_load(f)

    key_env = cfg.get("api_key_env", "GROQ_API_KEY")
    api_key = os.environ.get(key_env, "")
    api_key_cmd = cfg.get("api_key_cmd")     # vd Vertex: "gcloud auth print-access-token"
    base_urls = cfg.get("base_urls")         # danh sách endpoint đa region (round-robin)
    _base_check = cfg.get("base_url") or (base_urls or [""])[0]
    if not api_key and not api_key_cmd and "localhost" not in _base_check:
        sys.exit(f"{key_env} chưa được set trong môi trường — export rồi chạy lại.")

    if args.topics:
        topics = load_topics(args.topics, args.n)
        n_repeats = args.repeats or 1
        tag = os.path.splitext(os.path.basename(args.topics))[0]
        if n_repeats > 1:
            tag += f"_r{n_repeats}"   # không đè output của run 1-repeat trước đó
    else:
        topics = SMOKE_TOPICS
        n_repeats = args.repeats or cfg.get("n_repeats", 1)
        tag = "smoke"

    os.makedirs(OUT, exist_ok=True)
    n_agents = cfg["n_agents"]
    n_answers = len(topics[0]["answers"])
    shuffle = cfg.get("shuffle_answers", True)
    rng = np.random.default_rng(cfg.get("seed", 0))
    total = len(topics) * n_repeats
    print(f"== {tag}: {len(topics)} topics x {n_repeats} repeats = {total} debates, "
          f"{n_agents} agents x {cfg['n_rounds']} rounds, model={cfg['model']}")

    # Checkpoint từng debate — server sập giữa chừng thì chạy lại đúng lệnh cũ
    # là tiếp tục từ chỗ dừng (xóa file ckpt nếu muốn chạy lại từ đầu).
    ckpt_path = os.path.join(OUT, f"ckpt_{tag}.jsonl")
    runs = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                r = json.loads(line)
                r["trajectory"] = np.asarray(r["trajectory"])
                runs.append(r)
        print(f"RESUME: đã có {len(runs)}/{total} debates trong {ckpt_path}")
    n_done = len(runs)

    idx = 0
    for rep in range(n_repeats):
        for ti, topic in enumerate(topics):
            # perm PHẢI rút mỗi vòng lặp kể cả khi skip — giữ nguyên chuỗi rng
            # để debate thứ k của lần resume trùng với lần chạy gốc.
            perm = rng.permutation(n_answers) if shuffle else np.arange(n_answers)
            idx += 1
            if idx <= n_done:
                continue
            answers = [topic["answers"][j] for j in perm]
            a_star = int(np.where(perm == topic["a_star"])[0][0])
            print(f"--- [{len(runs) + 1}/{total}] {topic['question'][:70]}")

            agents = [Agent(i, model=cfg["model"], temperature=cfg["temperature"],
                            max_tokens=cfg["max_tokens"],
                            base_url=cfg.get("base_url", "none"), base_urls=base_urls,
                            api_key=api_key or "none",
                            retries=cfg.get("retries", 5),
                            transcript_window=cfg.get("transcript_window", 4),
                            timeout=cfg.get("timeout", 120),
                            api_key_cmd=api_key_cmd)
                      for i in range(n_agents)]
            orch = DebateOrchestrator(agents, topic["question"], answers,
                                      cfg["n_rounds"],
                                      request_interval=cfg.get("request_interval", 3.0))
            Z = orch.run()                      # (n_rounds+1, N*K), includes z_0
            p_final = np.exp(Z[-1]).reshape(n_agents, n_answers).mean(0)
            correct = bool(np.argmax(p_final) == a_star)
            print(f"    final belief = {np.round(p_final, 3)} -> "
                  f"{'ĐÚNG' if correct else 'SAI'} (a*={a_star})")
            n_empty = sum(1 for t in orch.transcript if not t["text"].strip())
            if n_empty * 4 > len(orch.transcript):
                sys.exit(f"DỪNG: {n_empty}/{len(orch.transcript)} câu trả lời rỗng "
                         f"trong debate này (server sập?) — KHÔNG lưu debate hỏng. "
                         f"Khắc phục server rồi chạy lại đúng lệnh cũ, sẽ tự resume "
                         f"từ debate {len(runs) + 1}/{total}.")
            rec = {"question": topic["question"], "answers": answers,
                   "a_star": a_star, "correct": correct,
                   "subject": topic.get("subject", ""),
                   "transcript": orch.transcript}
            with open(ckpt_path, "a") as f:
                f.write(json.dumps({**rec, "trajectory": Z.tolist()},
                                   ensure_ascii=False) + "\n")
            runs.append({**rec, "trajectory": Z})

    # ---- save trajectories + transcripts ----
    np.savez(os.path.join(OUT, f"trajs_{tag}.npz"),
             **{f"traj_{i}": r["trajectory"] for i, r in enumerate(runs)})
    with open(os.path.join(OUT, f"transcripts_{tag}.json"), "w") as f:
        json.dump([{k: r[k] for k in ("question", "answers", "a_star",
                                      "correct", "subject", "transcript")}
                   for r in runs], f, ensure_ascii=False, indent=1)
    acc = float(np.mean([r["correct"] for r in runs]))
    print(f"\nsaved -> {OUT}/trajs_{tag}.npz, transcripts_{tag}.json")
    print(f"debate accuracy = {acc:.3f} ({sum(r['correct'] for r in runs)}/{len(runs)})")

    # ---- pooled EDMD ----
    trajs = [r["trajectory"] for r in runs]
    Z_t = np.concatenate([Z[:-1] for Z in trajs], axis=0)
    Z_tp1 = np.concatenate([Z[1:] for Z in trajs], axis=0)
    d = PolynomialDictionary(degree=dcfg["degree"])
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    M, S = Psi_X.shape
    if S < 2 * M:
        print(f"CẢNH BÁO: {S} transitions < 2x{M} features — phổ chỉ minh họa; "
              f"tăng số debates hoặc giảm degree.")
    K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=dcfg["edmd"]["reg"])
    eigvals, V, W = spectral_decomposition(K)
    n_at_one = int(np.sum(np.abs(eigvals - 1.0) < 1e-2))
    print(f"|lambda| top-5   = {np.round(np.abs(eigvals[:5]), 4)}")
    print(f"spectral gap     = {spectral_gap(eigvals):.4f}")
    print(f"trị riêng tại 1  = {n_at_one} "
          f"(Prop 1: >1 => nhiều lớp đồng thuận, không unique) | "
          f"unique_consensus = {is_unique_consensus(eigvals, tol=1e-2)}")

    # ---- tau per debate against pooled phi_1; summary theo đúng/sai ----
    Phi = eigenfunctions(d, W, Z_t)
    offset = 0
    taus = []
    for r in runs:
        n_rows = r["trajectory"].shape[0] - 1
        h = along_trajectory(h_star_soft, r["trajectory"][:-1], n_agents=n_agents,
                             n_answers=n_answers, a_star=r["a_star"])
        taus.append(float(truth_alignment_index(h, Phi[offset:offset + n_rows, 0])))
        offset += n_rows
    taus = np.array(taus)
    ok = np.array([r["correct"] for r in runs])
    if len(runs) <= 15:
        for i, r in enumerate(runs):
            print(f"tau (run {i}: {'ĐÚNG' if r['correct'] else 'SAI '}) = {taus[i]:.3f}")
    if ok.any():
        print(f"tau | debates ĐÚNG: {taus[ok].mean():.3f} ± {taus[ok].std():.3f}  (n={ok.sum()})")
    if (~ok).any():
        print(f"tau | debates SAI : {taus[~ok].mean():.3f} ± {taus[~ok].std():.3f}  (n={(~ok).sum()})")

    with open(os.path.join(OUT, f"results_{tag}.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "n_debates": len(runs),
            "eigvals_top10": [[float(np.real(v)), float(np.imag(v))]
                              for v in eigvals[:10]],
            "spectral_gap": float(spectral_gap(eigvals)),
            "n_eigs_at_one": n_at_one,
            "runs": [{"question": r["question"], "subject": r["subject"],
                      "correct": r["correct"], "tau": taus[i]}
                     for i, r in enumerate(runs)],
        }, f, ensure_ascii=False, indent=1)
    print(f"summary -> {OUT}/results_{tag}.json")


if __name__ == "__main__":
    main()
