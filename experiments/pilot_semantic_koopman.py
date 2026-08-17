"""PILOT cửa B (semantic Koopman) — PROTOCOL SẠCH (không leakage).
Belief-logit quasi-static ⇒ persistence quá mạnh ⇒ operator chết. Không cứu bằng degree/reg trên belief.
Thử đổi cái Koopman mô hình hóa + siết protocol:

  (1) OBSERVABLE giàu, per-agent:  z_t = [e_{1,t}..e_{N,t} (L2-norm), belief, confidence, disagreement, novelty]
  (2) TRANSITION/residual fit:     next = z_t + f(z_t,u_t); FIT chỉ trên transition CÓ chuyển động (flip/‖Δz‖>median),
                                   EVAL trên toàn test (tránh cherry-pick). Ridge.
  (3) EXCITATION + regime:         u_t = [one-hot(c)·1{t≥K_DEF}, post-flag, t/T]  (intervention nhắm target khác nhau)
  PROTOCOL: PCA fit TRAIN-ONLY (normalize+whiten), giữ ~var; baseline (persist vs climatology) chọn TRÊN TRAIN rồi CỐ ĐỊNH;
            leave-one-QUESTION-out; bootstrap theo QUESTION. KHÔNG dùng test để chọn baseline/dim/reg.

TIÊU CHUẨN SỐNG: operator VƯỢT baseline-trivial-chọn-trên-train, CI-lower>0 (nhất là CHANGED). Không ⇒ dừng claim
'working Koopman', chốt honest: 'MAD empirically controllable but not Koopman-identifiable under these observables'.
CHECKPOINT/RESUME. Dùng (user tự chạy — có gọi LLM):
  python experiments/pilot_semantic_koopman.py --topics experiments/topics/mmlu_clean6.json --n 24
  # tune (validation, KHÔNG theo test): --var 0.9 --maxd 12 --ridge 1.0 --fit-moving 1
  # embed lỗi: ollama pull nomic-embed-text ; thêm --embed-model nomic-embed-text
"""
import os, sys, json, argparse, re, urllib.request
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.run_twin_mpc_defense import run_debate, empty_bad, K_DEF
from src.debate.state import mean_belief

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))

# ---------------- embeddings (Ollama native trước, fallback) ----------------
def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def embed(text, url, model):
    text = text.strip() or " "
    host = url.split("/v1/")[0].split("/api/")[0]
    try:
        return np.asarray(_post(host + "/api/embeddings", {"model": model, "prompt": text})["embedding"], float)
    except Exception:
        try:
            return np.asarray(_post(host + "/api/embed", {"model": model, "input": text})["embeddings"][0], float)
        except Exception:
            return np.asarray(_post(host + "/v1/embeddings", {"model": model, "input": text})["data"][0]["embedding"], float)

# ---------------- observable ----------------
def _l2(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

def raw_feats(rec):
    """R×D_raw: per-agent L2-emb ++ belief ++ [confidence, disagreement, novelty]."""
    emb = [np.asarray(E, float) for E in rec["emb"]]
    bel = [np.asarray(b, float) for b in rec["belief"]]
    feats, prev = [], None
    for t in range(len(emb)):
        En = _l2(emb[t]); iu = np.triu_indices(len(En), 1)
        dis = float(1 - (En @ En.T)[iu].mean()) if len(iu[0]) else 0.0
        conf = float(np.max(bel[t]))
        nov = 0.0 if prev is None else float(np.mean(1 - np.sum(En * _l2(prev), axis=1)))
        feats.append(np.concatenate([En.reshape(-1), bel[t], [conf, dis, nov]]))
        prev = emb[t]
    return np.asarray(feats, float)

def build_u(c, K, t, R):
    base = np.zeros(K)
    if c is not None and c >= 0 and t >= K_DEF:
        base[c] = 1.0
    return np.concatenate([base, [1.0 if t >= K_DEF else 0.0, t / max(R - 1, 1)]])

# ---------------- Koopman LOO test (protocol sạch) ----------------
def _pca_train(Xtr, var_keep, maxd):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    _, S, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False)
    ev = (S ** 2); cum = np.cumsum(ev) / ev.sum()
    d = int(np.searchsorted(cum, var_keep) + 1)
    d = max(2, min(d, maxd, Vt.shape[0]))
    return mu, sd, Vt[:d].T

def loo_skill(recs, var_keep, maxd, ridge, fit_moving):
    raws = [raw_feats(r) for r in recs]
    K = len(recs[0]["belief"][0])
    def transitions(P, rec):
        bel = [np.asarray(b) for b in rec["belief"]]; rows = []
        for t in range(len(P) - 1):
            flip = int(np.argmax(bel[t])) != int(np.argmax(bel[t + 1]))
            rows.append((P[t], build_u(rec.get("c"), K, t, len(P)), P[t + 1],
                         flip, float(np.linalg.norm(P[t + 1] - P[t]))))
        return rows
    op_p, op_m, op_best, op_best_chg, bases, ds = [], [], [], [], [], []
    dmall = []
    for i in range(len(recs)):
        tr_i = [j for j in range(len(recs)) if j != i]
        Xtr = np.vstack([raws[j] for j in tr_i])
        mu, sd, Wp = _pca_train(Xtr, var_keep, maxd); ds.append(Wp.shape[1])
        proj = lambda R: ((R - mu) / sd) @ Wp
        TR = [r for j in tr_i for r in transitions(proj(raws[j]), recs[j])]
        TE = transitions(proj(raws[i]), recs[i])
        dmall += [r[4] for r in TR]
        med = np.median([r[4] for r in TR]) if TR else 0.0
        changed = lambda r: r[3] or r[4] > med
        fit_rows = [r for r in TR if changed(r)] if fit_moving else TR
        Phi = np.array([np.concatenate([z, u, [1.0]]) for z, u, _, _, _ in fit_rows])
        Y = np.array([nx for _, _, nx, _, _ in fit_rows])
        theta = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ Y)
        ybar = np.array([nx for _, _, nx, _, _ in TR]).mean(0)          # climatology từ TRAIN (mọi transition)
        # chọn baseline TRÊN TRAIN rồi CỐ ĐỊNH (không dùng test)
        ep_tr = sum(np.sum((nx - z) ** 2) for z, _, nx, _, _ in TR)
        em_tr = sum(np.sum((nx - ybar) ** 2) for _, _, nx, _, _ in TR)
        base = "persist" if ep_tr <= em_tr else "mean"; bases.append(base)
        def errs(rows):
            X = np.array([np.concatenate([z, u, [1.0]]) for z, u, _, _, _ in rows])
            Zt = np.array([z for z, _, _, _, _ in rows]); NX = np.array([nx for _, _, nx, _, _ in rows])
            return (np.sum((NX - X @ theta) ** 2), np.sum((NX - Zt) ** 2), np.sum((NX - ybar) ** 2))
        eo, ep, em = errs(TE); eb = ep if base == "persist" else em
        op_p.append(1 - eo / (ep + 1e-12)); op_m.append(1 - eo / (em + 1e-12))
        op_best.append(1 - eo / (eb + 1e-12))
        ce = [r for r in TE if changed(r)]
        if ce:
            eo2, ep2, em2 = errs(ce); eb2 = ep2 if base == "persist" else em2
            op_best_chg.append(1 - eo2 / (eb2 + 1e-12))
    spread = np.mean([np.linalg.norm(raws[i] - raws[i].mean(0)) for i in range(len(recs))]) + 1e-9
    A = lambda L: np.array(L, float)
    return dict(op_p=A(op_p), op_m=A(op_m), op_best=A(op_best), op_best_chg=A(op_best_chg),
                base_persist=bases.count("persist"), d=int(np.median(ds)),
                mv=float(np.mean(dmall)), mvr=float(np.mean(dmall) / spread))

def boot(v, n=3000):
    v = np.asarray(v, float)
    if len(v) == 0: return (float("nan"),) * 3
    m = len(v); rng = np.random.default_rng(7)
    bs = np.array([v[rng.integers(0, m, m)].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=19); ap.add_argument("--model", default=None)
    ap.add_argument("--embed-model", default=None); ap.add_argument("--base-url", default=None)
    ap.add_argument("--var", type=float, default=0.9); ap.add_argument("--maxd", type=int, default=12)
    ap.add_argument("--ridge", type=float, default=1.0); ap.add_argument("--fit-moving", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
    if args.model: cfg["model"] = args.model
    if args.base_url: cfg["base_url"] = args.base_url
    N, T = cfg["n_agents"], cfg["n_rounds"]; ri = cfg.get("request_interval", 0.0)
    W = cfg.get("transcript_window", 4)
    ak = os.environ.get(cfg.get("api_key_env", ""), ""); akc = cfg.get("api_key_cmd"); burls = cfg.get("base_urls")
    emb_url = cfg.get("base_url", "http://localhost:11434/v1/chat/completions")
    emb_model = args.embed_model or cfg["model"]
    topics = json.load(open(args.topics)); Kans = len(topics[0]["answers"])
    base_seed = cfg.get("seed", 0) + args.seed; shuffle = cfg.get("shuffle_answers", True)
    mslug = re.sub(r"[^0-9a-zA-Z]+", "_", cfg["model"]).strip("_")
    tag = os.path.splitext(os.path.basename(args.topics))[0] + "_" + mslug

    def prep(ti):
        topic = topics[ti]; trng = np.random.default_rng(base_seed + ti)
        perm = trng.permutation(Kans) if shuffle else np.arange(Kans)
        a_star = int(np.where(perm == topic["a_star"])[0][0])
        x_adv = int(trng.choice([a for a in range(Kans) if a != a_star]))
        c = int(np.random.default_rng(base_seed + ti + 9973).choice([-1] + list(range(Kans))))
        return [topic["answers"][j] for j in perm], topic["question"], a_star, x_adv, c

    ck = os.path.join(OUT, f"ckpt_semantic2_{tag}.jsonl")
    done = [json.loads(l) for l in open(ck)] if os.path.exists(ck) else []
    print(f"embed {emb_model} @ {emb_url} | debate {cfg['model']} | resume {len(done)}/{args.n}")
    try:
        dim = len(embed("test", emb_url, emb_model)); print(f"embed dim={dim}")
    except Exception as e:
        sys.exit(f"LỖI embeddings ({emb_model}): {e}\n  thử: ollama pull nomic-embed-text ; --embed-model nomic-embed-text")

    for ti in range(len(done), args.n):
        ans, q, a_star, x_adv, c = prep(ti)
        print(f"[{ti+1}/{args.n}] adv={x_adv} intervene c={c} — debate + embed")
        traj, ts = run_debate(cfg, ak, akc, burls, q, ans, x_adv, c, T, ri, W)
        if empty_bad(ts):
            sys.exit("DỪNG: reply rỗng — sửa server, chạy lại (resume giữ nguyên).")
        buck = {}
        for t in ts:
            buck.setdefault(t["round"], {})[t["agent_id"]] = embed(t["text"], emb_url, emb_model)
        emb, belief = [], []
        for r in range(T):
            if len(buck.get(r, {})) == N:
                emb.append([buck[r][i].tolist() for i in range(N)])
                belief.append(mean_belief(traj[r + 1], N, Kans).tolist())
        open(ck, "a").write(json.dumps({"ti": ti, "x_adv": x_adv, "a_star": a_star,
                                        "c": c, "emb": emb, "belief": belief}) + "\n")
        done.append(1)

    recs = [json.loads(l) for l in open(ck) if len(json.loads(l)["belief"]) >= 3]
    R = loo_skill(recs, args.var, args.maxd, args.ridge, bool(args.fit_moving))
    pp, pl, ph = boot(R["op_p"]); mm, ml, mh = boot(R["op_m"])
    bb, bl, bh = boot(R["op_best"]); cc = boot(R["op_best_chg"])
    print(f"\n===== SEMANTIC-KOOPMAN PILOT ({tag}, n_q={len(recs)}, PCA d≈{R['d']}, var={args.var}, ridge={args.ridge},"
          f" fit_moving={bool(args.fit_moving)}) =====")
    print(f"[chẩn đoán] ‖Δz‖ TB={R['mv']:.3f}  Δ/spread={R['mvr']:.2f}"
          f"  ⇒ {'CÓ dịch chuyển' if R['mvr'] > 0.3 else 'gần STATIC (nghi chết)'}")
    print(f"[protocol]  PCA train-only | baseline chọn-trên-train = persist ở {R['base_persist']}/{len(recs)} fold")
    print(f"[ctx] operator vs persistence   = {pp:+.3f} CI[{pl:+.3f},{ph:+.3f}]   (tham khảo)")
    print(f"[ctx] operator vs climatology   = {mm:+.3f} CI[{ml:+.3f},{mh:+.3f}]   (tham khảo)")
    print(f"[TEETH]     op vs baseline-train-chọn        = {bb:+.3f} CI[{bl:+.3f},{bh:+.3f}]  ← BAR THẬT")
    print(f"[TEETH-chg] op vs baseline, CHANGED-only     = {cc[0]:+.3f} CI[{cc[1]:+.3f},{cc[2]:+.3f}]  (nhạy nhất)")
    live = bl > 0 or (cc[1] == cc[1] and cc[1] > 0)
    print(f"[verdict] {'✅ RĂNG semantic — escalate 4 model + theorem' if live else '❌ KHÔNG vượt baseline — dừng claim working-Koopman, chốt honest'}")
    print("\n[đọc] Δ>0 & CI-lower>0 ⇒ tín hiệu thật. Δ>0 nhưng CI chồng 0 ⇒ yếu: TĂNG SỐ QUESTION (không phải transition)")
    print("      + giảm nhiễu (var/maxd/ridge tune trên validation, KHÔNG theo test-CI). Δ<0 rõ, CI toàn âm ⇒ ĐỪNG")
    print("      tăng sample (càng khẳng định thua) — phải đổi observable/dynamics hoặc dừng cứu Koopman.")

if __name__ == "__main__":
    main()
