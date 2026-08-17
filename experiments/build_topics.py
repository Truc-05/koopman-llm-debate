"""Build topic-set JSON (schema {question, answers:[K], a_star}) cho debate pipeline từ HF datasets.
Mỗi file topic phải UNIFORM số option (pipeline lọc theo topics[0]). Dùng:
  python experiments/build_topics.py mmlu_math   200 experiments/topics/math.json        # K=4
  python experiments/build_topics.py truthfulqa  200 experiments/topics/truthfulqa.json   # K=4
  python experiments/build_topics.py arc         200 experiments/topics/arc.json          # K=4
  python experiments/build_topics.py commonsense_qa 200 experiments/topics/commonsense_qa.json # K=5
  python experiments/build_topics.py bbh_logic5  200 experiments/topics/bbh_logic5.json   # K=5
  python experiments/build_topics.py bbh_logic7  200 experiments/topics/bbh_logic7.json   # K=7
  python experiments/build_topics.py bbh_track7  200 experiments/topics/bbh_track7.json   # K=7
Cần: pip install datasets  (+ internet). Nếu không có, báo mình đổi sang đọc file local.
KHÔNG tự chạy — user tự chạy.
"""
import sys, json, re
import numpy as np

BENCH, N, OUTP = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(0)
MATH_SUBJ = {"high_school_mathematics", "college_mathematics", "abstract_algebra", "elementary_mathematics"}


def pack(question, options, correct_idx, k=4):
    """→ dict schema chuẩn nếu đúng k option; ngược lại None (bỏ)."""
    if len(options) != k or not (0 <= correct_idx < k):
        return None
    if any(not str(o).strip() for o in options):
        return None
    return {"question": str(question).strip(), "answers": [str(o).strip() for o in options], "a_star": int(correct_idx)}


def from_mmlu_math():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.filter(lambda r: r["subject"] in MATH_SUBJ)
    for r in ds:
        yield pack(r["question"], r["choices"], r["answer"])


def from_truthfulqa():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    for r in ds:
        ch = r["mc1_targets"]["choices"]; lab = r["mc1_targets"]["labels"]
        cor = [c for c, l in zip(ch, lab) if l == 1]
        wr = [c for c, l in zip(ch, lab) if l == 0]
        if not cor or len(wr) < 3:
            yield None; continue
        opts = [cor[0]] + list(rng.choice(wr, 3, replace=False))
        order = rng.permutation(4)
        yield pack(r["question"], [opts[i] for i in order], int(np.where(order == 0)[0][0]))


def from_arc():
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for r in ds:
        opts = r["choices"]["text"]; labs = r["choices"]["label"]
        if r["answerKey"] not in labs:
            yield None; continue
        yield pack(r["question"], opts, labs.index(r["answerKey"]))


def from_commonsenseqa():
    """CommonsenseQA — K=5 (A–E). Dùng split validation (test không có answerKey)."""
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="validation")  # parquet, không cần trust_remote_code
    for r in ds:
        opts = r["choices"]["text"]; labs = r["choices"]["label"]
        if r["answerKey"] not in labs:
            yield None; continue
        yield pack(r["question"], opts, labs.index(r["answerKey"]), 5)


def _parse_bbh_mc(inp):
    """BBH input = stem + 'Options:\\n(A) ... (B) ...'. → (stem, [texts]) hoặc (None, None)."""
    if "Options:" not in inp:
        return None, None
    stem, block = inp.split("Options:", 1)
    # mỗi option: (X) <text tới option kế / hết chuỗi>
    matches = re.findall(r"\(([A-Z])\)\s*(.*?)(?=\s*\n\([A-Z]\)|\Z)", block, flags=re.S)
    letters = [m[0] for m in matches]
    texts = [re.sub(r"\s+", " ", m[1]).strip() for m in matches]
    return stem.strip(), (letters, texts)


def _from_bbh(task, k):
    """factory: trả generator đọc 1 task BBH multiple-choice, chỉ giữ câu đúng k option."""
    def gen():
        from datasets import load_dataset
        ds = load_dataset("lukaemon/bbh", task, split="test")  # nếu lỗi remote-code: thêm trust_remote_code=True
        for r in ds:
            stem, parsed = _parse_bbh_mc(r["input"])
            if parsed is None:
                yield None; continue
            letters, texts = parsed
            tgt = r["target"].strip().strip("()")               # '(D)' → 'D'
            if tgt not in letters:
                yield None; continue
            yield pack(stem, texts, letters.index(tgt), k)
    return gen


SRC = {"mmlu_math": from_mmlu_math, "truthfulqa": from_truthfulqa, "arc": from_arc,
       "commonsense_qa": from_commonsenseqa,
       "bbh_logic5": _from_bbh("logical_deduction_five_objects", 5),
       "bbh_logic7": _from_bbh("logical_deduction_seven_objects", 7),
       "bbh_track7": _from_bbh("tracking_shuffled_objects_seven_objects", 7)}
if BENCH not in SRC:
    sys.exit(f"benchmark phải là một trong {list(SRC)}")

topics = [t for t in SRC[BENCH]() if t is not None]
if len(topics) < N:
    print(f"⚠️ chỉ có {len(topics)} câu 4-option hợp lệ (< {N}) — dùng hết.")
topics = topics[:N]
json.dump(topics, open(OUTP, "w"), ensure_ascii=False, indent=1)
print(f"✓ {BENCH}: ghi {len(topics)} topic → {OUTP}")
