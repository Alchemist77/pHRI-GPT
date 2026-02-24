import os, json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Prompt (BINARY)
# -------------------------
def build_prompt(description: str) -> str:
    return (
        "You are an expert evaluator of SELF-REPORTED human experience in physical human–robot interaction (PHRI).\n"
        "This is NOT about basic emotions (happy/sad) and NOT about task success or performance.\n"
        "Judge ONLY how the interaction FELT to the human in terms of comfort, safety, trust, and naturalness.\n\n"

        "You must choose exactly ONE label: Negative or Positive.\n\n"

        "Use these criteria (focus on physical interaction quality):\n"
        "- Comfort & ease: relaxed vs tense/guarded\n"
        "- Safety & control: safe/predictable vs risky/unpredictable\n"
        "- Physical feel: smooth/compliant vs stiff/jarring/jerky\n"
        "- Timing & coordination: in-sync vs out-of-sync/awkward\n"
        "- Force/contact: appropriate/gentle vs excessive/uncertain\n"
        "- Trust: confident/reassured vs doubtful/on edge\n\n"

        "Negative examples (ANY strong sign => Negative):\n"
        "- awkward, uncomfortable, tense, uneasy, unsettling\n"
        "- clunky, jerky, sudden, too forceful, unstable, unpredictable\n"
        "- hard to relax, cautious, on edge, mismatch in timing\n\n"

        "Positive examples:\n"
        "- smooth, comfortable, natural, in sync\n"
        "- safe, controlled, gentle, well-timed\n"
        "- reassuring, trustworthy, confident\n\n"

        "Interaction description:\n"
        f"{description}\n\n"

        "Answer with exactly one word: Negative or Positive.\n"
        "Label: "
    )


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def cm2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in (0, 1) and p in (0, 1):
            cm[t, p] += 1
    return cm

def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0

def precision_recall_f1(cm: np.ndarray) -> dict:
    tp = float(cm[1, 1])
    fp = float(cm[0, 1])
    fn = float(cm[1, 0])

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision_pos": prec, "recall_pos": rec, "f1_pos": f1}

def find_stage2_csvs(exp_dir: str, kfold: int = 5) -> List[Tuple[int, str]]:
    out = []
    for f in range(1, kfold + 1):
        p = os.path.join(exp_dir, f"fold_{f}", "stage2_outputs.csv")
        if os.path.exists(p):
            out.append((f, p))
    return out

def discover_experiments(stage2_root: str, tasks: List[str]) -> List[str]:
    exp_dirs = []
    for task in tasks:
        task_dir = os.path.join(stage2_root, task)
        if not os.path.isdir(task_dir):
            continue
        for model_name in sorted(os.listdir(task_dir)):
            p_model = os.path.join(task_dir, model_name)
            if not os.path.isdir(p_model):
                continue
            for llm_name in sorted(os.listdir(p_model)):
                p_exp = os.path.join(p_model, llm_name)
                if not os.path.isdir(p_exp):
                    continue
                if os.path.exists(os.path.join(p_exp, "fold_1", "stage2_outputs.csv")):
                    exp_dirs.append(p_exp)
    return exp_dirs

def match_filter(path: str, exp_filter: Optional[str]) -> bool:
    if not exp_filter:
        return True
    return exp_filter.lower() in path.lower()


# -------------------------
# Judge
# -------------------------
@torch.no_grad()
def classify_binary_logits(model, tokenizer, sentence: str, label_token_ids: List[int]) -> Tuple[int, List[float]]:
    prompt = build_prompt(sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    scores = [float(logits[0, tid].item()) for tid in label_token_ids]
    pred = int(np.argmax(scores))
    return pred, scores


def evaluate_one_csv(
    csv_path: str,
    model,
    tokenizer,
    labels: List[str],
    label_token_ids: List[int],
) -> Tuple[pd.DataFrame, dict]:

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"CSV missing label column: {csv_path}")

    y_true = df["label"].astype(int).to_numpy()
    uniq = sorted(list(set(y_true.tolist())))
    if not set(uniq).issubset({0, 1}):
        raise ValueError(f"Binary judge requires GT in {{0,1}} but got {uniq}")

    rows = []
    y_pred_real, y_pred_zero = [], []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        for mode in ["real", "zero"]:
            col = f"gen_{mode}"
            if col not in df.columns:
                continue

            sent = str(r[col])
            pred, scores = classify_binary_logits(model, tokenizer, sent, label_token_ids)

            if mode == "real":
                y_pred_real.append(pred)
            else:
                y_pred_zero.append(pred)

            rows.append({
                "episode_id": r.get("episode_id", ""),
                "gt_label": int(r["label"]),
                "mode": mode,
                "pred_label": labels[pred],
                "pred_idx": int(pred),
                "match": int(int(pred) == int(r["label"])),
                col: sent,
                "logit_Negative": scores[0],
                "logit_Positive": scores[1],
            })

    metrics = {}
    for mode, y_pred in [("real", y_pred_real), ("zero", y_pred_zero)]:
        if len(y_pred) == 0:
            continue
        y_pred = np.array(y_pred, dtype=int)
        cm = cm2(y_true, y_pred)
        m = {
            f"acc_{mode}": acc(y_true, y_pred),
            f"cm_{mode}": cm.tolist(),
        }
        m.update({f"{k}_{mode}": v for k, v in precision_recall_f1(cm).items()})
        metrics.update(m)

    return pd.DataFrame(rows), metrics


def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--stage2_root", type=str, default="./stage2_out")
    ap.add_argument("--stage3_root", type=str, default="./stage3_out")
    ap.add_argument("--task", type=str, default="all", choices=["all", "ts", "img", "fusion"])
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--exp_filter", type=str, default="")
    ap.add_argument("--judge_model", type=str, default="/home/toor/jaeseok/language_models/gpt-oss-20b")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    args = ap.parse_args()

    tasks = ["ts", "img", "fusion"] if args.task == "all" else [args.task]
    exp_dirs = discover_experiments(args.stage2_root, tasks)
    exp_dirs = [p for p in exp_dirs if match_filter(p, args.exp_filter)]
    exp_dirs = sorted(exp_dirs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True)

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    ).eval()

    LABELS = ["Negative", "Positive"]
    label_token_ids = [tokenizer.encode(" " + lab, add_special_tokens=False)[-1] for lab in LABELS]

    for exp_dir in exp_dirs:
        rel = os.path.relpath(exp_dir, args.stage2_root)
        out_exp = ensure_dir(os.path.join(args.stage3_root, rel))

        fold_csvs = find_stage2_csvs(exp_dir, args.kfold)
        all_preds, fold_metrics = [], []

        for f, csv_path in fold_csvs:
            out_fold = ensure_dir(os.path.join(out_exp, f"fold_{f}"))
            pred_df, metrics = evaluate_one_csv(csv_path, model, tokenizer, LABELS, label_token_ids)

            pred_df.to_csv(os.path.join(out_fold, "stage3_predictions.csv"), index=False)
            save_json(os.path.join(out_fold, "metrics.json"), metrics)

            fold_metrics.append({"fold": f, **metrics})
            all_preds.append(pred_df)

        merged = pd.concat(all_preds, ignore_index=True)
        merged_csv = os.path.join(out_exp, "stage3_predictions_all_folds.csv")
        merged.to_csv(merged_csv, index=False)

        summary = {
            "exp_dir": exp_dir,
            "out_dir": out_exp,
            "fold_metrics": fold_metrics,
            "merged_predictions_csv": merged_csv,
        }

        save_json(os.path.join(out_exp, "summary_all_folds.json"), summary)


if __name__ == "__main__":
    main()
