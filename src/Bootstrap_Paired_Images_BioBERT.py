# Bootstrap_Paired_Images_BioBERT.py

"""
Bootstrap Paired Analysis over Images for Triple Matching with BioBERT + Hungarian
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 2025-11-04

Description:
1) For each EVAL file (prompt variant), matches GPT triples to gold triples PER IMAGE:
   - Encode full triples as sentences with BioBERT
   - Cosine similarity
   - One-to-one global assignment within an image via the Hungarian algorithm
   - Count TP/FP/FN per image at a fixed similarity threshold

2) Stores per-image TP/FP/FN ONCE per prompt (heavy part).

3) Runs *paired bootstrap over images* for every pair of prompts:
   - Unit of resampling = image
   - B iterations (default 10_000)
   - Reports ΔF1, ΔPrecision, ΔRecall with 95% CI and two-sided p-values.

Inputs:
    - Gold standard triples subset (Excel)
    - GPT-extracted triples (for different prompts) subset (Excel)
   
Outputs:
- Console report with per-pair results
- CSV table with Δ metrics and CIs
- (Optional) CSV with raw bootstrap draws for reproducibility
- (Optional) Hist PNGs of ΔF1

Usage:
python src/Bootstrap_Paired_Images_BioBERT.py \
  --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard.xlsx \
  --eval data/gold_standard_comparison/Triples_GPT_freeform.xlsx --label FreeForm \
  --eval data/gold_standard_comparison/Triples_GPT_template.xlsx --label Template \
  --eval data/gold_standard_comparison/Triples_GPT_hybrid.xlsx --label Hybrid \
  --threshold 0.85 --B 10000 --seed 42 --outdir results/bootstrap --save_raw
"""

import os
import re
import argparse
import itertools
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------
# Model (loaded once)
# -----------------------
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model

# -----------------------
# Text utils
# -----------------------
def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def format_triple(s, p, o):
    return f"{normalize(s)} {normalize(p)} {normalize(o)}".strip()

# -----------------------
# Embeddings (with cache)
# -----------------------
class SentenceEmbedder:
    def __init__(self, max_length=64):
        self.tokenizer, self.model = _load_model()
        self.max_length = max_length
        self.cache = {}

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        if text in self.cache:
            return self.cache[text]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                padding=True, max_length=self.max_length)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze()  # [hidden]
        self.cache[text] = emb
        return emb

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# -----------------------
# Data shaping
# -----------------------
def group_triples(df: pd.DataFrame, image_col="Image_number"):
    """Return dict: image_id -> list of (S,P,O)"""
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        s, p, o = row.get("Subject"), row.get("Predicate"), row.get("Object")
        if all(pd.notna([s, p, o])):
            grouped[row[image_col]].append((s, p, o))
    return dict(grouped)

# -----------------------
# Matching per image
# -----------------------
def match_image_triples(gold_triples, eval_triples, embedder: SentenceEmbedder, threshold: float):
    """Return TP, FP, FN and optionally details for one image."""
    if len(gold_triples) == 0 and len(eval_triples) == 0:
        return 0, 0, 0

    gold_sent = [format_triple(*t) for t in gold_triples]
    eval_sent = [format_triple(*t) for t in eval_triples]

    emb_gold = [embedder.embed(s) for s in gold_sent]
    emb_eval = [embedder.embed(s) for s in eval_sent]

    if len(gold_sent) == 0:
        # no gold, everything predicted is FP
        return 0, len(eval_sent), 0
    if len(eval_sent) == 0:
        # no predictions, all gold are FN
        return 0, 0, len(gold_sent)

    sim_matrix = np.zeros((len(eval_sent), len(gold_sent)), dtype=float)
    for i, e_emb in enumerate(emb_eval):
        for j, g_emb in enumerate(emb_gold):
            sim_matrix[i, j] = cosine_similarity(e_emb, g_emb)

    # Hungarian on cost
    cost = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_eval = set()
    matched_gold = set()
    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i, j] >= threshold:
            matched_eval.add(i)
            matched_gold.add(j)

    TP = len(matched_eval)
    FP = max(0, len(eval_sent) - TP)
    FN = max(0, len(gold_sent) - len(matched_gold))
    return TP, FP, FN

# -----------------------
# Evaluate a whole file (per-image stats)
# -----------------------
def per_image_stats(df_gold: pd.DataFrame, df_eval: pd.DataFrame, threshold: float):
    """Return dict: image_id -> {'TP':int,'FP':int,'FN':int}"""
    gold_dict = group_triples(df_gold)
    eval_dict = group_triples(df_eval)

    # intersect only images present in both
    common = sorted(set(gold_dict.keys()) & set(eval_dict.keys()),
                    key=lambda x: int(re.findall(r"\d+", str(x))[-1]) if re.findall(r"\d+", str(x)) else 0)
    embedder = SentenceEmbedder()
    stats = {}
    for image_id in common:
        TP, FP, FN = match_image_triples(gold_dict[image_id], eval_dict[image_id], embedder, threshold)
        stats[image_id] = {"TP": TP, "FP": FP, "FN": FN}
    return stats

# -----------------------
# Metrics / Bootstrap
# -----------------------
def _prec_rec_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1

def bootstrap_paired(resultsA: dict, resultsB: dict, B=10000, seed=42, return_raw=False):
    """
    Paired bootstrap over images.
    results* : dict[image_id] -> {'TP','FP','FN'}
    Returns summary dict and, optionally, raw draws for ΔF1/ΔP/ΔR.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # keep only intersection of images
    imgs = sorted(set(resultsA.keys()) & set(resultsB.keys()))
    n = len(imgs)
    if n == 0:
        raise ValueError("No overlapping images between systems A and B.")

    dF1, dP, dR = np.empty(B), np.empty(B), np.empty(B)

    for b in range(B):
        sample = [rng.choice(imgs) for _ in range(n)]

        tpA = sum(resultsA[i]["TP"] for i in sample)
        fpA = sum(resultsA[i]["FP"] for i in sample)
        fnA = sum(resultsA[i]["FN"] for i in sample)

        tpB = sum(resultsB[i]["TP"] for i in sample)
        fpB = sum(resultsB[i]["FP"] for i in sample)
        fnB = sum(resultsB[i]["FN"] for i in sample)

        pA, rA, f1A = _prec_rec_f1(tpA, fpA, fnA)
        pB, rB, f1B = _prec_rec_f1(tpB, fpB, fnB)

        dF1[b] = f1A - f1B
        dP[b]  = pA - pB
        dR[b]  = rA - rB

    def summarize(arr):
        low, high = np.percentile(arr, [2.5, 97.5])
        p_two = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        return float(np.mean(arr)), float(low), float(high), float(p_two)

    f1_mean, f1_lo, f1_hi, f1_p = summarize(dF1)
    p_mean, p_lo, p_hi, p_p = summarize(dP)
    r_mean, r_lo, r_hi, r_p = summarize(dR)

    summary = {
        "dF1_mean": f1_mean, "dF1_CI_low": f1_lo, "dF1_CI_high": f1_hi, "dF1_p": f1_p,
        "dP_mean": p_mean,   "dP_CI_low": p_lo,   "dP_CI_high": p_hi,   "dP_p": p_p,
        "dR_mean": r_mean,   "dR_CI_low": r_lo,   "dR_CI_high": r_hi,   "dR_p": r_p,
        "N_images": n, "B": B, "seed": seed
    }

    if return_raw:
        return summary, {"dF1": dF1, "dP": dP, "dR": dR}
    return summary

# -----------------------
# Orchestration
# -----------------------
def load_xlsx(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_excel(path)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def pairwise(iterable):
    return itertools.combinations(iterable, 2)

def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap over images for BioBERT+Hungarian triple matching.")
    parser.add_argument("--gold", required=True, help="Path to gold standard subset triples (.xlsx) with columns: Image_number, Subject, Predicate, Object")
    parser.add_argument("--eval", action="append", required=True, help="Path to EVAL (.xlsx) for a prompt variant; can be repeated")
    parser.add_argument("--label", action="append", help="Label for the corresponding --eval; same count/order as --eval")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold [default: 0.85]")
    parser.add_argument("--B", type=int, default=10000, help="Bootstrap iterations [default: 10000]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed [default: 42]")
    parser.add_argument("--outdir", type=str, default="data/bootstrap", help="Output directory")
    parser.add_argument("--save_raw", action="store_true", help="Save raw bootstrap draws per pair")
    parser.add_argument("--save_plots", action="store_true", help="Save ΔF1 histograms per pair")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # Labels
    eval_paths = args.eval
    if args.label:
        if len(args.label) != len(eval_paths):
            raise ValueError("--label must be given the same number of times as --eval")
        labels = args.label
    else:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in eval_paths]

    print(f"Loading GOLD: {args.gold}")
    df_gold = load_xlsx(args.gold)

    # Compute per-image stats once per prompt file
    per_prompt_stats = {}
    for pth, lab in zip(eval_paths, labels):
        print(f"\n=== Evaluating prompt '{lab}' from {pth}")
        df_eval = load_xlsx(pth)
        stats = per_image_stats(df_gold, df_eval, threshold=args.threshold)
        n_imgs = len(stats)
        # Also store aggregate P/R/F1 on full intersection
        tp = sum(v["TP"] for v in stats.values())
        fp = sum(v["FP"] for v in stats.values())
        fn = sum(v["FN"] for v in stats.values())
        P, R, F1 = _prec_rec_f1(tp, fp, fn)
        print(f"-> {lab}: N_images={n_imgs}, P={P:.3f}, R={R:.3f}, F1={F1:.3f}")
        per_prompt_stats[lab] = stats

    # Pairwise bootstrap across all labels
    rows = []
    for (labA, statsA), (labB, statsB) in pairwise(per_prompt_stats.items()):
        print(f"\n### Bootstrap: {labA} vs {labB} (paired over images)")
        try:
            summary, raw = bootstrap_paired(statsA, statsB, B=args.B, seed=args.seed, return_raw=True)
        except ValueError as e:
            print(f"Skipping {labA} vs {labB}: {e}")
            continue

        print(f"ΔF1={summary['dF1_mean']:.3f} "
              f"[{summary['dF1_CI_low']:.3f}; {summary['dF1_CI_high']:.3f}], p={summary['dF1_p']:.3f}")
        print(f"ΔP ={summary['dP_mean']:.3f} "
              f"[{summary['dP_CI_low']:.3f}; {summary['dP_CI_high']:.3f}], p={summary['dP_p']:.3f}")
        print(f"ΔR ={summary['dR_mean']:.3f} "
              f"[{summary['dR_CI_low']:.3f}; {summary['dR_CI_high']:.3f}], p={summary['dR_p']:.3f}")
        print(f"N_images={summary['N_images']}, B={summary['B']}, seed={summary['seed']}")

        rows.append({
            "A_label": labA, "B_label": labB,
            **summary
        })

        # Optional raw draws
        if args.save_raw:
            raw_df = pd.DataFrame({
                "dF1": raw["dF1"],
                "dP": raw["dP"],
                "dR": raw["dR"]
            })
            out_raw = os.path.join(args.outdir, f"bootstrap_raw_{labA}_vs_{labB}.csv")
            raw_df.to_csv(out_raw, index=False)
            print(f"Saved raw draws to: {out_raw}")

        # Optional histogram
        if args.save_plots:
            plt.figure(figsize=(6,4), dpi=300)
            plt.hist(raw["dF1"], bins=30, edgecolor="black")
            plt.xlabel("ΔF1 (A − B)")
            plt.ylabel("Frequency")
            plt.title(f"Bootstrap ΔF1: {labA} − {labB}")
            out_png = os.path.join(args.outdir, f"hist_dF1_{labA}_vs_{labB}.png")
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()
            print(f"Saved ΔF1 histogram to: {out_png}")

    # Save summary table
    if rows:
        out_csv = os.path.join(args.outdir, "bootstrap_summary_pairs.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSaved summary table to: {out_csv}")
    else:
        print("\nNo pairwise results produced (check overlaps).")


if __name__ == "__main__":
    main()

# python src/Bootstrap_Paired_Images_BioBERT.py --gold data/prompt_engineering/cbm_files/CBM_subset_50_URL_triples.xlsx --eval data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_0.xlsx --label FreeForm --eval data/prompt_engineering/gpt_files/GPT_subset_triples_prompt2_param0_0.xlsx --label Template --eval data/prompt_engineering/gpt_files/GPT_subset_triples_prompt3_param0_0.xlsx --label Hybrid --threshold 0.85 --B 10000 --seed 42 --outdir data/bootstrap --save_raw --save_plots