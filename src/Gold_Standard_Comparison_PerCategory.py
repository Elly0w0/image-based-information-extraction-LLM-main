"""
Per-Category Semantic Triple Comparison using BioBERT (Hungarian Matching)
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 30/10/2025

Description:
    Computes precision, recall, and F1 scores for GPT-4o-extracted triples
    against CBM Gold Standard, grouped by biological domain (Category).
    Uses BioBERT embeddings and Hungarian matching for semantic alignment.

Input:
    - Gold standard triples (Excel)
    - GPT-extracted triples (Excel)

Output:
    - Excel file with per-category performance metrics

Usage:
    python src/Gold_Standard_Comparison_PerCategory.py \
    --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard_SubjObj_Categorized.xlsx \
    --eval data/gold_standard_comparison/Triples_GPT_for_comparison_SubjObj_Categorized.xlsx \
    --threshold 0.85
"""

import pandas as pd
import numpy as np
import torch
import re
import argparse
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
import os

# === Load BioBERT ===
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# === Helper Functions ===

def normalize(text):
    if pd.isna(text):
        return ""
    text = text.lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_triple(s, p, o):
    return f"{normalize(s)} {normalize(p)} {normalize(o)}"

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def group_triples(df):
    grouped = {}
    for _, row in df.iterrows():
        if all(pd.notna([row['Subject'], row['Predicate'], row['Object']])):
            grouped.setdefault(row['Image_number'], []).append(
                (row['Subject'], row['Predicate'], row['Object'])
            )
    return grouped

def evaluate_images(df_gold, df_eval, threshold=0.85, silent=False):
    gold_dict = group_triples(df_gold)
    eval_dict = group_triples(df_eval)

    common_images = sorted(set(gold_dict.keys()) & set(eval_dict.keys()), key=lambda x: int(x.split('_')[-1]))
    total_TP = total_FP = total_FN = 0

    for image_id in common_images:
        gold_triples = gold_dict[image_id]
        eval_triples = eval_dict[image_id]

        gold_sentences = [format_triple(*t) for t in gold_triples]
        eval_sentences = [format_triple(*t) for t in eval_triples]

        emb_gold = [get_embedding(t) for t in gold_sentences]
        emb_eval = [get_embedding(t) for t in eval_sentences]

        sim_matrix = np.zeros((len(eval_triples), len(gold_triples)))

        for i, e_emb in enumerate(emb_eval):
            for j, g_emb in enumerate(emb_gold):
                sim_matrix[i, j] = cosine_similarity(e_emb, g_emb)

        cost_matrix = 1 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gpt = set()
        matched_cbm = set()

        for i, j in zip(row_ind, col_ind):
            if sim_matrix[i, j] >= threshold:
                matched_gpt.add(i)
                matched_cbm.add(j)

        TP = len(matched_gpt)
        FP = len(eval_triples) - TP
        FN = len(gold_triples) - len(matched_cbm)

        total_TP += TP
        total_FP += FP
        total_FN += FN

        if not silent:
            print(f"Image {image_id}: TP={TP}, FP={FP}, FN={FN}")

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return total_TP, total_FP, total_FN, precision, recall, f1


def evaluate_per_category(df_gold, df_eval, threshold=0.85):
    categories = sorted(set(df_gold['Category']).intersection(set(df_eval['Category'])))
    results = []

    for cat in categories:
        print(f"\n=== Evaluating category: {cat} ===")
        df_gold_cat = df_gold[df_gold['Category'] == cat]
        df_eval_cat = df_eval[df_eval['Category'] == cat]

        TP, FP, FN, precision, recall, f1 = evaluate_images(df_gold_cat, df_eval_cat, threshold=threshold, silent=True)

        print(f"→ TP={TP}, FP={FP}, FN={FN}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        results.append({
            "Category": cat,
            "TP": TP, "FP": FP, "FN": FN,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3)
        })

    df_res = pd.DataFrame(results)
    out_dir = "data/gold_standard_comparison"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Per_Category_Performance.xlsx")
    df_res.to_excel(out_path, index=False)
    print(f"\n✅ Saved per-category performance to {out_path}")

    return df_res


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GPT and CBM triples per biological category using BioBERT")
    parser.add_argument("--gold", required=True, help="Path to gold standard (.xlsx)")
    parser.add_argument("--eval", required=True, help="Path to GPT triples (.xlsx)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")
    args = parser.parse_args()

    df_gold = pd.read_excel(args.gold)
    df_eval = pd.read_excel(args.eval)

    print("\nStarting per-category evaluation...")
    evaluate_per_category(df_gold, df_eval, threshold=args.threshold)

# python src/Gold_Standard_Comparison_PerCategory.py --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard_SubjObj_Categorized.xlsx --eval data/gold_standard_comparison/Triples_GPT_for_comparison_SubjObj_Categorized.xlsx