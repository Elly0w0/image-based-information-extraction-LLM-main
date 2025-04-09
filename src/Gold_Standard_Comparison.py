# Gold_Standard_Comparison.py

"""
Semantic Triple Gold Standard Comparison
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 27/03/2025

Description:
    This script evaluates the semantic similarity of subject-object triples extracted from biomedical images.
    It compares automatically extracted triples (e.g., GPT-generated) with a manually curated gold standard (CBM)
    using Sentence-BERT embeddings. 

    It calculates a Combined Similarity Score (CSS), which blends cosine similarity and F1-score, for each image.

Focus:
    - Compares only Subject–Object pairs (ignores Predicate).
    - Uses pre-trained sentence-transformers.
    - Skips images that are missing in either dataset.

Input:
    - Two Excel files:
        1. Gold standard file (e.g., CBM annotated triples)
        2. Evaluation file (e.g., GPT-generated triples)

Output:
    - Console output of per-image comparisons (with match quality)
    - Average Combined Similarity Score (CSS) across all images

Usage:
    Run from the project root:
    >>> python src/Gold_Standard_Comparison.py --gold <gold_file.xlsx> --eval <eval_file.xlsx>

Requirements:
    - pandas
    - numpy
    - sentence-transformers
    - matplotlib (optional, for future extensions)
"""

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import argparse

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Compare GPT triples to gold standard using semantic similarity (CSS).")
parser.add_argument("--gold", required=True, help="Path to manually curated (gold standard) Excel file")
parser.add_argument("--eval", required=True, help="Path to evaluation/generated triples file")
args = parser.parse_args()

file_Gold_path = args.gold
file_Eval_path = args.eval

# === Load Data ===
df_Gold = pd.read_excel(file_Gold_path)
df_Eval = pd.read_excel(file_Eval_path)

# === Load Pretrained Sentence-BERT Model ===
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_subject_object_dict(df):
    """
    Groups triples by Image_number, returning subject-object phrases.
    """
    triple_dict = {}
    for _, row in df.iterrows():
        key = row['Image_number']
        subj = row['Subject'].replace('_', ' ') if pd.notna(row['Subject']) else ''
        obj = row['Object'].replace('_', ' ') if pd.notna(row['Object']) else ''
        if subj and obj:
            phrase = f"{subj} {obj}"
            triple_dict.setdefault(key, []).append(phrase)
    return triple_dict

def similarity_score_subject_object(df_gold, df_extracted, image_key):
    """
    Computes Combined Similarity Score (CSS) for one image.
    CSS = 0.6 * avg_similarity + 0.4 * F1-score
    """
    gold_dict = create_subject_object_dict(df_gold)
    eval_dict = create_subject_object_dict(df_extracted)

    gold = gold_dict.get(image_key, [])
    extracted = eval_dict.get(image_key, [])
    if not gold or not extracted:
        return None

    gold_embeds = model.encode(gold, convert_to_tensor=True)
    eval_embeds = model.encode(extracted, convert_to_tensor=True)

    sims = util.cos_sim(eval_embeds, gold_embeds).cpu().numpy()
    SIM_THRESHOLD = 0.5
    TP, FP, FN = 0, 0, 0
    best_scores = []

    print(f"\nImage {image_key} - Matching extracted to gold triples:")
    for i, extracted_text in enumerate(extracted):
        if sims.shape[1] > 0:
            best_idx = np.argmax(sims[i])
            best_score = sims[i][best_idx]
            gold_match = gold[best_idx]
            match_str = "✅ MATCH" if best_score >= SIM_THRESHOLD else "❌ NO MATCH"
            print(f'  "{extracted_text}" ↔ "{gold_match}" | Score: {best_score:.4f} → {match_str}')
            best_scores.append(best_score)
            if best_score >= SIM_THRESHOLD:
                TP += 1
            else:
                FP += 1
        else:
            best_scores.append(0)

    FN = max(0, len(gold) - TP)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    avg_sim = np.mean(best_scores)

    css = 0.6 * avg_sim + 0.4 * f1
    return css

def compare_all_images(df_gold, df_eval):
    """
    Loops through all common images and calculates average CSS.
    """
    keys_gold = set(df_gold['Image_number'].unique())
    keys_eval = set(df_eval['Image_number'].unique())
    common_keys = keys_gold & keys_eval

    def natural_sort_key(k):
        return int(k.split('_')[-1])

    css_list = []
    for key in sorted(common_keys, key=natural_sort_key):
        css = similarity_score_subject_object(df_gold, df_eval, key)
        if css is not None:
            css_list.append(css)

    return sum(css_list) / len(css_list) if css_list else 0

# === Main Execution ===
if __name__ == "__main__":
    avg_css = compare_all_images(df_Gold, df_Eval)
    print(f"\n✅ Average Combined Similarity Score (CSS): {avg_css:.4f}")

# === Example usage ===
# python src/Gold_Standard_Comparison.py --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard.xlsx --eval data/gold_standard_comparison/Triples_GPT_for_comparison.xlsx
