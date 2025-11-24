# src/NearThreshold_and_ErrorTypology.py
"""
Near-Threshold Analysis and Error Typology for GPT–CBM Triple Matching
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 04/11/2025

Description:
    This utility consumes the per-pair similarity report produced by the BioBERT + Hungarian
    matching pipeline (e.g., CBM_comparison_Report_Threshold_85.xlsx) and performs:
      (i) Near-threshold similarity analysis around the chosen cutoff,
      (ii) Automatic qualitative error typing for below-threshold pairs.

    It reports global counts, near-threshold proportions, and a breakdown of error categories.
    Filtered tables are saved to Excel for manual inspection and for inclusion in supplementary materials.

Input:
    - Pairwise comparison report (.xlsx) with at least the columns:
        ['Image', 'GPT_Triple', 'CBM_Triple', 'Similarity']
      Optional columns (if present) will be used:
        ['System Decision']  # e.g., "Match" / "No Match"

Output:
    - Console summary with counts and percentages
    - Excel files:
        data/gold_standard_comparison/NearThreshold_Triples_<L>_<U>.xlsx
        data/gold_standard_comparison/ErrorTypology_Summary.xlsx
        data/gold_standard_comparison/ErrorTypology_Detailed.xlsx

Usage:
    python src/NearThreshold_and_ErrorTypology.py \
        --report data/gold_standard_comparison/CBM_comparison_Report_Threshold_85.xlsx \
        --threshold 0.85 --delta 0.05 \
        --outdir data/gold_standard_comparison
"""

import argparse
import os
import re
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def ensure_float_series(s):
    """Coerce a pandas Series to float, tolerating comma decimal separators."""
    return s.astype(str).str.replace(",", ".", regex=False).astype(float)


def normalize(text: str) -> str:
    """Lowercase, remove punctuation except spaces, squeeze spaces."""
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9\s\-_]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


PREDICATE_LEXICON = [
    # multi-word first (checked in order)
    "leads to", "contributes to", "associated with", "associates with",
    "correlates with", "promotes", "induces", "triggers", "activates",
    "inhibits", "increases", "decreases", "reduces", "causes",
    "modulates", "mediates", "drives", "results in", "up regulates",
    "down regulates", "upregulates", "downregulates"
]


def split_triple_smart(triple: str):
    """
    Split 'subject predicate object' using a small predicate lexicon;
    fallback to a naive split if nothing is found.
    """
    txt = normalize(triple)
    for pred in PREDICATE_LEXICON:
        if f" {pred} " in f" {txt} ":
            parts = txt.split(pred, 1)
            subj = parts[0].strip()
            obj = parts[1].strip()
            return subj, pred, obj
    # Fallback: last two tokens heuristic (robust for 'x increases y')
    tokens = txt.split()
    if len(tokens) < 3:
        return "", "", txt
    # assume predicate is a single token between subj and obj
    subj = " ".join(tokens[:-2])
    pred = tokens[-2]
    obj = tokens[-1]
    return subj, pred, obj


def str_sim(a: str, b: str) -> float:
    """String similarity ratio [0,1] using SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()


def classify_error_row(row, cutoff: float):
    """
    Classify an unmatched pair (Similarity < cutoff) into a coarse error type.
    Rules:
      - Entity boundary: predicate is similar, but subj or obj diverges strongly
      - Predicate mismatch: subj & obj are similar, predicate diverges
      - Granularity: subj or obj moderately similar (near miss)
      - Other: does not meet above patterns
      - Match: Similarity >= cutoff
    """
    if row["Similarity"] >= cutoff:
        return "Match"

    pred_sim, subj_sim, obj_sim = row["sim_pred"], row["sim_subj"], row["sim_obj"]

    # Predicate close, but entities off
    if pred_sim > 0.80 and (subj_sim < 0.70 or obj_sim < 0.70):
        return "Entity boundary"

    # Entities close, predicate off
    if pred_sim < 0.80 and (subj_sim > 0.80 and obj_sim > 0.80):
        return "Predicate mismatch"

    # Entities near but not close enough (granularity drift)
    if (0.70 <= subj_sim < 0.90) or (0.70 <= obj_sim < 0.90):
        return "Granularity"

    return "Other"


# ----------------------------
# Core
# ----------------------------
def near_threshold_and_errors(report_path: str, threshold: float, delta: float, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Load
    df = pd.read_excel(report_path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if "Similarity" not in df.columns:
        raise ValueError("Input report must contain a 'Similarity' column.")
    # Coerce floats
    df["Similarity"] = ensure_float_series(df["Similarity"])

    # Near-threshold slice
    lower, upper = threshold - delta, threshold + delta
    df_near = df[(df["Similarity"] >= lower) & (df["Similarity"] <= upper)]

    total = len(df)
    near_n = len(df_near)
    near_pct = 100.0 * near_n / total if total else 0.0
    near_mean = df_near["Similarity"].mean() if near_n else float("nan")
    near_median = df_near["Similarity"].median() if near_n else float("nan")

    print("\n=== Near-threshold analysis ===")
    print(f"Report: {report_path}")
    print(f"Threshold: {threshold:.2f} | Delta: ±{delta:.2f}  → Range: [{lower:.2f}, {upper:.2f}]")
    print(f"Total pairs: {total}")
    print(f"Near-threshold pairs: {near_n} ({near_pct:.1f}%)")
    if near_n:
        print(f"Mean similarity (near): {near_mean:.3f}")
        print(f"Median similarity (near): {near_median:.3f}")

    # Save near-threshold rows
    near_out = os.path.join(outdir, f"NearThreshold_Triples_{int(lower*100)}_{int(upper*100)}.xlsx")
    df_near.to_excel(near_out, index=False)
    print(f"Saved near-threshold examples to: {near_out}")

    # Error typology (requires triple strings)
    required_cols = {"GPT_Triple", "CBM_Triple"}
    if not required_cols.issubset(set(df.columns)):
        print("\n[Warn] Error typology skipped: missing GPT_Triple/CBM_Triple columns.")
        return

    # Prepare normalized triplets & parts
    for col in ["GPT_Triple", "CBM_Triple"]:
        df[f"{col}_norm"] = df[col].apply(normalize)

    (
        df["GPT_subj"], df["GPT_pred"], df["GPT_obj"]
    ) = zip(*df["GPT_Triple_norm"].apply(split_triple_smart))
    (
        df["CBM_subj"], df["CBM_pred"], df["CBM_obj"]
    ) = zip(*df["CBM_Triple_norm"].apply(split_triple_smart))

    # Part-wise similarities
    df["sim_pred"] = df.apply(lambda r: str_sim(r["GPT_pred"], r["CBM_pred"]), axis=1)
    df["sim_subj"] = df.apply(lambda r: str_sim(r["GPT_subj"], r["CBM_subj"]), axis=1)
    df["sim_obj"] = df.apply(lambda r: str_sim(r["GPT_obj"], r["CBM_obj"]), axis=1)

    # Classify
    df["ErrorType"] = df.apply(lambda r: classify_error_row(r, threshold), axis=1)

    # Summaries
    counts = Counter(df["ErrorType"])
    # counts over all pairs
    df_summary_all = (
        pd.DataFrame.from_dict(counts, orient="index", columns=["Count"])
        .assign(Percent=lambda x: (x["Count"] / total * 100.0).round(2))
        .sort_values("Count", ascending=False)
        .rename_axis("ErrorType")
        .reset_index()
    )

    print("\n=== Error typology (all pairs) ===")
    print(df_summary_all.to_string(index=False))

    # Optional: if System Decision exists, show near-threshold breakdown
    if "System Decision" in df.columns:
        near_decision = (
            df_near["System Decision"]
            .value_counts(dropna=False)
            .rename_axis("System Decision")
            .reset_index(name="Count")
        )
        near_decision["Percent"] = (near_decision["Count"] / max(near_n, 1) * 100.0).round(2)
        print("\n=== Near-threshold by System Decision ===")
        print(near_decision.to_string(index=False))

    # Save detailed and summary outputs
    out_summary = os.path.join(outdir, "ErrorTypology_Summary.xlsx")
    with pd.ExcelWriter(out_summary, engine="xlsxwriter") as xw:
        df_summary_all.to_excel(xw, index=False, sheet_name="AllPairs")
    print(f"\nSaved error typology summary to: {out_summary}")

    out_detailed = os.path.join(outdir, "ErrorTypology_Detailed.xlsx")
    df.to_excel(out_detailed, index=False)
    print(f"Saved detailed annotated pairs to: {out_detailed}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Near-threshold (±delta) and error typology analysis for GPT–CBM pairwise similarity report."
    )
    ap.add_argument("--report", required=True, help="Path to pairwise comparison report (.xlsx)")
    ap.add_argument("--threshold", type=float, default=0.85, help="Cosine-similarity cutoff (default: 0.85)")
    ap.add_argument("--delta", type=float, default=0.05, help="Half-width for near-threshold range (default: 0.05)")
    ap.add_argument("--outdir", default="data/gold_standard_comparison", help="Output directory")
    args = ap.parse_args()

    near_threshold_and_errors(
        report_path=args.report,
        threshold=args.threshold,
        delta=args.delta,
        outdir=args.outdir,
    )

# python src/NearThreshold_and_ErrorTypology.py --report data/gold_standard_comparison/CBM_comparison_Report_Threshold_85.xlsx --threshold 0.85 --delta 0.05 --outdir data/gold_standard_comparison
