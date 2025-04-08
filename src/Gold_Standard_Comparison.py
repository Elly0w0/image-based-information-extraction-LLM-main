# === Import required libraries ===
import pandas as pd  
from sentence_transformers import SentenceTransformer, util  # For embedding triples and calculating similarity
import numpy as np  
import matplotlib.pyplot as plt  

"""
Semantic Triple Gold Standard Comparison
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 27/03/2025
Description:
    This script evaluates the semantic similarity of subject-object triples extracted from biomedical images.
    It compares automatically extracted triples with a manually curated gold standard to assess extraction quality.

    Focus:
    - Only compares **Subject–Object** pairs (Predicate is excluded).
    - Uses SentenceTransformer embeddings to compute semantic similarity.
    - Calculates Combined Similarity Score (CSS) based on similarity and F1-score.
    - Skips images that are missing in either dataset to ensure fair evaluation.

    Input:
        - Two Excel files containing extracted triples:
            1. Gold standard file (with column 'Image_number')
            2. Evaluation file (same format)

    Output:
        - Console output of the average CSS across evaluated images.

    Requirements:
        - sentence-transformers
        - pandas
        - numpy

    Usage:
        - Run the script in an environment with the required libraries installed.
        - Ensure both Excel files are in the correct directory.
"""

# === Load Excel Data ===
file_Gold_path = "./data/GoldSt_comparison/Triples_CBM_Gold_Standard.xlsx"  # Path to the gold standard triples
file_Eval_path = "./data/GoldSt_comparison/Triples_GPT_for_comparison.xlsx"  # Path to the evaluated/generated triples

df_Gold = pd.read_excel(file_Gold_path)  
df_Eval = pd.read_excel(file_Eval_path)  

# === Load Pre-trained SentenceTransformer Model ===
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_subject_object_dict(df):
    """
    Converts the DataFrame into a dictionary mapping each image (Image_number)
    to a list of subject-object strings. Predicates are ignored.

    Args:
        df (pd.DataFrame): DataFrame containing 'Subject', 'Object', and 'Image_number' columns.

    Returns:
        dict: {Image_number: [list of 'Subject Object' strings]}
    """
    triple_dict = {}
    key_col = 'Image_number'  # Column used to group triples by image

    for _, row in df.iterrows():
        key = row[key_col]  # Get image identifier
        subject = row['Subject'].replace('_', ' ') if pd.notna(row['Subject']) else ''  # Clean subject
        # predicate = row['Predicate'].replace('_', ' ') if pd.notna(row['Predicate']) else ''
        obj = row['Object'].replace('_', ' ') if pd.notna(row['Object']) else ''  # Clean object

        # Only add triple if both subject and object are non-empty
        if subject and obj:
            triple = f"{subject} {obj}"  # Concatenate subject and object
            if key not in triple_dict:
                triple_dict[key] = []
            triple_dict[key].append(triple)  # Add triple to the list for this image

    return triple_dict

def similarity_score_subject_object(df_gold, df_extracted, image_key):
    """
    Computes the Combined Similarity Score (CSS) for a single image.
    Compares subject-object pairs from evaluated data to the gold standard.

    Args:
        df_gold (pd.DataFrame): Gold standard DataFrame
        df_extracted (pd.DataFrame): Evaluated/generated DataFrame
        image_key (str): Image_number used for filtering

    Returns:
        float or None: CSS value, or None if either set is empty
    """
    # Create dictionaries mapping image → [triples]
    gold_dict = create_subject_object_dict(df_gold)
    extracted_dict = create_subject_object_dict(df_extracted)

    # Get triples for the specific image
    gold_triples = gold_dict.get(image_key, [])
    extracted_triples = extracted_dict.get(image_key, [])

    if not gold_triples or not extracted_triples:
        return None  # If no data, skip this image

    # Encode subject-object pairs into sentence embeddings
    gold_embeds = model.encode(gold_triples, convert_to_tensor=True)
    extracted_embeds = model.encode(extracted_triples, convert_to_tensor=True)

    # Compute cosine similarity matrix between all pairs
    similarities = util.cos_sim(extracted_embeds, gold_embeds).cpu().numpy()

    SIMILARITY_THRESHOLD = 0.5  # Threshold to consider a match meaningful
    TP, FP = 0, 0  # True Positives, False Positives
    best_match_scores = []  # Best similarity score per extracted triple

    # # For each extracted triple, find the best match in gold standard
    # for i, extracted in enumerate(extracted_triples):
    #     if similarities.shape[1] > 0:
    #         best_match_idx = np.argmax(similarities[i])  # Get the index of the best match
    #         best_match_score = similarities[i][best_match_idx]  # Get the similarity score
    #         best_match_scores.append(best_match_score)

    #         # If similarity is above threshold, count as TP; else FP
    #         if best_match_score >= SIMILARITY_THRESHOLD:
    #             TP += 1
    #         else:
    #             FP += 1
    #     else:
    #         best_match_scores.append(0)

    print(f"\n Image {image_key} - Comparing Extracted vs Gold Standard Triples:")
    for i, extracted in enumerate(extracted_triples):
        if similarities.shape[1] > 0:
            best_match_idx = np.argmax(similarities[i])
            best_match_score = similarities[i][best_match_idx]
            gold_match = gold_triples[best_match_idx]
            best_match_scores.append(best_match_score)

            match_status = "✅ MATCH" if best_match_score >= SIMILARITY_THRESHOLD else "❌ NO MATCH"
            print(f"Extracted: \"{extracted}\" ↔ Gold: \"{gold_match}\" | Score: {best_match_score:.4f} → {match_status}")

            if best_match_score >= SIMILARITY_THRESHOLD:
                TP += 1
            else:
                FP += 1
        else:
            best_match_scores.append(0)
            print(f"Extracted: \"{extracted}\" ↔ Gold: [NO MATCH FOUND] → Score: 0.000 ❌")


    # Count false negatives: gold triples that had no match
    FN = max(0, len(gold_triples) - TP)

    # Compute precision, recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    average_similarity = np.mean(best_match_scores) if best_match_scores else 0

    # Compute Combined Similarity Score (CSS)
    CSS = 0.6 * average_similarity + 0.4 * f1_score
    return CSS

def compare_all_images(df_gold, df_extracted):
    """
    Computes the average CSS over all images that exist in both datasets.

    Args:
        df_gold (pd.DataFrame): Gold standard DataFrame
        df_extracted (pd.DataFrame): Evaluated/generated DataFrame

    Returns:
        float: Average CSS across all common images
    """
    # Identify common image keys in both datasets
    image_keys_gold = set(df_gold['Image_number'].unique())
    image_keys_eval = set(df_extracted['Image_number'].unique())
    common_image_keys = image_keys_gold.intersection(image_keys_eval)

    css_scores = []

        # Natural sort based on image number
    def sort_key(key):
        return int(key.split('_')[-1])

    for key in sorted(common_image_keys, key=sort_key):
        css = similarity_score_subject_object(df_Gold, df_Eval, key)
        css = similarity_score_subject_object(df_gold, df_extracted, key)
        if css is not None:
            css_scores.append(css)

    # Average CSS, excluding images with no comparisons
    average_css = sum(css_scores) / len(css_scores) if css_scores else 0
    return average_css

# === Main Execution ===
if __name__ == "__main__":
    avg_css = compare_all_images(df_Gold, df_Eval)  # Compare the two datasets
    print(f"Average Combined Similarity Score (CSS) across evaluated images: {avg_css:.4f}")
 
