import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Hyperparameters and Prompt Assessment Script
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 15/02/2025
Description:
    This script evaluates the **semantic similarity** of extracted triples from images using OpenAI's GPT model.
    It performs **two main analyses**:
    
    1. **Prompt Performance Assessment**  
       - Compares different prompt-based extractions (`Prompt_1`, `Prompt_2`, `Prompt_3`) with a **manual gold standard**.  
       - Computes **Combined Similarity Scores (CSS)** across multiple images.  
       - Provides insights into **prompt effectiveness** for extracting meaningful triples.  

    2. **Hyperparameter Impact Analysis**  
       - Analyzes how **temperature** and **top-p** settings affect extractions.  
       - Compares `Prompt_1` runs with different parameter values (0.0, 0.25, 0.5, 0.75).  
       - Identifies the **optimal hyperparameter settings** for stable and meaningful triple extraction.  

    **Key functionalities include:**  
    - **Cosine similarity computation** using a **BERT-based model (`all-MiniLM-L6-v2`)**.  
    - **Precision, Recall, F1-score, and Accuracy calculations** for evaluating triple matches.  
    - **Comparison of extracted triples against a gold standard** using similarity metrics.  
    - **Graphical visualization of hyperparameter impact** on **CSS performance**.

    **Input:**
        - Excel file (`Supplementary_material_Table_1.xlsx`) containing extracted triples from different prompts and hyperparameter settings.

    **Output:**
        - Console output with similarity scores and evaluation metrics.
        - A **graph showing the impact of hyperparameters on extraction quality**.

    **Requirements:**
        - **Sentence Transformers** for embedding generation.
        - **Pandas** for data handling.
        - **Matplotlib & Seaborn** for visualization.
        - **NumPy** for numerical operations.

    **Usage:**
        - Ensure the Excel file (`Supplementary_material_Table_1.xlsx`) is in the correct location.
        - Run this script in an environment with all dependencies installed.
        - Review the similarity scores and plots to assess prompt performance and hyperparameter effects.
"""


# Load the data
path = "./data/Supplementary_material_Table_1.xlsx"


def create_triples_dict(df):
    """
    Create a dictionary gathering triples for each URL.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'URL', 'Subject', 'Predicate', 'Object' columns.
    
    Returns:
        dict: Dictionary {URL: [list of triples]}
    """
    triples_dict = {}

    for _, row in df.iterrows():
        url = row['URL']
        subject = row['Subject'].replace('_', ' ') if pd.notna(row['Subject']) else ''
        predicate = row['Predicate'].replace('_', ' ') if pd.notna(row['Predicate']) else ''
        obj = row['Object'].replace('_', ' ') if pd.notna(row['Object']) else ''

        if subject and predicate and obj:  # Ensuring all parts exist
            triple = f"{subject} {predicate} {obj}"

            if url not in triples_dict:
                triples_dict[url] = []
            
            triples_dict[url].append(triple)

    return triples_dict

def compute_best_match_scores(extracted, gold_standard):
    """
    Compute cosine similarity scores between extracted triples and gold standard triples.

    Args:
        extracted (list of str): List of extracted triples.
        gold_standard (list of str): List of gold standard triples.

    Returns:
        np.ndarray: Matrix of cosine similarity scores between extracted and gold standard triples.
    """
    extracted_embeddings = model.encode(extracted, convert_to_tensor=True)
    gold_standard_embeddings = model.encode(gold_standard, convert_to_tensor=True)
    similarities = util.cos_sim(extracted_embeddings, gold_standard_embeddings).cpu().numpy()
    return similarities

def similarity_score(df_gold_standart, df_2, URL):
    """
    Computes similarity scores between extracted triples and gold standard triples.
    Handles different numbers of triples robustly and calculates key performance metrics.

    Args:
        df_gold_standart (pd.DataFrame): DataFrame containing gold standard triples.
        df_2 (pd.DataFrame): DataFrame containing extracted triples.
        URL (str): URL key to retrieve the triples for comparison.

    Returns:
        float: Final Combined Similarity Score (CSS), incorporating both semantic similarity and F1-score.
    """
    # Load pre-trained BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract triples for the given URL
    extracted_triples = create_triples_dict(df_2).get(URL, [])
    gold_standard_triples = create_triples_dict(df_gold_standart).get(URL, [])

    # Check for empty lists
    if not extracted_triples:
        print(f"Warning: No extracted triples found for URL: {URL}")
        return 0
    if not gold_standard_triples:
        print(f"Warning: No gold standard triples found for URL: {URL}")
        return 0

    # Encode triples into embeddings
    extracted_embeddings = model.encode(extracted_triples, convert_to_tensor=True)
    gold_standard_embeddings = model.encode(gold_standard_triples, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(extracted_embeddings, gold_standard_embeddings).cpu().numpy()

    # Define a similarity threshold for a valid match
    SIMILARITY_THRESHOLD = 0.6  

    # Initialize counts for evaluation metrics
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = max(0, len(gold_standard_triples) - TP)  # False Negatives
    best_match_scores = []  # Store best match similarity scores

    # Iterate through extracted triples and find the best match in the gold standard
    for i, extracted in enumerate(extracted_triples):
        if similarities.shape[1] > 0:
            best_match_idx = np.argmax(similarities[i])  # Find the best match index

            # Ensure the index is within valid bounds
            if best_match_idx >= len(gold_standard_triples):
                print(f"Warning: Best match index {best_match_idx} is out of range. Adjusting.")
                best_match_idx = len(gold_standard_triples) - 1

            best_match_score = similarities[i][best_match_idx]  # Get highest similarity score
            best_match_scores.append(best_match_score)  # Store the score

            # print(f"Extracted Triple: {extracted}")
            # print(f"Best Match in Gold Standard: {gold_standard_triples[best_match_idx]}")
            # print(f"Similarity Score: {best_match_score:.4f}\n")

            # Compute TP & FP
            if best_match_score >= SIMILARITY_THRESHOLD:
                TP += 1  # Correct match
            else:
                FP += 1  # Incorrect match
        else:
            best_match_scores.append(0)

    # Compute False Negatives (FN)
    FN = max(0, len(gold_standard_triples) - TP)

    # Compute evaluation metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    # Compute the average similarity
    average_similarity = np.mean(best_match_scores) if best_match_scores else 0

    # Compute Combined Similarity Score (CSS)
    λ1 = 0.6  # Weight for Average Similarity
    λ2 = 0.4  # Weight for F1-score
    css = (λ1 * average_similarity) + (λ2 * f1_score)

    # print(f"Average Similarity: {average_similarity:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1-score: {f1_score:.4f}")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Final Combined Similarity Score (CSS): {css:.4f}")

    # # Confusion Matrix Visualization
    # labels = ["Match", "No Match"]
    # cm = np.array([[TP, FP], [FN, 0]])  # No True Negatives in this setting
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.title("Confusion Matrix")
    # plt.show()

    # Precision-Recall Curve Calculation
    thresholds = np.linspace(0, 1, 50)
    precisions, recalls = [], []

    for thresh in thresholds:
        tp_temp = sum(1 for score in best_match_scores if score >= thresh)
        fp_temp = len(best_match_scores) - tp_temp
        fn_temp = max(0, len(gold_standard_triples) - tp_temp)

        precision_temp = tp_temp / (tp_temp + fp_temp) if (tp_temp + fp_temp) > 0 else 0
        recall_temp = tp_temp / (tp_temp + fn_temp) if (tp_temp + fn_temp) > 0 else 0

        precisions.append(precision_temp)
        recalls.append(recall_temp)

    # # Plot Precision-Recall Curve
    # plt.figure(figsize=(6, 5))
    # plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("Precision-Recall Curve")
    # plt.grid(True)
    # plt.show()

    return css

def similarity_across_images(df_gold_standart, df_2):
    """
    Computes the average Combined Similarity Score (CSS) across multiple images.

    Args:
        df_gold_standart (pd.DataFrame): DataFrame containing gold standard triples.
        df_2 (pd.DataFrame): DataFrame containing extracted triples.

    Returns:
        float: Average CSS across all image URLs.
    """
    # Define image URLs to compare
    image_urls = df_gold_standart['URL'].unique()

    css_scores = []
    for url in image_urls:
        css = similarity_score(df_gold_standart, df_2, url)
        if css is not None:  # Ensure valid CSS scores are added
            css_scores.append(css)

    # Compute average CSS while avoiding division by zero
    average_css = sum(css_scores) / len(css_scores) if css_scores else 0

    return average_css

def plot_hyperparameters(scores_list):
    """
    Plots the impact of hyperparameter settings (Temperature = Top_P) on Average CSS.

    Args:
        scores_list (list of float): List of average CSS values corresponding to different hyperparameter settings.

    Returns:
        None: Displays the plot.
    """
    # Define hyperparameter values
    hyperparameters = [0.0, 0.25, 0.5, 0.75]

    # Validate input length
    if len(scores_list) != len(hyperparameters):
        raise ValueError("Length of scores_list must match the number of hyperparameter settings.")

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(hyperparameters, scores_list, marker='o', linestyle='-', linewidth=2, color='b')

    # Labels and title
    plt.xlabel("Hyperparameter Setting (Temperature = Top_P)", fontsize=12)
    plt.ylabel("Average CSS (Compared to Gold Standard)", fontsize=12)
    plt.title("Impact of Hyperparameters on Average CSS", fontsize=14)

    # Add value labels on points
    for i, txt in enumerate(scores_list):
        plt.annotate(f"{txt:.3f}", (hyperparameters[i], scores_list[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Grid and limits
    plt.ylim(min(scores_list) - 0.02, max(scores_list) + 0.02)  # Dynamic y-axis limits
    plt.xticks(hyperparameters)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Show plot
    plt.show()


if __name__ == "__main__":

    # Load the data
    # Prompt Performance Assessment
    manual_data = pd.read_excel(path, sheet_name='Manual')
    prompt_1_data = pd.read_excel(path, sheet_name='Prompt_1')
    prompt_2_data = pd.read_excel(path, sheet_name='Prompt_2')
    prompt_3_data = pd.read_excel(path, sheet_name='Prompt_3')
    # Hyperparameters Analysis
    data_run_0_0 = pd.read_excel(path, sheet_name='Prompt_1_Parameters_0.0')
    data_run_0_25 = pd.read_excel(path, sheet_name='Prompt_1_Parameters_0.25')
    data_run_0_5 = pd.read_excel(path, sheet_name='Prompt_1_Parameters_0.5')
    data_run_0_75 = pd.read_excel(path, sheet_name='Prompt_1_Parameters_0.75')

    print("Prompt Performance Assessment")
    average_css_prompt_1 = similarity_across_images(manual_data, prompt_1_data)
    average_css_prompt_2 = similarity_across_images(manual_data, prompt_2_data)
    average_css_prompt_3 = similarity_across_images(manual_data, prompt_3_data)

    # Print the mean scores for prompt analysis
    print(f"Average CSS across images (Manual vs. Prompt_1): {average_css_prompt_1:.2f}")
    print(f"Average CSS across images (Manual vs. Prompt_2): {average_css_prompt_2:.2f}")
    print(f"Average CSS across images (Manual vs. Prompt_3): {average_css_prompt_3:.2f}")

    print("Hyperparameters Analysis")
    average_css_first_00 = similarity_across_images(manual_data, data_run_0_0)
    average_css_first_025 = similarity_across_images(manual_data, data_run_0_25)
    average_css_first_05 = similarity_across_images(manual_data, data_run_0_5)
    average_css_first_075 = similarity_across_images(manual_data, data_run_0_75)

    # Print the mean scores for hyperparameters assessment
    print(f"Average CSS across images (Manual vs. Prompt_1 (temperature = 0.0; top_p = 0.0)): {average_css_first_00:.3f}")
    print(f"Average CSS across images (Manual vs. Prompt_1 (temperature = 0.25; top_p = 0.25)): {average_css_first_025:.3f}")
    print(f"Average CSS across images (Manual vs. Prompt_1 (temperature = 0.5; top_p = 0.5)): {average_css_first_05:.3f}")
    print(f"Average CSS across images (Manual vs. Prompt_1 (temperature = 0.75; top_p = 0.75)): {average_css_first_075:.3f}")

    # Plot the Impact of Hyperparameters on Average CSS
    average_css_scores = [average_css_first_00, average_css_first_025, average_css_first_05, average_css_first_075]
    plot_hyperparameters(average_css_scores)
    

    
