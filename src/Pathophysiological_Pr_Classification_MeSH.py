# === Import Required Libraries ===
import xml.etree.ElementTree as ET
import pandas as pd
import json
import re
from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict, Counter

"""
BERT-Based Classification of Pathophysiological Processes using MeSH Keywords
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 27/03/2025

Description:
    This script classifies biomedical pathophysiological processes into predefined categories
    using BERT-based semantic similarity. It builds keyword sets by parsing MeSH descriptors
    and matches each process against these using cosine similarity of BERT embeddings.

Categories:
    1. Viral Entry and Neuroinvasion
    2. Immune and Inflammatory Response
    3. Neurodegenerative Mechanisms
    4. Vascular Effects
    5. Psychological and Neurological Symptoms
    6. Systemic Cross-Organ Effects

Input:
    - MeSH descriptors file: desc2025.xml
    - Dataset file: Triples_Final_All_Relevant.csv

Output:
    - Categorized dataset as CSV and Excel files
    - Category counts per process

Requirements:
    - sentence-transformers
    - pandas
    - torch
"""

# === Utility Functions ===

def normalize_text(text):
    """
    Normalize text by replacing underscores/hyphens with spaces,
    converting to lowercase, and stripping extra whitespace.
    """
    text = re.sub(r"[_\-]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_mesh_keywords(xml_path, category_keywords):
    """
    Extracts relevant MeSH terms for each category based on seed keywords.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    category_terms = defaultdict(set)

    for descriptor in root.findall('DescriptorRecord'):
        descriptor_name_el = descriptor.find('DescriptorName/String')
        if descriptor_name_el is None:
            continue
        descriptor_name = descriptor_name_el.text
        term_elements = descriptor.findall('ConceptList/Concept/TermList/Term/String')
        concept_terms = [term_el.text for term_el in term_elements if term_el is not None]
        all_text = f"{descriptor_name} " + ' '.join(concept_terms)

        for category, keywords in category_keywords.items():
            if any(keyword.lower() in all_text.lower() for keyword in keywords):
                category_terms[category].update([descriptor_name] + concept_terms)

    return {cat: list(terms) for cat, terms in category_terms.items()}

def bert_keyword_classify(process_text, category_keyword_embeddings, model, threshold=0.5, aggregation="max"):
    """
    Classify a process description into a category using cosine similarity
    between its BERT embedding and category keyword embeddings.
    """
    process_embedding = model.encode(process_text, convert_to_tensor=True)
    best_category = None
    best_score = 0.0

    for category, keyword_embeddings in category_keyword_embeddings.items():
        if keyword_embeddings is None or len(keyword_embeddings) == 0:
            continue
        cosine_scores = util.pytorch_cos_sim(process_embedding, keyword_embeddings)[0]
        score = torch.max(cosine_scores).item() if aggregation == "max" else torch.mean(cosine_scores).item()
        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score >= threshold else "Uncategorized"

# === Main Execution ===
def main():
    # Define paths
    xml_path = './data/MeSh_data/desc2025.xml'
    data_path = './data/Triples_Final_All_Relevant.csv'
    output_csv = './data/Triples_Final_All_Relevant_Categorized_BERT_Keywords_mesh.csv'
    output_excel = './data/Triples_Final_All_Relevant_Categorized_BERT_Keywords_mesh.xlsx'
    output_json = './data/MeSh_data/mesh_category_terms.json'

    # Define seed keywords for each category
    category_keywords = {
        "Viral Entry and Neuroinvasion": [
            "neuroinvasion", "receptor", "ACE2", "blood-brain barrier", "BBB", "virus entry", "olfactory", 
            "retrograde transport", "endocytosis", "direct invasion", "cranial nerve", "neural pathway", 
            "transcribrial", "neurotropic", "trans-synaptic", "neuronal route", "olfactory nerve", 
            "hematogenous", "choroid plexus", "neuronal transmission", "entry into CNS"
        ],
        "Immune and Inflammatory Response": [
            "immune", "cytokine", "inflammation", "interferon", "TNF", "IL-6", "IL6", "cytokine storm", 
            "immune response", "inflammatory mediators", "macrophage", "microglia", "neutrophil", 
            "lymphocyte", "innate immunity", "immune dysregulation", "chemokine", "T cell", "NLRP3", 
            "antibody", "immune activation", "immune imbalance", "immune-mediated", "complement"
        ],
        "Neurodegenerative Mechanisms": [
            "neurodegeneration", "protein aggregation", "apoptosis", "cell death", "synaptic loss", 
            "neurotoxicity", "oxidative stress", "mitochondrial dysfunction", "tau", "amyloid", 
            "α-synuclein", "prion", "demyelination", "neuron loss", "misfolded proteins", 
            "chronic neuronal damage", "neurodegenerative", "neuroinflammation"
        ],
        "Vascular Effects": [
            "stroke", "thrombosis", "vascular", "ischemia", "coagulation", "blood clot", "microthrombi", 
            "endothelial", "vasculitis", "hemorrhage", "blood vessel", "vascular damage", "capillary", 
            "clotting", "hypoperfusion", "angiopathy", "vasculopathy"
        ],
        "Psychological and Neurological Symptoms": [
            "cognitive", "memory", "fatigue", "depression", "anxiety", "brain fog", "psychiatric", 
            "mood", "confusion", "neuropsychiatric", "emotional", "behavioral", "neurocognitive", 
            "insomnia", "psychosocial", "attention", "motivation", "executive function", "suicidality"
        ],
        "Systemic Cross-Organ Effects": [
            "lungs", "liver", "kidney", "systemic", "multi-organ", "gastrointestinal", "heart", 
            "cardiovascular", "endocrine", "renal", "pancreas", "organ failure", "liver damage", 
            "pulmonary", "myocardial", "respiratory", "hypoxia", "oxygen deprivation", "fibrosis"
        ]
    }

    # === Extract keywords from MeSH
    category_terms = extract_mesh_keywords(xml_path, category_keywords)

    # Save extracted keywords
    with open(output_json, "w") as f:
        json.dump(category_terms, f, indent=2)

    print('MeSH keywords are saved to the json file.')

    # === Load data
    df = pd.read_csv(data_path)
    df['Normalized_Process'] = df['Pathophysiological Process'].apply(normalize_text)

    for category in category_terms:
        category_terms[category] = [normalize_text(term) for term in category_terms[category]]

    # === BERT Embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    category_keyword_embeddings = {
        category: model.encode(terms, convert_to_tensor=True)
        for category, terms in category_terms.items() if terms
    }

    # === Classify Processes
    df['Category_BERT_Keywords'] = df['Normalized_Process'].apply(
        lambda x: bert_keyword_classify(x, category_keyword_embeddings, model, threshold=0.5, aggregation="max")
    )

    # === Output Results
    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False)

    print('Final classification files are saved.')

    category_counts = Counter(df['Category_BERT_Keywords'])
    print("=== Category Counts (BERT + Keywords) ===")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

# === Entry Point ===
if __name__ == "__main__":
    main()
