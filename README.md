# Identifying Mechanistic Links Between COVID-19 and Neurodegenerative Diseases

**Project Goal:**  
This research project aims to uncover mechanistic connections between **COVID-19** and **neurodegenerative diseases** by extracting structured information from **biomedical figures and graphical abstracts** using advanced natural language processing (NLP) and image analysis techniques.

---

## Project Structure (Scripts Overview)

| Script | Description |
|--------|-------------|
| `Gold_Standard_Comparison.py` | *TBD* |
| `Hyperparameters_and_Prompt_Assessment.py` | *TBD* |
| `Image_Enrichment_Analysis.py` | *TBD* |
| `Pathophysiological_Pr_Classification_MeSH.py` | See below ↓ |
| `Triple_Extraction_GPT4o.py` | *TBD* |
| `URLs_Relevance_Check.py` | *TBD* |

---

## Pathophysiological Process Classification using BERT + MeSH

**Script:** `Pathophysiological_Pr_Classification_MeSH.py`  
This module classifies biomedical pathophysiological processes into predefined mechanistic categories using **BERT-based semantic similarity** with **ontology-derived keywords** from the MeSH database.

### Categories Used:
- Viral Entry and Neuroinvasion  
- Immune and Inflammatory Response  
- Neurodegenerative Mechanisms  
- Vascular Effects  
- Psychological and Neurological Symptoms  
- Systemic Cross-Organ Effects

### Methodology:
- Extracts category-specific keywords from the **MeSH ontology**
- Normalizes all terms and descriptions
- Embeds both using **Sentence-BERT**
- Matches based on **cosine similarity**

### Large File Notice
This script depends on the **MeSH descriptor XML file**:  
`desc2025.xml` (299MB)  
This file is not tracked in the repository due to GitHub’s size limits.

[Download it here](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml)

Place it in the following directory:
```
/data/MeSh_data/desc2025.xml
```

---

## Results & Outputs
*To be filled after full pipeline integration.*

---

## Requirements
- Python 3.8+
- `sentence-transformers`
- `pandas`
- `torch`

---

## How to Run
```bash
python Pathophysiological_Pr_Classification_MeSH.py
```

---
