# Identifying Mechanistic Links Between COVID-19 and Neurodegenerative Diseases

**Project Goal:**  
This research project aims to uncover mechanistic connections between **COVID-19** and **neurodegenerative diseases** by extracting structured information from **biomedical figures and graphical abstracts** using advanced natural language processing (NLP) and image analysis techniques.

---

## Project Structure (Scripts Overview)

| Script | Description |
|--------|-------------|
| `Gold_Standard_Comparison.py` | Compares GPT-extracted triples against a manually curated gold standard using a Combined Similarity Score (CSS) that integrates semantic and structural accuracy. |
| `Hyperparameters_and_Prompt_Assessment.py` | Evaluates different GPT prompt templates and hyperparameter configurations (temperature, top-p) to optimize extraction quality. Generates a CSS performance plot. |
| `Image_Enrichment_Analysis.py` | Automates large-scale biomedical image scraping from Google Images and removes near-duplicates via perceptual hashing and resolution filtering. |
| `Triples_Categorization.py` | See detailed explanation below ↓ |
| `Triple_Extraction_GPT4o.py` | Extracts structured semantic triples from biomedical images using GPT-4o, following strict prompt formats for consistency and downstream integration. |
| `URLs_Relevance_Check.py` | Filters biomedical images based on relevance to COVID-19 and neurodegenerative diseases using GPT-4o. Includes accessibility check and result export. |
| `GPT4o_uncategorized_handling.py` | Assigns categories to previously Uncategorized triples using GPT-4o based on their pathophysiological process (PP). |

---

## Pathophysiological Process Classification using BERT + MeSH

**Script:** `Triples_Categorization.py`  
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
This file contains the full set of descriptors from the U.S. National Library of Medicine's MeSH thesaurus. It is used to expand each category with synonyms and biomedical variants to improve semantic coverage.

Due to its size, the file is **not included in this repository**. You must download it manually from the official source:

[Download desc2025.xml](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml)

Place it in the following directory:
```
/data/MeSh_data/desc2025.xml
```

---
## Fallback Categorization using GPT-4o

**Script:** `GPT4o_uncategorized_handling.py`  
To resolve previously *Uncategorized* triples (due to ambiguous or rare phrasing), we use GPT-4o as a semantic fallback. For each triple where the BERT + MeSH classifier fails, we send the **Pathophysiological Process** (PP) to GPT-4o with a structured prompt asking it to assign one of the six defined categories.

This fallback system outputs:
- `Category_GPT`: The category predicted by GPT-4o
- `Final_Category`: A merged field that keeps BERT's prediction (if available) or uses GPT's if not

This hybrid approach improves recall and ensures a complete dataset for downstream graph-based or statistical analyses.

### How to Run:
```bash
python src/GPT4o_uncategorized_handling.py \
  --input data/triples_output/Triples_Final_All_Relevant_Categorized.xlsx \
  --output data/triples_output/Triples_Final_All_Relevant_Categorized_GPT4o \
  --api_key YOUR_API_KEY

---

## GPT API Usage
This project relies heavily on **OpenAI GPT-4o** for multimodal processing of biomedical images. You must have valid API access to use the scripts for:
- Relevance classification
- Triple extraction

Ensure that you:
1. Have an OpenAI account with GPT-4 API access.
2. Store your API key as an environment variable:
```bash
export OPENAI_API_KEY=sk-...
```
3. Or provide it as a command-line argument when running scripts.

⚠️ **Note**: Due to API call limits and costs, full pipeline execution may require batching or quota management.

---

## Results & Outputs
All major outputs are saved as `.xlsx` and `.csv` files under the `/data/` or `/triples_output/` directories:
- `Relevant_URLs_only_GPT_4o.xlsx` – Final set of filtered relevant image URLs.
- `Triples_Final_All_Relevant.csv/.xlsx` – Semantic triples extracted from figures.
- `Triples_Final_All_Relevant_Categorized.xlsx/.csv` – Categorized mechanisms with BERT + MeSH.
- `Triples_Final_All_Relevant_Categorized_GPT4o.xlsx/.csv` – Same as above but with previously uncategorized items resolved using GPT-4o.
- `Supplementary_material_Table_1.xlsx` – Prompt and hyperparameter evaluations.
- Evaluation metrics (CSS scores) from gold standard comparisons.

---

## Requirements
- Python 3.8+
- `sentence-transformers`
- `torch`
- `openai`
- `pandas`
- `numpy`
- `requests`
- `selenium`
- `imagehash`
- `openpyxl`

---

## How to Run (Example)
```bash
# 1. Extract images from Google
python src/Image_Enrichment_Analysis.py --query "Covid-19 and Neurodegeneration" --main 100 --similar 100 --output_raw Enrichment_Search_URLs --output_clean Enrichment_Cleaned --outdir ./data

# 2. Check URL accessibility and relevance
python src/URLs_Relevance_Check.py --input data/Enrichment_Cleaned.xlsx

# 3. Extract semantic triples
python src/Triple_Extraction_GPT4o.py --input data/Relevant_URLs_only_GPT_4o.xlsx --output_dir ./triples_output

# 4. Categorize pathophysiological processes
python src/Triples_Categorization.py --input triples_output/Triples_Final_All_Relevant.csv --output triples_output/Triples_Final_All_Relevant_Categorized --mode pp

# 5. Resolve uncategorized entries using GPT-4o
python src/GPT4o_uncategorized_handling.py --input triples_output/Triples_Final_All_Relevant_Categorized.xlsx --output triples_output/Triples_Final_All_Relevant_Categorized_GPT4o

```

---

## Data and Code Availability
All source code and scripts are available in this repository:  
🔗 [GitHub Repository Link — TBA upon publication]

The annotated data, intermediate outputs, and supplementary results can be found under:
```
/data/
/triples_output/
```

For reproducibility, the pipeline uses open-source tools and publicly available biomedical datasets.

---

## License
This repository is licensed under the **MIT License**.

---

## Citation
To cite this work, please reference the corresponding article once published. A citation in BibTeX and DOI will be provided here.

---

For questions or contributions, contact:
**Elizaveta Popova** (University of Bonn) elizaveta.popova@uni-bonn.de
**Negin Sadat Babaiha** (Fraunhofer SCAI) negin.babaiha@scai.fraunhofer.de 

---

