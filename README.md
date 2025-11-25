
# Leveraging Multimodal Large Language Models to Extract Mechanistic Insights from Biomedical Visuals: A Case Study on COVID-19 and Neurodegenerative Diseases 

**Authors**: Elizaveta Popova, Marc Jacobs, Martin Hofmann-Apitius, Negin Sadat Babaiha 

**Institutions**: 
- Bonn-Aachen International Center for Information Technology (b-it), University of Bonn, Bonn, Germany 
- Department of Bioinformatics, Fraunhofer Institute for Algorithms and Scientific Computing (SCAI), Schloss Birlinghoven, Sankt Augustin, Germany  

**Contact**: elizaveta.popova@uni-bonn.de, negin.babaiha@scai.fraunhofer.de  

---
![Workflow Diagram](data/workflow.png)
---
## Abstract

This project presents a computational framework to identify mechanistic connections between **COVID-19** and **neurodegenerative diseases** by extracting and analyzing **semantic triples** from biomedical figures. The pipeline combines **LLM-based triple extraction**, **BioBERT-based semantic comparison**, and **MeSH-informed categorization** to map biological processes depicted in literature into structured and interpretable knowledge.

---

## Project Overview

### Goals
- Extract biological mechanisms from graphical abstracts
- Compare GPT-generated triples to a manually curated gold standard
- Classify processes into domain-relevant pathophysiological categories

### Key Techniques
- GPT-4o (multimodal) for image-to-text triple extraction
- BioBERT for semantic embedding and similarity
- Sentence-BERT + MeSH ontology for mechanistic classification

---

## Methodology Overview

### Step 1: Image Extraction and Filtering
- Automated Google Images scraping
- Relevance filtering via GPT-4o
- Output: `Relevant_URLs_only_GPT_4o.xlsx`

### Step 2: Similarity Threshold, Prompt and Hyperparameters Evaluation

#### 2.1 Similarity Threshold Evaluation
- Optimizes the cosine similarity threshold for matching GPT-predicted triples to CBM gold-standard triples using BioBERT full-triple embeddings.
- Full triples (subject–predicate–object) are compared as normalized sentences, and the Hungarian algorithm is applied for optimal one-to-one matching.
- Generates threshold-dependent performance metrics to identify the optimal cutoff.
- A helper script (URL_subset_selector.py) can be used to randomly select a 50-image CBM subset for controlled evaluation.

#### 2.2 Prompt Assessment
- Compares GPT triples generated from different prompt formulations to the CBM gold standard using BioBERT full-triple embedding.
- Applies the Hungarian algorithm to align predicted and gold triples for each prompt version.
- Outputs per-prompt precision, recall, and F1 scores, along with comparative plots.
- Performs a paired bootstrap resampling over images (B = 10 000) to test whether differences between prompts are statistically significant (Bootstrap_Paired_Images_BioBERT.py).

#### 2.3 Hyperparameters Assessment
- Evaluates the effect of GPT decoding parameters (temperature and top_p) on triple extraction quality.
- Uses BioBERT embeddings and Hungarian algorithm matching against the CBM gold standard.
- Produces comparative metrics to identify parameter settings that maximize alignment accuracy.

### Step 3: Triple Extraction from Biomedical Images
#### 3.1 From Biomedical Images
- Extracts semantic triples (subject | predicate | object) from biomedical images depicting mechanisms linking COVID-19 and neurodegeneration using OpenAI’s GPT-4o.
- Reads image URLs from an input Excel file and sends each image to GPT-4o with strict, standardized prompts to ensure consistent triple formatting.
- Parses model responses, validates triple structure, and removes non-informative outputs.

Outputs:
- `Triples_Final_All_Relevant.csv` — All extracted triples from relevant images
- `Triples_Final_All_Relevant.xlsx` — Excel version of the same data

#### 3.2 From Full-Text Biomedical Articles
- Extracts semantic triples from paragraph-level text in biomedical articles related to COVID-19 and neurodegeneration.
- Reads full-text data from a JSON input, processes each paragraph with GPT-4o, and applies a carefully designed prompt for high-precision triple extraction.
- Validates and normalizes triples, replacing malformed or incomplete outputs with a fallback standard triple when necessary, and discarding non-informative entries.

Outputs:
- `Triples_Full_Text_GPT_for_comp.csv` — All extracted triples from article text
- `Triples_Full_Text_GPT_for_comp.xlsx` — Excel version of the same data

### Step 4: Triple Evaluation and Extended Error Analysis

- Compares full semantic triples (subject–predicate–object) by encoding them as complete sentences using BioBERT embeddings and applying the Hungarian algorithm (Gold_Standard_Comparison_BioBERT.py).
- Per-category evaluation: Computes precision, recall, and F1 per biological mechanism category (e.g., Immune Response, Vascular Effects) using Gold_Standard_Comparison_PerCategory.py.
- Near-threshold and error typology: Analyzes GPT–CBM pairs near the similarity cutoff and automatically categorizes errors into boundary, predicate, and granularity mismatches (NearThreshold_and_ErrorTypology.py).

Inputs: Gold-standard triples and GPT-extracted triples (Excel/CSV).

Outputs: similarity histograms, global and per-category metrics, and detailed per-triple comparison logs.

### Step 5: Mechanism Categorization
- Uses Sentence-BERT embeddings and MeSH keyword expansion
- Categories:
  - Viral Entry and Neuroinvasion
  - Immune and Inflammatory Response
  - Neurodegenerative Mechanisms
  - Vascular Effects
  - Psychological and Neurological Symptoms
  - Systemic Cross-Organ Effects
- Fallback: GPT-4o assigns categories for ambiguous cases

### Step 6: Ontology Normalization, Entity Typing, and Graph-Based Integration in Neo4j

Ontology-based entity linking:
- fast_entity_linking_embedding_advanced.py links each subject/object to biomedical ontologies (e.g., HGNC, UniProt, GO, DOID, MeSH, NCIt, Wikidata), using multi-strategy lexical + embedding search and per-group thresholds.
- Produces enriched triples with canonical ontology IDs, IRIs, and normalized labels.

Entity type classification:
- classify_entities.py assigns broad biomedical types (Gene/Protein, Disease/Condition, Chemical/Drug, Anatomy/Tissue/Cell Type, etc.) to each entity using heuristics plus GPT-4o, with caching for efficiency. The resulting subject_type / object_type columns are used as node labels in Neo4j.

Neo4j ingestion with BEL-style relations:

- final_neo4j_load.py uploads enriched, typed triples into Neo4j:
- Nodes are merged by stable IDs (ontology_id → IRI → internal ID → canonical name) using a unique key property and constraint.
- subject_* / object_* columns become node properties; all other metadata remain on relationships.
- Predicate texts are mapped to a compact BEL-style relation set (INCREASE, DECREASE, REGULATES, ASSOCIATION, PART_OF, etc.) while preserving original predicates.

Output: a normalized, ontology-grounded, queryable biomedical knowledge graph combining CBM, image-derived GPT, and full-text GPT triples.

---

## Repository Structure

```
SCAI-BIO/covid-NDD-image-based-information-extraction/
├── config/
│   ├── config.ini                       ← Stores Neo4j connection credentials and the OpenAI API key.
│
├── data/
│   ├── CBM_data/                        ← Data for CBM manual curation; triples extracted by CBM; full-text papers of the CBM images.
│        └── images_CBM/                 ← Images used for CBM manual triple extraction                       
│   ├── enrichment_data/                 ← URL collection using Google Image Search, URL pull cleaned
│   ├── figures_output/                  ← Figures obtained by scripts
│   ├── gold_standard_comparison/        ← Curated and predicted triples
│   ├── prompt_engineering/              ← Files related to the similarity threshold, prompt and hyperparameters evaluation
│        └── cbm_files/                  ← CBM 50 URLs subset files
│        └── gpt_files/                  ← GPT-extracted triple files for the same 50 URL subset
│        └── statistical_data/           ← Statistical data and log files accompanying plots for prompt and hyperparameters assessment
│   ├── MeSh_data/                       ← MeSH XML & category/synonym outputs
│   ├── neo4j_queries/                   ← Files containing Neo4j queries used in this work for comparisons
│   ├── bio_insights/                    ← Biological insights from the neo4j data
│        └── neo4j_results/              ← Files containing results of the neo4j queries
│        └── outputs/                    ← Figures and .csv files summarizing bioligical analysis
│   ├── baselines_and_ablations/         ← Non-LLM baseline and GPT-4o ablation experiments (image-only vs. image+caption)
│   ├── bootstrap/                       ← Paired bootstrap resampling analysis for prompt comparisons
│   ├── prompt_templates/                ← Files in .txt format containing prompts used in this work
│   ├── URL_relevance_analysis/          ← URL relevance check results (GPT-4o and manual)
│   └── triples_output/                  ← Extracted and categorized triples
│
├── src/
│   ├── Image_Enrichment_Analysis.py
│   ├── URLs_Relevance_Check.py
│   ├── Triple_Extraction_GPT4o.py
│   ├── Triple_Extraction_FullText.py
│   ├── URL_subset_selector.py
│   ├── Threshold_Selection_BioBERT.py
│   ├── Prompt_Assessment.py
│   ├── Hyperparameter_Assessment.py
│   ├── Gold_Standard_Comparison_BioBERT.py
│   ├── Gold_Standard_Comparison_PerCategory.py
│   ├── Bootstrap_Paired_Images_BioBERT.py
│   ├── NearThreshold_and_ErrorTypology.py
│   ├── mesh_category_extraction.py
│   ├── Triples_Categorization.py
│   ├── GPT4o_uncategorized_handling.py
│   ├── classify_entities.py
│   ├── fast_entity_linking_embedding_advanced.py
│   ├── final_neo4j_load.py
│   ├── review_labels_neo4j.py
│   └── bio_insights.py
```

---

## Important Files and Descriptions

| File | Description |
|------|-------------|
| **CBM_data** |
| `Data_CBM.xlsx` | Image-level metadata for CBM-annotated subset |
| `full_text_articles.json` | Parsed text file for papers from which the biomedical images for the CBM analysis were extracted |
| `Data_CBM_with_GitHub_URLs.xlsx` | Metadata for CBM images with the GitHub URLs added for the stable access by GPT-4o |
| `Triples_CBM_Gold_Standard.xlsx` | Manually curated CBM triples |
| **enrichment_data** |
| `Enrichment_Cleaned.xlsx` | The URLs from the Google Image Search preprocessed |
| `Enrichment_Search_URLs.xlsx` | The full set of gathered URLs from the Google Image Search |
| **gold_standard_comparison** |
| `Triples_GPT_for_comparison.xlsx/csv` | GPT triples for CBM subset, with image-level mapping |
| `Triples_GPT_for_comparison_SubjObj_Categorized.xlsx/csv` | Same triples, with MeSH-based subject/object category |
| `Triples_CBM_Gold_Standard.xlsx` | Manually curated CBM triples |
| `Triples_CBM_Gold_Standard_cleaned.csv` | Manually curated CBM triples cleaned for neo4j upload |
| `Triples_CBM_Gold_Standard_SubjObj_Categorized.xlsx/csv` | CBM triples with subject/object category labels |
| `Triples_Full_Text_GPT_for_comp_with_URLs.xlsx/csv` | GPT-extracted triples from full-text papers with corresponding URLs|
| `Triples_Full_Text_GPT_for_comp.xlsx/csv` | GPT-extracted triples from full-text papers |
| `Triples_Full_Text_GPT_for_comp_cleaned.csv` | GPT-extracted triples from full-text papers cleaned for neo4j upload |
| `CBM_comparison_Report_Threshold_85.xlsx` | Triple comparison log for CBM and GPT-extracted triples from images |
| **MeSh_data** |
| `mesh_category_terms.json` | MeSH category → keyword dictionary |
| **prompt_engineering/cbm_files** |
| `CBM_subset_50_URL_triples.xlsx` | 	Gold-standard triples for a randomly selected subset of 50 CBM images (used in evaluation and prompt testing) |
| `CBM_subset_50_URLs.xlsx` | List of 50 CBM image URLs randomly selected for subset-based evaluation. |
| **prompt_engineering/gpt_files** |
| `GPT_subset_triples_prompt1_param0_0.xlsx/csv` | GPT-extracted triples for the 50-image CBM subset, generated with Prompt 1 and decoding parameters temperature=0, top_p=0 |
| **prompt_engineering/statistical_data** |
| `Hyperparameter_Assessment.xlsx` | Performance metrics (precision, recall, F1) comparing GPT triples generated under different decoding hyperparameters |
| `Prompt_Comparison.xlsx` | Evaluation results comparing different GPT prompt formulations for triple extraction |
| `Similarity_Threshold_Report.xlsx` | Log of the triples compared for assessment of the optimal triple matching accuracy |
| **triples_output** |
| `Triples_Final_All_Relevant_Categorized.xlsx/csv` | Categorized GPT triples via BERT + MeSH keywords |
| `Triples_Final_All_Relevant_Categorized_GPT4o.xlsx/csv` | Final categorized triples with GPT fallback |
| `Triples_Final_All_Relevant.csv/xlsx` | All semantic triples extracted from the full image pool using GPT-4o |
| `Triples_Final_comparison_with_CBM.csv/xlsx` | GPT triples for subset of images annotated by CBM |
| **URL_relevance_analysis** |
| `Comparison_GPT_Manual_Relevance.xlsx` | Manual evaluation of GPT-extracted captions |
| `Relevant_URLs_only_GPT_4o.xlsx` | Final image set deemed relevant via GPT-4o |
| `Final_Relevant_URLs.xlsx` | The final list of the relevant URLs |
| `Relevance_assignment_GPT_4o.xlsx` | The full URL list with relevance labels assigned by GPT-4o |
| `URL_relevance_final_manual_check.xlsx` | Manual relevance assessment of the URLs considered relevant by GPT-4o |
| **bio_insights** |
| `bio_insights_with_neo4j_queries.docx` | Biological findings, figures and neo4j queries used |
| **data** |
| `Supplementary_material_S4_Table.xlsx` | Results from GPT prompt & hyperparameter tuning |
| `neo4j.dump` | The final graph to be open in neo4j |

---

## How to Run (Pipeline)

```bash
# Step 1: Image enrichment and duplicate removal
python src/Image_Enrichment_Analysis.py \
    --query "Covid-19 and Neurodegeneration" \
    --main 100 \
    --similar 100 \
    --output_raw Enrichment_Search_URLs \
    --output_clean Enrichment_Cleaned \
    --outdir ./data/enrichment_data

# Step 2: Relevance classification of images (GPT-based)
python src/URLs_Relevance_Check.py \
    --input data/enrichment_data/Enrichment_Search_URLs.xlsx \
    --api_key YOUR_API_KEY

# Step 3: Triple extraction from biomedical images
python src/Triple_Extraction_GPT4o.py \
    --input data/URL_relevance_analysis/Relevant_URLs_only_GPT_4o.xlsx \
    --output data/triples_output/Triples_Final_All_Relevant \
    --api_key YOUR_API_KEY

# Step 4: Triple extraction from full-text biomedical articles
python src/Triple_Extraction_FullText.py \
    --input data/CBM_data/full_text_articles.json \
    --output_dir ./data/gold_standard_comparison \
    --api_key YOUR_API_KEY

# Step 5 (Optional): Select a 50-image CBM subset for evaluation
python src/URL_subset_selector.py

# Step 6 (Optional): Evaluate optimal BioBERT similarity thresholds
python src/Threshold_Selection_BioBERT.py \
    --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard.xlsx \
    --eval data/gold_standard_comparison/Triples_GPT_for_comparison.xlsx

# Step 7 (Optional): Assess effect of different prompts on triple quality
python src/Prompt_Assessment.py

# Step 8 (Optional): Paired bootstrap test for prompt differences
python src/Bootstrap_Paired_Images_BioBERT.py \
  --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard.xlsx \
  --eval data/gold_standard_comparison/Triples_GPT_freeform.xlsx --label FreeForm \
  --eval data/gold_standard_comparison/Triples_GPT_template.xlsx --label Template \
  --eval data/gold_standard_comparison/Triples_GPT_hybrid.xlsx --label Hybrid \
  --threshold 0.85 --B 10000 --seed 42 --outdir results/bootstrap --save_raw

# Step 9 (Optional): Assess effect of GPT decoding hyperparameters
python src/Hyperparameter_Assessment.py

# Step 10: Compare GPT triples to gold standard using BioBERT + Hungarian matching
python src/Gold_Standard_Comparison_BioBERT.py \
    --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard.xlsx \
    --eval data/gold_standard_comparison/Triples_GPT_for_comparison.xlsx \
    --threshold 0.85

# Step 11 (Optional): Near-threshold analysis and error typology
python src/NearThreshold_and_ErrorTypology.py \
        --report data/gold_standard_comparison/CBM_comparison_Report_Threshold_85.xlsx \
        --threshold 0.85 --delta 0.05 \
        --outdir data/gold_standard_comparison

# Step 12: Extract MeSH category terms from official descriptor file
python src/mesh_category_extraction.py \
    --mesh_xml data/MeSh_data/desc2025.xml \
    --output data/MeSh_data/mesh_category_terms.json

# Step 13: Categorize triples into mechanistic classes (BERT + MeSH keywords)
python src/Triples_Categorization.py \
    --input data/triples_output/Triples_Final_All_Relevant.csv \
    --output data/triples_output/Triples_Final_All_Relevant_Categorized \
    --mode pp

# Step 14: Assign categories to uncategorized entries using GPT-4o
python src/GPT4o_uncategorized_handling.py \
    --input data/triples_output/Triples_Final_All_Relevant_Categorized.xlsx \
    --output data/triples_output/Triples_Final_All_Relevant_Categorized_GPT4o \
    --api_key YOUR_API_KEY

# Step 15 (Optional): Per-category comparison (by biological domain)
python src/Gold_Standard_Comparison_PerCategory.py \
    --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard_SubjObj_Categorized.xlsx \
    --eval data/gold_standard_comparison/Triples_GPT_for_comparison_SubjObj_Categorized.xlsx \
    --threshold 0.85

# Step 16: Ontology-based entity linking for all triples
python fast_entity_linking_embedding_advanced.py \
        --input ../data/gold_standard_comparison/Triples_Full_Text_GPT_for_comp_with_URLs.csv \
        --subject-col "Subject" \
        --object-col "Object" \
        --outdir ../data/gold_standard_comparison/output_Triples_Full_Text_GPT_for_comp_with_URLs \
        --use-embeddings \
        --embed-model text-embedding-3-large \
        --threshold 0.55 \
        --max-workers 16 \
        --fast

# Step 17: Broad entity typing for Neo4j node labels
python src/classify_entities.py \
    data/neo4j_data/triples_enriched/gpt-full.csv

# Step 18: Upload image- and full-text-derived triples to Neo4j
python final_neo4j_load.py \
        --file ./classified_triples.xlsx \
        --uri bolt://localhost:7687 --user neo4j --password YOUR_PASSWORD

# Step 19 (Optional): Review and correct Neo4j node labels using GPT
python src/review_labels_neo4j.py

# Step 20: Get the biological insights from the neo4j data 
python python src/bio_insights.py \
        --input-dir data/bio_insights/neo4j_results \
        --output-dir data/bio_insights/outputs \
        --top-n 20 \
        --run-neo4j \
        --neo4j-uri neo4j://127.0.0.1:7687 \
        --neo4j-user neo4j \
        --neo4j-password YOUR_PASSWORD \
        --neo4j-db neo4j
```

---

## MeSH Integration

This project uses:
- `desc2025.xml`: official MeSH descriptor file from NLM
- Keyword-based category matching (`mesh_category_terms.json`)
- Entity-level synonym normalization (`mesh_triples_synonyms.json`)

> [Download desc2025.xml](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml)  
Place in: `/data/MeSh_data/desc2025.xml`

---

## Neo4j Integration

The script `final_neo4j_load.py` uploads all extracted and/or curated semantic triples into a local Neo4j graph database using the **Bolt** protocol.

- Each node (Subject/Object) is labeled based on ontology categories (e.g., Gene, Disease, Biological_Process)
- Relationships retain metadata (image URL, pathophysiological process, source)
- External APIs (HGNC, MeSH, GO, DOID, ChEMBL) are used to classify and normalize terms

Neo4j Setup:

- Install Neo4j Desktop
- Create and start a local database (default port: bolt://localhost:7687)
- Username: neo4j
- Provide the password when prompted

You can visualize the resulting biomedical knowledge graph using Neo4j's Explore interface.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key packages:
- `openai`
- `torch`
- `pandas`, `numpy`
- `sentence-transformers`
- `transformers`
- `scipy`
- `openpyxl`, `matplotlib`, `imagehash`, `selenium`

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

## Citation

> Please cite this work as:


---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

- **Elizaveta Popova**  
  University of Bonn  
  elizaveta.popova@uni-bonn.de  

- **Negin Sadat Babaiha**  
  Fraunhofer SCAI  
  negin.babaiha@scai.fraunhofer.de  

---

## Funding

This research was supported by the Bonn-Aachen International Center for Information Technology (b-it) foundation, Bonn, Germany, and Fraunhofer Institute for Algorithms and Scientific Computing (SCAI). Additional financial support was provided through the COMMUTE project, which receives funding from the European Union under Grant Agreement No. 101136957. 

---

## Reproducibility Checklist

| Requirement | Status |
|------------|--------|
| Open-source code | Available |
| Data files (intermediate and gold standard) | Included |
| Prompt templates | In scripts |
| Environment dependencies | `requirements.txt` |
| External model APIs | GPT-4o via OpenAI API (key required) |

---
