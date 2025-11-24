# classify_entities.py

"""
Broad Biomedical Entity Typing for Enriched Triples
Authors: Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 31/10/2025

Description:
    Classifies each subject and object in enriched triples into a broad biomedical
    entity type (e.g., Gene/Protein, Disease/Condition, Chemical/Drug, Pathway, etc.).
    The script combines fast heuristics, ontology metadata, and (optionally) an OpenAI
    model to assign one high-level type per entity, separately for subject and object roles.

    Main features:
        - Reads triples_enriched.csv/XLSX (output of the entity linking pipeline)
          and builds a unique (entity, role, matched_label) list.
        - Uses ontology-aware heuristics (GO, HGNC, ChEBI, DOID, MONDO, HP, UBERON, etc.)
          to assign types without model calls when possible.
        - For remaining entities, queries an OpenAI model (default: gpt-4o-mini)
          with a JSON schema-constrained prompt to get:
              * type (from a fixed set of VALID_TYPES)
              * optional subtype
              * confidence and rationale
        - Verifies entity–label alignment with an additional LLM check when needed.
        - Uses a persistent JSON cache (entity_type_cache.json) to avoid reclassification.
        - Supports directory-based I/O:
              * If input is a directory, automatically finds triples_enriched.* and
                writes outputs into a sibling classified_<dirname>/ folder.

Input:
    - A CSV/XLSX triples file OR a directory containing it.
      Expected columns (case-insensitive):
        * subject / subject_normalized
        * object / object_normalized
        * subject_matched_label, subject_ontology_id, subject_ontology_name (optional)
        * object_matched_label, object_ontology_id, object_ontology_name (optional)
        * support_sentence (optional context)

Output:
    - classified_triples.csv (or classified_triples_pdf.csv for *pdf* dirs) in:
        classified_<input_dir_name>/
      containing all original columns plus:
        * subject_type, subject_type_confidence, subject_subtype
        * object_type,  object_type_confidence,  object_subtype
    - entity_type_cache.json stored in the same output directory.

Requirements:
    - Python ≥ 3.8
    - pandas
    - tqdm
    - openai (for LLM-based classification; otherwise only heuristics & cache are used)

Example usage (Mac/Linux):
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    python3 classify_entities.py ../data/neo4j_data/triples_enriched/gpt-full.csv

    # Using auto output dir/name from a result folder:
    python3 classify_entities.py entity_linking_results_pdf

    # Custom output CSV:
    python3 classify_entities.py entity_linking_results/triples_enriched.csv -o classified_triples.csv

Example usage (Windows, PowerShell):
    py -3.11 -m pip install openai
    $env:OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    py -3.11 classify_entities.py .\entity_linking_results\triples_enriched.csv -o classified_triples.csv
"""

import os
import re
import json
import math
import time
import argparse
import pathlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import logging
from tqdm import tqdm

# ====== Defaults (some will be overridden after we resolve paths) ======
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CACHE_PATH = pathlib.Path("entity_type_cache.json")
BATCH_SIZE = 100
RATE_LIMIT_SLEEP = 1.0
TIMEOUT_SEC = 120

# ---------------- util: safe lowercase ----------------
def _lower_safe(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        try:
            if math.isnan(x):
                return ""
        except Exception:
            pass
    return str(x).lower()

def _strip_md_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

def _extract_first_json_block(s: str) -> str:
    s = _strip_md_fences(s)
    start = None
    stack = []
    for i, ch in enumerate(s):
        if ch in "{[":
            if start is None:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opening = stack.pop()
            if (opening, ch) in {("{", "}"), ("[", "]")}:
                if not stack and start is not None:
                    return s[start:i+1]
    for opener, closer in (("{", "}"), ("[", "]")):
        a = s.find(opener)
        b = s.rfind(closer)
        if a != -1 and b != -1 and b > a:
            return s[a:b+1]
    return s

def _parse_any_json_loose(text: str):
    import json
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        block = _extract_first_json_block(s)
        return json.loads(block)

def _normalize_items(parsed):
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if "items" in parsed and isinstance(parsed["items"], list):
            return parsed["items"]
        for k in ("result", "results", "entities", "data", "output"):
            v = parsed.get(k)
            if isinstance(v, list):
                return v
    return None

def _role_key(entity: str, role: str | None) -> str:
    role = (role or "").strip().lower() or "unknown"
    return f"{entity}||{role}"

# ====== Heuristics (unchanged) ======
VALID_TYPES = {
    "Gene/Protein","RNA/microRNA","Biological Process/Function","Cellular Component/Organelle",
    "Disease/Condition","Phenotype/Clinical Finding","Chemical/Drug","Anatomy/Tissue/Cell Type",
    "Organism/Strain","Pathway","Intervention/Exposure/Behavior","Assay/Measurement",
    "Variant/Mutation","Device/Material","Unknown"
}

def heuristic_type(entity_text: str, ontology_name: str | None, ontology_id: str | None, matched_label: str | None) -> str | None:
    t    = _lower_safe(entity_text)
    lbl  = _lower_safe(matched_label)
    oid  = _lower_safe(ontology_id)
    onto = _lower_safe(ontology_name)

    if any(x in onto for x in ("reactome", "kegg", "wikipathways", "pathway")) or "wp:" in oid or "reactome:" in oid:
        return "Pathway"
    if onto == "go" or oid.startswith("go:"):
        if any(k in t for k in ("mitochondr", "organelle", "membrane", "nucleus", "golgi", "lysosome", "peroxisome")):
            return "Cellular Component/Organelle"
        return "Biological Process/Function"
    if any(p in oid for p in ("chebi:", "chembl:", "drugbank:", "pubchem:")) \
       or any(k in onto for k in ("chebi", "chembl", "drugbank", "pubchem")) \
       or any(k in t for k in ("inhibitor", "agonist", "antagonist", "compound", "drug")):
        return "Chemical/Drug"
    if any(p in oid for p in ("doid:", "mondo:", "icd", "mesh:")) or any(k in onto for k in ("doid", "mondo", "icd", "ncit")):
        return "Disease/Condition"
    if any(w in t for w in ("disease", "disorder", "syndrome", "cancer", "neoplasm", "diabetes", "hypertension", "sarcopenia")):
        return "Disease/Condition"
    if onto in ("hp", "hpo") or oid.startswith("hp:") or "phenotype" in t or "clinical finding" in t:
        return "Phenotype/Clinical Finding"
    if onto in ("uberon", "fma") or oid.startswith(("uberon:", "fma:")):
        return "Anatomy/Tissue/Cell Type"
    if onto == "cl" or oid.startswith("cl:") or "cell" in t or "tissue" in t:
        return "Anatomy/Tissue/Cell Type"
    if any(p in oid for p in ("hgnc:", "entrez:", "ncbigene:", "uniprot:", "pr:")) \
       or any(k in onto for k in ("hgnc", "uniprot", "entrez", "ncbigene", "pr")):
        return "Gene/Protein"
    if any(k in t for k in ("protein", "enzyme", "kinase", "receptor", "transporter", "subunit")):
        return "Gene/Protein"
    if re.fullmatch(r"[A-Z0-9\-]{2,8}", entity_text or ""):
        return "Gene/Protein"
    if ("microrna" in t) or ("mir-" in t) or (" mir" in t) or (lbl == "microrna") or "mirbase:" in oid \
       or entity_text.lower().startswith(("mir-", "mir", "mirna", "hsa-mir")):
        return "RNA/microRNA"
    if oid.startswith(("enst", "ensr")):
        return "RNA/microRNA"
    if re.search(r"\brs\d+\b", t) or any(p in oid for p in ("dbsnp:", "clinvar:", "so:")) or "variant" in t or "mutation" in t:
        return "Variant/Mutation"
    if any(k in t for k in ("mitochondria", "mitochondrial", "organelle", "nucleus", "membrane", "cytoskeleton", "ribosome")):
        return "Cellular Component/Organelle"
    if any(k in t for k in ("exercise", "training", "physical activity", "diet", "intervention", "supplementation")):
        return "Intervention/Exposure/Behavior"
    if any(k in t for k in ("assay", "measurement", "biomarker", "rt-pcr", "western blot", "elisa", "immunohistochemistry", "mass spectrometry")):
        return "Assay/Measurement"
    if "pathway" in t:
        return "Pathway"
    if "function" in t and not ("liver function" in t or "kidney function" in t):
        return "Biological Process/Function"
    return None

# ====== OpenAI client ======
def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("The 'openai' package is required. Install with: pip install openai") from e
    return OpenAI()

# ====== Prompts & schema ======
JSON_SCHEMA = {
    "name": "entity_type_result",
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["entity", "type", "confidence"],
                    "properties": {
                        "uid": {"type": "integer"},
                        "entity": {"type": "string"},
                        "type": {"type": "string", "enum": sorted(list(VALID_TYPES))},
                        "subtype": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string"}
                    }
                }
            }
        },
        "required": ["items"],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = """You are an expert biomedical curator.

Task: For each entity, assign ONE broad biological entity type from the allowed set.
Return only JSON using the provided schema. If a 'uid' integer is provided in the prompt,
include it back in the corresponding item for reliable mapping.

Allowed types:
- Gene/Protein
- RNA/microRNA
- Biological Process/Function
- Cellular Component/Organelle
- Disease/Condition
- Phenotype/Clinical Finding
- Chemical/Drug
- Anatomy/Tissue/Cell Type
- Organism/Strain
- Pathway
- Intervention/Exposure/Behavior
- Assay/Measurement
- Variant/Mutation
- Device/Material
- Unknown

Be conservative; prefer concrete types over Unknown. Return ONLY JSON; no prose.
"""

def make_user_prompt(batch: List[Dict[str, Any]]) -> str:
    lines = ["Classify the following entities (avoid 'Unknown' unless necessary):\n"]
    for i, item in enumerate(batch, 1):
        entity = item["entity"]
        sent = item.get("context") or ""
        onto = item.get("ontology_name") or ""
        oid = item.get("ontology_id") or ""
        label = item.get("matched_label") or ""
        role  = item.get("role") or ""
        lines.append(
            f"{i}. role: {role}\n   entity: {entity}\n   context: {sent}\n   ontology_name: {onto}   ontology_id: {oid}   matched_label: {label}"
        )
    return "\n".join(lines)

def make_user_prompt_with_uid(batch: List[Dict[str, Any]], uids: List[int]) -> str:
    lines = ["Classify the following entities (echo each uid exactly). Avoid 'Unknown' unless nothing fits:\n"]
    for i, (item, uid) in enumerate(zip(batch, uids), 1):
        lines.append(
            f"{i}. uid: {uid}\n"
            f"   role: {item.get('role','')}\n"
            f"   entity: {item['entity']}\n"
            f"   context: {item.get('context','')}\n"
            f"   ontology_name: {item.get('ontology_name','')}   "
            f"ontology_id: {item.get('ontology_id','')}   "
            f"matched_label: {item.get('matched_label','')}"
        )
    return "\n".join(lines)

VERIFY_SYSTEM_PROMPT = """You are a strict biomedical curator.
Decide if these two strings denote the SAME concept. Be conservative.
Return ONLY JSON like: {"same": true|false, "reason": "<short reason>"}"""

def verify_alignment_with_llM(client, entity: str, matched_label: str, context: str | None = None) -> dict:
    if not matched_label or not entity:
        return {"same": True, "reason": "No matched label or entity; skipping."}
    ctx = (context or "").strip()
    user_prompt = (
        "ENTITY: " + entity.strip() + "\n"
        "MATCHED_LABEL: " + matched_label.strip() + ("\nCONTEXT: " + ctx if ctx else "") +
        "\n\nAnswer with JSON only."
    )
    try:
        resp = client.responses.create(
            model=MODEL, timeout=TIMEOUT_SEC, temperature=0.0,
            input=[{"role":"system","content":VERIFY_SYSTEM_PROMPT},
                   {"role":"user","content":user_prompt}]
        )
        txt = getattr(resp, "output_text", "") or ""
        txt = _strip_md_fences(txt)
        data = _parse_any_json_loose(txt) or {}
        return {"same": bool(data.get("same")), "reason": str(data.get("reason",""))}
    except Exception:
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.0,
                messages=[{"role":"system","content":VERIFY_SYSTEM_PROMPT},
                          {"role":"user","content":user_prompt}],
                response_format={"type":"json_object"},
            )
            data = _parse_any_json_loose(resp.choices[0].message.content) or {}
            return {"same": bool(data.get("same")), "reason": str(data.get("reason",""))}
        except Exception:
            return {"same": True, "reason": "Verifier failed; allowing."}

def call_llm(client, items: List[Dict[str, Any]], prompt_override: str | None = None) -> List[Dict[str, Any]]:
    prompt = prompt_override or make_user_prompt(items)
    try:
        resp = client.responses.create(
            model=MODEL, timeout=TIMEOUT_SEC, temperature=0.0,
            input=[{"role":"system","content":SYSTEM_PROMPT + "\nReturn ONLY one JSON object with key 'items'."},
                   {"role":"user","content":prompt}]
        )
        txt = getattr(resp, "output_text", None)
        if not txt:
            chunks = []
            for block in getattr(resp, "output", []) or []:
                for c in getattr(block, "content", []) or []:
                    if getattr(c, "type", "") == "output_text" and hasattr(c, "text"):
                        chunks.append(c.text)
            txt = "".join(chunks) if chunks else None
        if not txt:
            raise RuntimeError("Responses API returned no text.")
        parsed = _parse_any_json_loose(txt)
        items_list = _normalize_items(parsed)
        if items_list is None:
            raise KeyError("'items' not found")
        return items_list
    except Exception:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.0,
            messages=[{"role":"system","content":SYSTEM_PROMPT + "\nReturn ONLY one JSON object with key 'items'."},
                      {"role":"user","content":prompt}],
            response_format={"type":"json_object"},
        )
        parsed = _parse_any_json_loose(resp.choices[0].message.content)
        items_list = _normalize_items(parsed)
        if items_list is None:
            # fallback to per-item minimal Unknowns
            return [{"entity": it["entity"], "type": "Unknown", "confidence": 0.3, "subtype": "", "rationale": "Fallback"}
                    for it in items]
        return items_list

# ====== Cache helpers ======
def load_cache() -> Dict[str, Dict[str, Any]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}

def save_cache(cache: Dict[str, Dict[str, Any]]):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

# ====== Build entity list from enriched triples ======
def build_entity_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()  # (entity, role, matched_label)

    def pick_series(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
        s = frame.get(primary)
        if s is None or (hasattr(s, "isna") and s.isna().all()):
            s = frame.get(fallback)
        if s is None:
            raise KeyError(f"Neither '{primary}' nor '{fallback}' exists in the dataframe.")
        return s

    def add(role: str, entity_col_prefix: str):
        text     = pick_series(df, f"{entity_col_prefix}_normalized", entity_col_prefix).fillna("").astype(str)
        matched  = df.get(f"{entity_col_prefix}_matched_label")
        oid      = df.get(f"{entity_col_prefix}_ontology_id")
        oname    = df.get(f"{entity_col_prefix}_ontology_name")
        contextS = df.get("support_sentence")

        for idx, ent in enumerate(text):
            ent = ent.strip()
            if not ent:
                continue
            mlbl = "" if matched is None else str(matched.iloc[idx] or "").strip()
            key = (ent, role, mlbl)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "entity": ent,
                "role": role,
                "context": str(contextS.iloc[idx] if contextS is not None else "")[:400],
                "ontology_id":  "" if oid   is None else str(oid.iloc[idx] or ""),
                "ontology_name": "" if oname is None else str(oname.iloc[idx] or ""),
                "matched_label": mlbl,
            })

    add("subject", "subject")
    add("object",  "object")
    return items

def classify_entities_with_cache(to_classify: List[Dict[str, Any]], checkpoint_every: int = 0) -> Dict[str, Dict[str, Any]]:
    cache = load_cache()
    client = get_openai_client()

    cache_hits = 0
    heuristic_hits = 0
    worklist: List[Dict[str, Any]] = []

    for item in to_classify:
        entity = item["entity"]
        role   = item.get("role", "")
        ckey   = _role_key(entity, role)

        if ckey in cache:
            cache_hits += 1
            continue

        guess = heuristic_type(entity, item.get("ontology_name"), item.get("ontology_id"), item.get("matched_label"))
        if guess:
            heuristic_hits += 1
            cache[ckey] = {"type": guess, "confidence": 0.85, "subtype": "", "rationale": "Heuristic prelabel"}
            continue

        worklist.append(item)

    logging.info(f"Cache hits: {cache_hits:,} | Heuristic hits: {heuristic_hits:,} | To query: {len(worklist):,}")

    if not worklist:
        save_cache(cache)
        return cache

    num_batches = (len(worklist) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=num_batches, desc="LLM batches", unit="batch")

    for bi in range(0, len(worklist), BATCH_SIZE):
        batch = worklist[bi: bi + BATCH_SIZE]
        uids = list(range(bi + 1, bi + 1 + len(batch)))
        prompt = make_user_prompt_with_uid(batch, uids)

        try:
            llm_items = call_llm(client, batch, prompt_override=prompt)
        except Exception as e:
            logging.error(f"Batch {bi//BATCH_SIZE + 1}/{num_batches} failed: {e}")
            raise

        meta_by_uid  = {uid: meta for uid, meta in zip(uids, batch)}
        meta_by_pair = {(m["entity"], m.get("role","")): m for m in batch}

        for rec in llm_items:
            ent  = (rec.get("entity") or "").strip()
            uid  = rec.get("uid", None)
            base_conf = float(rec.get("confidence", 0.6))
            rationale = rec.get("rationale", "")
            subtype   = rec.get("subtype", "")
            meta = None

            if isinstance(uid, int) and uid in meta_by_uid:
                meta = meta_by_uid[uid]
            else:
                meta = meta_by_pair.get((ent, "subject")) or meta_by_pair.get((ent, "object")) or None
                if meta is None and ent:
                    for m in batch:
                        if m["entity"] == ent:
                            meta = m
                            break
            if not meta:
                continue

            role = meta.get("role", "")
            ckey = _role_key(ent, role)
            matched_label = (meta.get("matched_label") or "").strip()
            ctx = (meta.get("context") or "").strip()

            verdict = verify_alignment_with_llM(get_openai_client(), ent, matched_label, ctx) if matched_label else {"same": True}
            if not verdict.get("same", True):
                base_conf = 0.0
                rationale = (rationale + (" | " if rationale else "")) + f"alignment_check({role}): {verdict.get('reason','different concepts')}"

            cache[ckey] = {
                "type": rec.get("type", "Unknown"),
                "confidence": base_conf,
                "subtype": subtype,
                "rationale": rationale,
            }

        save_cache(cache)
        pbar.update(1)
        time.sleep(RATE_LIMIT_SLEEP)

    pbar.close()
    return cache

def build_items_and_log(df: pd.DataFrame) -> List[Dict[str, Any]]:
    items = build_entity_list(df)
    logging.info(f"Rows: {len(df):,}")
    logging.info(f"Unique (entity, role, matched_label) items: {len(items):,}")
    return items

# ====== NEW: IO resolver that derives output folder/file by input directory ======
def resolve_io(input_arg: str, output_arg: str | None) -> Tuple[Path, Path, Path, Path]:
    """
    Returns: (input_file, output_csv, cache_path, output_dir)
    Rules:
      • If input_arg is a DIR: look for triples_enriched.csv/xlsx in it.
      • Output dir = sibling 'classified_<dirname>'.
      • If 'pdf' in dirname (case-insensitive) → filename 'classified_triples_pdf.csv', else 'classified_triples.csv'.
      • Cache stored inside output dir as 'entity_type_cache.json'.
    """
    in_path = Path(input_arg)
    if in_path.is_dir():
        # Search preferred filenames
        candidates = [
            in_path / "triples_enriched.csv",
            in_path / "triples_enriched.xlsx",
            in_path / "triples_enriched.xls",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            # fallback: first CSV/XLSX in dir
            pool = list(in_path.glob("*.csv")) + list(in_path.glob("*.xlsx")) + list(in_path.glob("*.xls"))
            if not pool:
                raise FileNotFoundError(f"No triples file found in directory: {in_path}")
            found = pool[0]
        input_file = found
        base_dir = in_path
    else:
        if not in_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {in_path}")
        input_file = in_path
        base_dir = in_path.parent

    tag = base_dir.name
    outdir = base_dir.parent / f"classified_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    is_pdf = "pdf" in tag.lower()
    default_outname = "classified_triples_pdf.csv" if is_pdf else "classified_triples.csv"
    output_csv = Path(output_arg) if output_arg else (outdir / default_outname)

    cache_path = outdir / "entity_type_cache.json"
    return input_file, output_csv, cache_path, outdir

# ====== Main pipeline ======
def main(input_path_str: str, output_csv_str: str | None, checkpoint_every: int = 0):
    global CACHE_PATH

    input_file, output_csv, cache_path, outdir = resolve_io(input_path_str, output_csv_str)
    CACHE_PATH = cache_path  # make cache dir-specific
    logging.info(f"Resolved input file: {input_file}")
    logging.info(f"Output dir: {outdir}")
    logging.info(f"Output CSV: {output_csv}")
    logging.info(f"Cache path: {CACHE_PATH}")

    # Read input
    if input_file.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(input_file)
    else:
        try:
            df = pd.read_csv(input_file, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(input_file, encoding="utf-8")

    # --- make column lookup case-insensitive ---
    df.columns = [c.strip().lower() for c in df.columns]
    # Build items + classify
    items = build_items_and_log(df)
    results = classify_entities_with_cache(items, checkpoint_every=checkpoint_every)

    # Attach back
    df_out = df.copy()

    def pick_series(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
        s = frame.get(primary)
        if s is None or (hasattr(s, "isna") and s.isna().all()):
            s = frame.get(fallback)
        if s is None:
            raise KeyError(f"Neither '{primary}' nor '{fallback}' exists in the dataframe.")
        return s

    def attach(col_prefix: str):
        ent_series = pick_series(df_out, f"{col_prefix}_normalized", col_prefix).fillna("").astype(str)
        types, confs, subtypes = [], [], []
        for ent in tqdm(ent_series, desc=f"Attach {col_prefix}", unit="row"):
            ent = ent.strip()
            ckey = _role_key(ent, col_prefix)
            rec = results.get(ckey)
            if not rec:
                guess = heuristic_type(ent, None, None, None)
                if guess:
                    rec = {"type": guess, "confidence": 0.7, "subtype": "", "rationale": "Heuristic fallback"}
                else:
                    rec = {"type": "Unknown", "confidence": 0.4, "subtype": "", "rationale": "No match"}
            types.append(rec["type"])
            confs.append(rec["confidence"])
            subtypes.append(rec.get("subtype", ""))
        df_out[f"{col_prefix}_type"] = types
        df_out[f"{col_prefix}_type_confidence"] = confs
        df_out[f"{col_prefix}_subtype"] = subtypes

    attach("subject")
    attach("object")

    # Save
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    logging.info(f"Saved: {output_csv}  ({len(df_out):,} rows, {len(df_out.columns):,} cols)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Broad entity typing for subject/object columns (role-aware).")
    parser.add_argument("input", type=Path, help="Path to triples file OR directory containing it.")
    parser.add_argument("-o", "--output", default=None, help="Optional output CSV path (will override auto naming).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save cache every N batches (0=never)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY not set. The script will rely on heuristics and cache only.")

    main(str(args.input), args.output, checkpoint_every=args.checkpoint_every)
