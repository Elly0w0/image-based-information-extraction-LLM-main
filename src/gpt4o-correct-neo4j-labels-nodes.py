#!/usr/bin/env python3
# review_labels_neo4j.py
"""
Label Review and Correction of Neo4j Nodes via OpenAI GPT
Author: Elizaveta Popova
Institution: University of Bonn, Fraunhofer SCAI
Date: 06/08/2025

Description:
    This script connects to a Neo4j database, reads all nodes with a `name` property,
    and uses the OpenAI GPT API to verify or correct each node’s semantic label based
    on a controlled vocabulary. The model is prompted to select the most appropriate label 
    for each entity from a predefined list (e.g., Gene, Disease, Protein, etc.).

    If the model suggests a new label that differs from the current one, the script
    updates the node in Neo4j by removing the old label and assigning the new one.
    Labels are evaluated for consistency using a deterministic GPT prompt, and nodes 
    with the label “Unknown” are also reconsidered.

    All credentials are securely loaded from a local `config.ini` file.

Usage:
    python src/review_labels_neo4j.py
"""
import sys
import time
import logging
import configparser

from neo4j import GraphDatabase, basic_auth
from openai import OpenAI

# Configure logging to INFO (so DEBUG is suppressed)...
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ...then add a filter so only our two markers get through:
class OnlyUpdateUnchangedFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return msg.startswith("[UPDATED]") or msg.startswith("[UNCHANGED]")

for handler in logging.root.handlers:
    handler.addFilter(OnlyUpdateUnchangedFilter())

# Controlled vocabulary of allowed labels
CONTROLLED_VOCAB = [
    "Anatomical_Structure",
    "Biological_Process",
    "Cell",
    "Cell_Phenotype",
    "Chemical",
    "Disease",
    "Gene",
    "Phenotype",
    "Protein",
    "Pathway"
]

def review_label(client: OpenAI, entity: str, current_label: str) -> str:
    """
    Use the OpenAI GPT API to verify or correct the semantic label of a given entity.

    Args:
        client (OpenAI): An initialized OpenAI API client.
        entity (str): The name of the biological entity to classify.
        current_label (str): The entity's current label in the Neo4j database.

    Returns:
        str: The most appropriate label, either unchanged or corrected, based on GPT output.
    """
    logging.debug(f"Reviewing label for entity='{entity}', current_label='{current_label}'")
    prompt = (
        "You are a biomedical ontology expert. Your task is to verify or correct the label for a biological entity.\n\n"
        f"Entity: \"{entity}\"\n"
        f"Current Label: \"{current_label}\"\n\n"
        "Choose the **single most appropriate label** from the following controlled vocabulary:\n"
        f"{', '.join(CONTROLLED_VOCAB)}\n\n"
        "Rules:\n"
        "- Only return **one label**, as a plain string (e.g., `Gene` or `Phenotype`).\n"
        "- Do **not** include punctuation, quotes, extra words, or explanations.\n"
        "- Return the current label exactly as-is if it is already correct.\n"
        "- If none of the labels apply, return a **single new label** that best describes the entity.\n"
        "- The output must be **only the label name** — no commentary, no formatting.\n\n"
        "- If the entity current label is Unknown, choose the most semantically suitable label — avoid 'Unknown"
        "Output:\n"
        "The label string only."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You classify biological entities."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20
        )
        new_label = resp.choices[0].message.content.strip() or current_label
        logging.debug(f"GPT returned new_label='{new_label}' for entity='{entity}'")
        return new_label
    except Exception:
        logging.exception(f"GPT API error for entity='{entity}'")
        return current_label

def main():
    """
    Main routine to:
    1. Load configuration credentials from config.ini.
    2. Initialize OpenAI and Neo4j clients.
    3. Retrieve all Neo4j nodes with a 'name' property.
    4. For each node, use GPT to evaluate and potentially update its semantic label.
    5. Apply updates directly to the graph database.
    """
    logging.info("Starting review_labels_neo4j...")
    # 1) Load and validate config.ini
    cfg = configparser.ConfigParser()
    try:
        loaded = cfg.read("config.ini")
        if not loaded:
            raise FileNotFoundError("config.ini not found or could not be parsed")
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.exception("Unexpected error while loading config.ini")
        sys.exit(1)


    # Validate required sections
    for section in ("neo4j", "openai"):
        if section not in cfg:
            logging.error(f"Missing section [{section}] in config.ini")
            sys.exit(1)

    neo4j_conf = cfg["neo4j"]
    openai_conf = cfg["openai"]
    uri      = neo4j_conf.get("uri")
    user     = neo4j_conf.get("user")
    password = neo4j_conf.get("password")
    api_key  = openai_conf.get("api_key")

    if not all([uri, user, password, api_key]):
        logging.error("One or more credentials are missing in config.ini")
        sys.exit(1)

    # 2) Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # 3) Connect to Neo4j
    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    except Exception:
        logging.exception("Failed to connect to Neo4j")
        sys.exit(1)

    try:
        with driver.session() as session:
            query = (
                "MATCH (n)"
                " WHERE n.name IS NOT NULL"
                " RETURN id(n) AS id, n.name AS entity, labels(n) AS labels"
            )
            try:
                results = list(session.run(query))
            except Exception:
                logging.exception("Cypher query failed")
                return

            for i, rec in enumerate(results, start=1):
                try:
                    nid     = rec["id"]
                    ent     = rec["entity"]
                    labs    = rec["labels"]
                    current = next((L for L in labs if L in CONTROLLED_VOCAB), "")

                    new_label = review_label(client, ent, current)
                    if new_label != current:
                        remove_unknown = "REMOVE n:`Unknown`" if "Unknown" in labs else ""
                        remove_current = f"REMOVE n:`{current}`" if current else ""
                        add_clause     = f"SET n:`{new_label}`"
                        update_cypher = f"""
                            MATCH (n)
                            WHERE id(n) = $id
                            {remove_unknown}
                            {remove_current}
                            {add_clause}
                        """
                        session.run(update_cypher, id=nid)
                        logging.info(f"[UPDATED] '{ent}': {current or '<none>'} → {new_label}")
                    else:
                        logging.info(f"[UNCHANGED] '{ent}' remains '{current}'")

                except Exception:
                    logging.exception(f"Error processing record {i}")
                finally:
                    time.sleep(0.2)
    finally:
        driver.close()

if __name__ == "__main__":
    main()
