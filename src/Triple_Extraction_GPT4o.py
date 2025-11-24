# Triple_Extraction_GPT4o.py

"""
Semantic Triple Extraction from Biomedical Images using GPT
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 06/11/2024  (updated 30/10/2025 for caption support & modes)

Description:
    This script extracts semantic triples from images related to the comorbidity between COVID-19 and neurodegeneration,
    using OpenAI's GPT-4o model. Each image is processed to identify pathophysiological mechanisms represented visually,
    and for each mechanism, structured triples (subject|predicate|object) are extracted.

    Key functionalities:
    1. Reads image URLs (and optional captions) from an input Excel file.
    2. Sends each image (and optional caption) to GPT with a strict prompt to ensure triple consistency.
    3. Parses the model output and extracts structured triples.
    4. Saves the results to CSV and Excel files.

Input:
    - Excel file containing columns:
        "Image_number", "URL", "GitHub_URL", and (optional) "Caption_text".
      Example path: data/baselines_and_ablations/Subset_50_URLs_with_captions_data.xlsx
    - OpenAI API key (via CLI argument or environment variable).

Output:
    - Triples_Final_All.csv: All extracted semantic triples.
    - Triples_Final_All.xlsx: Excel version of the same data.

Requirements:
    - openai
    - pandas
    - requests
    - Internet connection and OpenAI API access.

Usage (images only):
    python src/Triple_Extraction_GPT4o.py --input data/baselines_and_ablations/Subset_50_URLs_with_captions_data.xlsx --output data/triples_output/GPT_subset_triples_image_only --mode images_only --api_key YOUR_API_KEY

Usage (images with captions; rows with Caption_text == Not_available are skipped):
    python src/Triple_Extraction_GPT4o.py --input data/baselines_and_ablations/Subset_50_URLs_with_captions_data.xlsx --output data/triples_output/GPT_subset_triples_image_plus_caption --mode images_with_captions --api_key YOUR_API_KEY
"""

from openai import OpenAI
import pandas as pd
import os
import requests
import time


def gpt_authenticate(API_key):
    return OpenAI(api_key=API_key)


def is_url_accessible(url, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    try:
        head = requests.head(url, headers=headers, timeout=timeout)
        if head.status_code == 200 and 'image' in head.headers.get("Content-Type", ""):
            return True
        # Fallback to GET
        get = requests.get(url, headers=headers, stream=True, timeout=timeout)
        return get.status_code == 200 and 'image' in get.headers.get("Content-Type", "")
    except requests.RequestException as e:
        print(f"URL access error for {url}: {e}")
        return False


def build_prompt_text(include_caption=False):
    """
    Returns the base instruction text; if include_caption=True the text explicitly allows using the caption.
    """
    base = '''Describe the image (Figure/Graphical abstract) from an article on comorbidity between COVID-19 and Neurodegeneration.
1. Name potential mechanisms (pathophysiological processes) of COVID-19's impact on the brain depicted in the image.
2. Describe each process depicted in the image as semantic triples (subject|predicate|object).

Example:
Pathophysiological Process: Astrocyte_Activation
Triples:
SARS-CoV-2_infection|triggers|astrocyte_activation

Use ONLY the information shown in the image{CAPTION_CLAUSE}! Follow the structure precisely and don't write anything else! Replace spaces in names with _ sign, make sure that words "Pathophysiological Process:" and "Triples:" are presented, don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a | sign, four and more are not allowed!'''
    if include_caption:
        return base.replace("{CAPTION_CLAUSE}", " and in the accompanying caption provided below strictly for disambiguation")
    else:
        return base.replace("{CAPTION_CLAUSE}", "")


def gpt_extract(client, url, caption=None):
    """
    Sends an image URL (and optional caption/legend) to GPT-4o to extract structured pathophysiological triples.

    Args:
        client (OpenAI): Authenticated API client.
        url (str): Image URL.
        caption (str|None): Optional caption/legend text.

    Returns:
        str: Raw GPT output with extracted mechanisms and triples.
    """
    use_caption = bool(caption) and str(caption).strip().lower() != "not_available"

    prompt_text = build_prompt_text(include_caption=use_caption)

    if use_caption:
        user_content = [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": f"Caption:\n{caption}"},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    else:
        user_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": url}},
        ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_content}],
        max_tokens=2000,
        temperature=0.0,    # deterministic decoding for ablation comparability
        top_p=0.0
    )
    return response.choices[0].message.content


def triples_extraction_from_urls(input_path, output_path_base, API_key, mode="images_only"):
    """
    mode:
      - "images_only": ignore captions, process every accessible image
      - "images_with_captions": require a non-empty/non-Not_available caption; otherwise skip the image
    """
    client = gpt_authenticate(API_key)
    df = pd.read_excel(input_path)

    parsed_data = []

    for idx, row in df.iterrows():
        image_number = row["Image_number"]
        original_url = row["URL"]
        github_url = row["GitHub_URL"]
        caption_text = row["Caption_text"] if ("Caption_text" in df.columns) else None

        # If mode requires captions, skip rows with Not_available/empty captions
        if mode == "images_with_captions":
            cap_norm = (str(caption_text).strip() if caption_text is not None else "")
            if (not cap_norm) or (cap_norm.lower() == "not_available"):
                print(f"⏭ Skipping (no usable caption): {image_number} | {github_url}")
                continue

        if not is_url_accessible(github_url):
            print(f"❌ Skipping inaccessible URL: {github_url}")
            continue

        try:
            time.sleep(1.5)

            caption_to_send = None
            if mode == "images_with_captions":
                caption_to_send = caption_text  # here we know it's usable

            content = gpt_extract(client, github_url, caption=caption_to_send)
            print(f"\n{github_url}\n{content}")

            mechanisms = content.strip().split('Pathophysiological Process: ')
            for mechanism_block in mechanisms[1:]:
                lines = mechanism_block.strip().split('\n')
                if not lines:
                    continue
                mechanism_name = lines[0].strip()

                # find "Triples:" line; keep original simple parsing style
                triples_start_idx = None
                for i, ln in enumerate(lines):
                    if ln.strip().lower().startswith('triples'):
                        triples_start_idx = i + 1
                        break
                if triples_start_idx is None:
                    triples_start_idx = 1  # fallback

                triples = lines[triples_start_idx:]

                for triple in triples:
                    triple = triple.strip()
                    if '|' not in triple:
                        continue
                    parts = triple.split('|')
                    if len(parts) != 3:
                        continue
                    subject, predicate, obj = [p.strip() for p in parts]
                    parsed_data.append([
                        image_number, original_url, github_url,
                        mechanism_name, subject, predicate, obj
                    ])

        except Exception as e:
            print(f"Error processing {github_url}: {e}")
            continue

    parsed_df = pd.DataFrame(parsed_data, columns=[
        'Image_number', 'URL', 'GitHub_URL',
        'Pathophysiological Process', 'Subject', 'Predicate', 'Object'
    ])

    os.makedirs(os.path.dirname(output_path_base), exist_ok=True)
    parsed_df.to_csv(f"{output_path_base}.csv", index=False)
    parsed_df.to_excel(f"{output_path_base}.xlsx", index=False)

    print(f"✅ Triples saved to:\n- {output_path_base}.csv\n- {output_path_base}.xlsx")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract semantic triples from biomedical images using OpenAI GPT-4o.")
    parser.add_argument("--input", required=True, help="Path to Excel file with image data (columns: 'Image_number', 'URL', 'GitHub_URL', optional 'Caption_text').")
    parser.add_argument("--output", required=True, help="Output file base path without extension.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--mode", choices=["images_only", "images_with_captions"], default="images_only",
                        help="Input mode: 'images_only' (ignore captions) or 'images_with_captions' (skip rows with Not_available captions).")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api_key or set the OPENAI_API_KEY environment variable.")

    triples_extraction_from_urls(args.input, args.output, args.api_key, mode=args.mode)

# Examples:
# python src/Triple_Extraction_GPT4o.py --input data/baselines_and_ablations/Subset_50_URLs_with_captions_data.xlsx --output data/triples_output/GPT_subset_triples_image_only --mode images_only --api_key YOUR_API_KEY
# python src/Triple_Extraction_GPT4o.py --input data/baselines_and_ablations/Subset_50_URLs_with_captions_data.xlsx --output data/baselines_and_ablations/GPT_subset_50_URLs_with_captions_triples --mode images_with_captions --api_key YOUR_API_KEY
