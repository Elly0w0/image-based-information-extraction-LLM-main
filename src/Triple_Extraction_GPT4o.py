from openai import OpenAI
import os
import requests
import pandas as pd


"""
Image Semantic Triples Extraction Script
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 06/11/2024
Description: 
    This script uses OpenAI's GPT model to analyze images depicting comorbidities between COVID-19 and neurodegeneration.
    The primary goal is to extract key pathophysiological mechanisms and represent each mechanism as semantic triples.
    Triples are in the format "subject–predicate–object," providing a structured way to capture information from the image.

    Key functionalities include:
    - Image URL processing: Reads a list of image URLs from an Excel file.
    - GPT-based mechanism and triples extraction: Uses OpenAI's API to analyze each image and extract mechanisms and related triples.
    - Data parsing and storage: Extracted data is parsed into a structured format and saved as both .csv and .xlsx files.

    Input:
        - Excel file containing image URLs (column named "URL").
    
    Output:
        - CSV and Excel files containing extracted mechanisms and triples in a structured format.

    Requirements:
        - Access to the OpenAI API (API key required).
        - Pandas library for data manipulation.
        - Requests library for HTTP requests.

    Usage:
        - Run this script in an environment where Pandas and Requests are installed.
        - Ensure a valid API key for OpenAI GPT is provided.

"""


def gpt_authenticate(API_key):
    """
    Authenticates with the OpenAI GPT API using the provided API key.
    
    Args:
        API_key (str): OpenAI API key for authentication.

    Returns:
        OpenAI: Authenticated OpenAI API client.
    """
    client = OpenAI(api_key=API_key)
    return client


def gpt_extract(client, url):
    """
    Sends an image URL to GPT for analysis, prompting the model to describe key pathophysiological mechanisms 
    between COVID-19 and neurodegeneration in the form of semantic triples.

    Args:
        client (OpenAI): Authenticated OpenAI API client.
        url (str): Image URL for analysis.

    Returns:
        str: GPT-generated content with extracted mechanisms and triples in structured text.
    """
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": '''Describe the image (Figure/Graphical abstract) from an article on comorbidity between COVID-19 and Neurodegeneration.   
                                      1. Name potential mechanisms (pathophysiological processes) of Covid-19's impact on the brain depicted in the image. 
                                      2. Describe each process depicted in the image as semantic triples (subject–predicate–object).  
                                      Example: 
                                      Pathophysiological Process: Astrocyte_Activation 
                                      Triples:
                                      SARS-CoV-2_infection|triggers|astrocyte_activation
                                      
                                      Use ONLY the information shown in the image! Follow the structure precisely and don't write anything else! Replace spaces in names with _ sign, make sure that words "Pathophysiology Process:" and "Triples:" are presented, don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a | sign, four and more are not allowed!'''},
          {
            "type": "image_url",
            "image_url": {
              "url": url,
            },
          },
        ],
      }
    ],
    max_tokens=900,
  )
    content = response.choices[0].message.content
    return content

def triples_extraction_from_urls(path_URLs, API_key):
    """
    Extracts pathophysiological mechanisms and triples from images at specified URLs using the GPT model.
    Each mechanism is represented as a semantic triple in the format "subject|predicate|object".

    Args:
        path_URLs (str): Path to an Excel file containing image URLs.
        API_key (str): OpenAI API key for authenticating GPT.

    Returns:
        None: Saves the extracted data to 'Triples_Final.csv' and 'Triples_Final.xlsx'.
    """
    client = gpt_authenticate(API_key)
    relevance_GPT = pd.read_excel(path_URLs)

    # Initialize list to store parsed data
    parsed_data = []

    # Loop through each URL in the dataframe
    for idx, row in relevance_GPT.iterrows(): 
        try:
            url = row["URL"]

            # Extract content from the image using GPT
            content = gpt_extract(client, url)
            print(url, content)

            # Parse the text for mechanisms and triples
            mechanisms = content.strip().split('Pathophysiological Process: ')
            for mechanism_block in mechanisms[1:]:  # Skip the first split part (before the first "Process")
                lines = mechanism_block.strip().split('\n')
                mechanism_name = lines[0].strip()
                triples = lines[2:]  # Skip the 'Triples:' line

                for triple in triples:
                    subject, predicate, obj = triple.strip().split('|')
                    parsed_data.append([url, mechanism_name, subject, predicate, obj])

        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue  # Continue to the next row if there's an error

    # Save parsed data to a DataFrame
    parsed_df = pd.DataFrame(parsed_data, columns=['URL', 'Pathophysiological Process', 'Subject', 'Predicate', 'Object'])

    # Save to CSV and Excel
    parsed_df.to_csv('Triples_Final.csv', index=False)
    parsed_df.to_excel('Triples_Final.xlsx', index=False)
    print('Triples_Final file is successfully saved as CSV and Excel.')


if __name__ == "__main__":
    path_URLs = input("Enter the path for the file with image URLs to analyze: ")
    API_key = input("Enter your API key for GPT: ")
    
    # Perform the Triples Extraction and save the files
    triples_extraction_from_urls(path_URLs, API_key)