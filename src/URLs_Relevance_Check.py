import os
import requests
import pandas as pd
import time
from openai import OpenAI
from http.client import RemoteDisconnected


"""
Relevance Check for Bioinformatics Image URLs
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 23/10/2024
Description: 
    This script analyzes a dataset of bioinformatics-related image URLs to assess their relevance
    for a scientific publication. It uses both automated URL accessibility checks and the OpenAI GPT API
    to classify each image as relevant or irrelevant based on predefined criteria.

    Key functionalities include:
    - URL accessibility check: Verifies if each image URL is accessible and points to an image file.
    - Relevance classification via GPT: Authenticates with the OpenAI API to analyze each image and determine
      relevance based on specific bioinformatics criteria related to Covid-19 and neurodegeneration.
    - Data storage: Saves the results (relevance classification) in Excel files, with one file containing
      all URLs and their classifications, and another file with only the relevant URLs.
    
    Input:
        - Excel file containing image URLs.
    
    Output:
        - Excel file with URLs and their GPT-based relevance classifications.
        - Excel file with only relevant URLs based on GPT's analysis.
    
    Requirements:
        - Access to the OpenAI API (API key required).
        - Pandas for data handling.
        - Requests library for HTTP requests.
    
    Usage:
        Run this script in an environment where Pandas, Requests, and OpenAI libraries are installed.
"""


def gpt_authenticate(API_key):
    """
    Authenticates with the OpenAI GPT API using a predefined API key and returns the authenticated client object.
    
    Returns:
        client (OpenAI): Authenticated OpenAI API client.
    """
    client = OpenAI(api_key = API_key)
    return client


def check_image_url(url, retries=3):
    """
    Checks the accessibility of a given image URL by sending an HTTP GET request and handling retries on failure.
    It validates whether the URL points to an image by examining the Content-Type header.

    Args:
        url (str): The URL of the image to be checked.
        retries (int, optional): Number of retry attempts in case of connection failures. Default is 3.

    Returns:
        bool: Returns False if the URL is accessible and contains an image; True if an error occurred or the content is not an image.
    """
    attempt = 0
    while attempt < retries:
        try:
            # Send a GET request to the image URL
            response = requests.get(url, timeout=5)  # Set a timeout to avoid hanging
            
            # Check if the response status code is 200 (OK)
            if response.status_code == 200:
                # Get the Content-Type header from the response
                content_type = response.headers.get('Content-Type', '')

                # Check if the Content-Type is an image format
                if 'image' in content_type:
                    print(f"Success: URL {url} is accessible and is of type {content_type}.")
                    return False  # No error
                else:
                    print(f"Error: URL {url} is not an image. Content-Type: {content_type}")
                    return True  # Error occurred
            else:
                print(f"Error: URL {url} returned status code {response.status_code}.")
                return True  # Error occurred

        except (requests.ConnectionError, requests.Timeout, RemoteDisconnected) as e:
            attempt += 1
            print(f"Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                print("Retrying after a short delay...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(f"Failed to access URL after {retries} attempts: {url}")
                return True  # Error occurred after retries
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return True  # Error occurred due to unknown exception

    return True  # Fallback in case of any unexpected behavior


def URLs_access_check(dataframe, retries=3):
    """
    Iterates over a DataFrame containing image URLs, checks the accessibility of each URL, and stores the results.
    
    Args:
        dataframe (pandas.DataFrame): DataFrame containing a column 'image_url' with image URLs.
        retries (int, optional): Number of retry attempts in case of connection failures. Default is 3.

    Returns:
        pandas.DataFrame: DataFrame containing two columns: 'URL' and 'Access' (Yes/No), indicating the accessibility of each URL.
    """
    access_data = []
    for idx, row in dataframe.iterrows():
        url = row['image_url']

        # Check if the image URL is accessible with retry mechanism
        access = "No" if check_image_url(url, retries=retries) else "Yes"
        access_data.append([url, access])

    # Create a new DataFrame to store the results
    access_df = pd.DataFrame(access_data, columns=['URL', 'Access'])

    return access_df


def gpt_extract(client, url):
    """
    Uses GPT to analyze an image and determine its relevance based on predefined criteria. 
    The URL is passed to the GPT model, which returns either "Yes" or "No" indicating the relevance of the image.

    Args:
        client (OpenAI): The authenticated OpenAI API client.
        url (str): The URL of the image to be analyzed.

    Returns:
        str: The GPT's assessment of the image's relevance ("Yes" or "No").
    """
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": '''Image URL is given. Analyze this image and assign relevance to it: "Not" for irrelevant images and "Yes" for relevant ones.
                                      Follow this classification:
                                      Relevant:
                                      Images which:
                                      Сlearly demonstrate relationship between Covid-19 and neurodegeneration (any neurological impacts).
                                      Don't contain lots of text (no more than 500 characters).
                                      Don't just depict a research outline.
                                      Don't be just graphs or represent photos derived from scientific tools (microscopic, histological images) and data visualization.
                                      Are likely to be cartoons drawn by article authors.
                                      
                                      Irrelevant:
                                      Unrelated images, for example just an image of a virus particle or a sick person.
                                      Images where correct interpretation of the data is impossible.
                                      Images which display insights into Covid-19 OR Neurodegeneration, if one is present and the other is missing.
                                      
                                      Your answer should contain only a final decision in the following format: No/Yes (without dots)
                                      Don't write anything else!'''},
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


def get_GPT_answers(dataframe, API_key):
    """
    Authenticates with GPT, processes a DataFrame containing image URLs, and retrieves the relevance classification for each image.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing a column 'image_url' with image URLs.

    Returns:
        list: A list of lists where each inner list contains the image URL and the GPT's relevance classification or "Error" in case of failure.
    """
    client = gpt_authenticate(API_key)
    parsed_data = []

    for idx, row in dataframe.iterrows():
        try:
            url = row["image_url"]  # Extract image URL
            relev_check = gpt_extract(client, url) # Extract content from the image using GPT
            print(url, relev_check)
            parsed_data.append([url, relev_check])

        except Exception as e:
            # Print the error and continue to the next row
            print(f"Error processing {idx, row}: {e}")
            parsed_data.append([url, "Error"])
            continue  # Skip to the next row in case of an error
    
    return parsed_data


def relevance_check_main(path, name_raw_data):
    """
    Main function to perform relevance checks on image URLs using GPT and save results to Excel.

    Args:
        path (str): Directory path containing the input file.
        name_raw_data (str): Name of the input Excel file with raw data.
    """
    # Convert the Excel file into dataframe
    data_raw = pd.read_excel(os.path.join(path, name_raw_data))

    # Drop 'Unnamed: 0' only if it exists in the DataFrame
    if 'Unnamed: 0' in data_raw.columns:
        data_raw.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Get relevance data
    parsed_data_all = get_GPT_answers(data_raw, API_key)
    
    # Loop through the list and clean up 'Yes.' or 'No.' to 'Yes' or 'No'
    for item in parsed_data_all:
        # Check if the second element ends with a period and is 'Yes.' or 'No.'
        if item[1] in ['Yes.', 'No.']:
            item[1] = item[1].rstrip('.')
            
    # Create a new DataFrame to store parsed results
    relevance_GPT_all = pd.DataFrame(parsed_data_all, columns=['URL', 'Relevance_GPT'])
    relevance_GPT_all.to_excel(os.path.join(path, "Relevance_assignment_GPT_4o.xlsx"), index=False)
    
    # Store only relevant URLs
    relevant_URLs = relevance_GPT_all[relevance_GPT_all['Relevance_GPT'] == 'Yes']
    relevant_URLs.reset_index(drop=True, inplace=True)
    relevant_URLs.to_excel(os.path.join(path, "Relevant_URLs_only_GPT_4o.xlsx"), index=False)
    
    # Number of images which GPT can't process
    print('Initial total number of URLs:', len(data_raw))
    print("Number of images which GPT can't process",len(relevance_GPT_all.loc[relevance_GPT_all['Relevance_GPT'] == 'Error']))
    print('Number of relevant images (GPT):', relevant_URLs['Relevance_GPT'].value_counts().get('Yes', 0))



    
if __name__ == "__main__":
    path = input("Enter the directory path for the raw file: ")
    name_raw_data = input("Enter the name of the raw data file: ")
    API_key = input("Enter your API key for GPT: ")
    
    # Perform the Relevance Check and save the files
    relevance_check_main(path, name_raw_data)
    