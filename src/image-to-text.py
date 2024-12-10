#%%
from openai import OpenAI
import os
import base64
import requests
import json
#%%
def gpt_authenticate():
  API_key = ""
  client = OpenAI(
      api_key = API_key)
  models = client.models.list()
  print(models)
  return client

#%% using links as images
def gpt_extract(client, url):
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "This figure has been extracted from a scientific publication discussing the relationship between Covid-19 and neurodegeneration. Please analyze the image and describe in detail what it shows. Focus on the main elements, any significant trends, patterns, or notable data points, and explain their relevance to the topic of Covid-19 and neurodegeneration"},
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

# %% generalize function
import pandas as pd
data = pd.read_excel("covid-neurodegeneration.xlsx")
client = gpt_authenticate()
for idx, row in data.iterrows():
  try:
    url = row["image_url"]
    doi = row["DOI"]
    #print(url)
    content = gpt_extract(client, url)
    #Write the content to a text file
    directory_name = '{}.txt'.format(doi)
    directory_name = "results/"
    if not os.path.exists(directory_name):
      os.makedirs(directory_name)
    # Construct the file path
    file_path = os.path.join(directory_name, "{}.txt".format(doi.replace("/","-")))
    # Write the content to the file with UTF-8 encoding
    with open(file_path, 'w+', encoding='utf-8') as file:
        file.write(content)
    print("Content written to response_content.txt")
  except Exception as e:
    print(e)
    import sys
    sys.exit()


# %%
