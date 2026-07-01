# Visual PDF RAG with Vespa - ColPali demo application[¶](#visual-pdf-rag-with-vespa-colpali-demo-application)

We created an end-to-end demo application for visual retrieval of PDF pages using Vespa, including a frontend web application. To see the live demo, visit <https://vespa-engine-colpali-vespa-visual-retrieval.hf.space/>.

The main goal of the demo is to make it easy for *you* to create your own PDF Enterprise Search application using Vespa. To deploy a full demo, you need two main components:

1. A Vespa application that lets you index and search PDF pages using ColPali embeddings.
1. A live web application that lets you interact with the Vespa application.

After running this notebook, you will have set up a Vespa application, and indexed some PDF pages. You can then test that you are able to query the Vespa application, and you will be ready to deploy the web application including the frontend.

Some of the features we want to highlight in this demo are:

- Visual retrieval of PDF pages using ColPali embeddings
- Explainability by displaying similarity maps over the patches in the PDF pages for each query token.
- Extracting queries and questions from the PDF pages using `gemini-1.5-8b` model.
- Type-ahead search suggestions based on the extracted queries and questions.
- Comparison of different retrieval and ranking strategies (BM25, ColPali MaxSim, and a combination of both).
- AI-generated responses to the query based on the top ranked PDF pages. Also using the `gemini-1.5-8b` model.

We also wanted to give a notion of which latency one can expect using Vespa for this use case. Event though your users might not state this explicitly, we consider it important to provide a snappy user experience.

In this notebook, we will prepare the Vespa backend application for our visual retrieval demo. We will use ColPali as the model to extract patch vectors from images of pdf pages. At query time, we use MaxSim to retrieve and/or (based on the configuration) rank the page results.

The steps we will take in this notebook are:

1. Setup and configuration
1. Download PDFs
1. Convert PDFs to images
1. Generate queries and questions
1. Generate ColPali embeddings
1. Prepare the Vespa application package
1. Deploy the Vespa application to Vespa Cloud
1. Feed the data to the Vespa application
1. Test a query to the Vespa application

All the steps that are needed to provision the Vespa application, including feeding the data, can be done by running this notebook. We have tried to make it easy for others to run this notebook, to create your own PDF Enterprise Search application using Vespa.

If you want to run this notebook in Colab, you can do so by clicking the button below:

## 1. Setup and Configuration[¶](#1-setup-and-configuration)

In \[ \]:

Copied!

```
!python --version
```

!python --version

Install dependencies:

Note that the python pdf2image package requires poppler-utils, see other installation options [here](https://pdf2image.readthedocs.io/en/latest/installation.html#installing-poppler).

In \[ \]:

Copied!

```
!sudo apt-get update && sudo apt-get install poppler-utils -y
```

!sudo apt-get update && sudo apt-get install poppler-utils -y

Now install the required python packages:

In \[ \]:

Copied!

```
!pip3 install colpali-engine==0.3.10 pdf2image pypdf==5.0.1 pyvespa>=0.50.0 vespacli numpy==1.26.4 pillow==10.4.0 google-generativeai==0.8.3 transformers python-dotenv
```

!pip3 install colpali-engine==0.3.10 pdf2image pypdf==5.0.1 pyvespa>=0.50.0 vespacli numpy==1.26.4 pillow==10.4.0 google-generativeai==0.8.3 transformers python-dotenv

In \[ \]:

Copied!

```
import os
import json
from typing import Tuple
import hashlib
import numpy as np

# Vespa
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Function,
    FieldSet,
    SecondPhaseRanking,
    Summary,
    DocumentSummary,
)
from vespa.deployment import VespaCloud
from vespa.application import Vespa
from vespa.io import VespaResponse

# Google Generative AI for Google Gemini interaction
import google.generativeai as genai

# Torch and other ML libraries
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pdf2image import convert_from_path
from pypdf import PdfReader

# ColPali model and processor
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Avoid warning from huggingface tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

import os import json from typing import Tuple import hashlib import numpy as np

# Vespa

from vespa.package import ( ApplicationPackage, Field, Schema, Document, HNSW, RankProfile, Function, FieldSet, SecondPhaseRanking, Summary, DocumentSummary, ) from vespa.deployment import VespaCloud from vespa.application import Vespa from vespa.io import VespaResponse

# Google Generative AI for Google Gemini interaction

import google.generativeai as genai

# Torch and other ML libraries

import torch from torch.utils.data import DataLoader from tqdm import tqdm from pdf2image import convert_from_path from pypdf import PdfReader

# ColPali model and processor

from colpali_engine.models import ColPali, ColPaliProcessor from colpali_engine.utils.torch_utils import get_torch_device

# Load environment variables

from dotenv import load_dotenv load_dotenv()

# Avoid warning from huggingface tokenizers

os.environ["TOKENIZERS_PARALLELISM"] = "false"

### Create a free trial in Vespa Cloud[¶](#create-a-free-trial-in-vespa-cloud)

Create a tenant from [here](https://vespa.ai/free-trial/). The trial includes $300 credit. Take note of your tenant name, and input it below.

In \[ \]:

Copied!

```
VESPA_TENANT_NAME = "vespa-team"  # Replace with your tenant name
```

VESPA_TENANT_NAME = "vespa-team" # Replace with your tenant name

Here, set your desired application name. (Will be created in later steps) Note that you can not have hyphen `-` or underscore `_` in the application name.

In \[ \]:

Copied!

```
VESPA_APPLICATION_NAME = "colpalidemodev"
VESPA_SCHEMA_NAME = "pdf_page"
```

VESPA_APPLICATION_NAME = "colpalidemodev" VESPA_SCHEMA_NAME = "pdf_page"

Next, you can to create a token. This is an optional authentication method (the default is mTLS), and will be used for feeding data, and querying the application. For details, see [Authenticating to Vespa Cloud](https://vespa-engine.github.io/pyvespa/authenticating-to-vespa-cloud.md). For now, we will use a single token with both read and write permissions. For production, we recommend separate tokens for feeding and querying, (the former with write permission, and the latter with read permission). The tokens can be created from the [Vespa Cloud console](https://console.vespa-cloud.com/) in the 'Account' -> 'Tokens' section. Please make sure to save the both the token id and it's value somwhere safe - you'll need it when you're going to connect to your app.

In \[ \]:

Copied!

```
# Replace this with the id of your token
VESPA_TOKEN_ID = "pyvespa_integration"  # This needs to match the token_id that you created in the Vespa Cloud Console
```

# Replace this with the id of your token

VESPA_TOKEN_ID = "pyvespa_integration" # This needs to match the token_id that you created in the Vespa Cloud Console

We also need to set the value of the write token to be able to feed data to the Vespa application (value of VESPA_TOKEN_ID_WRITE). Please run the cell below to set the variable.

In \[ \]:

Copied!

```
VESPA_CLOUD_SECRET_TOKEN = os.getenv("VESPA_CLOUD_SECRET_TOKEN") or input(
    "Enter Vespa cloud secret token: "
)
```

VESPA_CLOUD_SECRET_TOKEN = os.getenv("VESPA_CLOUD_SECRET_TOKEN") or input( "Enter Vespa cloud secret token: " )

We will use Google's Gemini API to create sample queries for our images. Create a Gemini API key from [here](https://aistudio.google.com/app/apikey). Once you have the key, please run the cell below. You can also use other VLM's to create these queries.

In \[ \]:

Copied!

```
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or input(
    "Enter Google Generative AI API key: "
)
# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
```

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or input( "Enter Google Generative AI API key: " )

# Configure Google Generative AI

genai.configure(api_key=GOOGLE_API_KEY)

### Loading the ColPali model from huggingface 🤗[¶](#loading-the-colpali-model-from-huggingface)

In \[ \]:

Copied!

```
MODEL_NAME = "vidore/colpali-v1.2"

# Set device for Torch
device = get_torch_device("auto")
print(f"Using device: {device}")

# Load the ColPali model and processor
model = ColPali.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=device,
).eval()

processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
```

MODEL_NAME = "vidore/colpali-v1.2"

# Set device for Torch

device = get_torch_device("auto") print(f"Using device: {device}")

# Load the ColPali model and processor

model = ColPali.from_pretrained( MODEL_NAME, torch_dtype=torch.float32, device_map=device, ).eval() processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

## 2. Download PDFs[¶](#2-download-pdfs)

We are going to use public reports from the Norwegian Government Pension Fund Global (also known as the Oil Fund). The fund puts transparency at the forefront and publishes reports on its investments, holdings, and returns, as well as its strategy and governance.

These reports are the ones we are going to use for this showcase. Here are some sample images:

As we can see, a lot of the information is in the form of tables, charts and numbers. These are not easily extractable using pdf-readers or OCR tools.

In \[ \]:

Copied!

```
import requests

pdfs = [
    {
        "url": "https://drive.google.com/uc?export=download&id=1nDO0KN_BjyFu42xFAfhJagOeeaJ8fhki",
        "path": "pdfs/gpfg-half-year-report-2024.pdf",
        "year": "2024",
    },
    {
        "url": "https://drive.google.com/uc?export=download&id=1Saw_wM8RI6Zej5qkWDDpeM-3tyOQQTwR",
        "path": "pdfs/gpfg-annual-report_2023.pdf",
        "year": "2023",
    },
]
```

import requests pdfs = [ { "url": "https://drive.google.com/uc?export=download&id=1nDO0KN_BjyFu42xFAfhJagOeeaJ8fhki", "path": "pdfs/gpfg-half-year-report-2024.pdf", "year": "2024", }, { "url": "https://drive.google.com/uc?export=download&id=1Saw_wM8RI6Zej5qkWDDpeM-3tyOQQTwR", "path": "pdfs/gpfg-annual-report_2023.pdf", "year": "2023", }, ]

### Downloading the PDFs[¶](#downloading-the-pdfs)

We create a function to download the PDFs from the web to the provided directory.

In \[ \]:

Copied!

```
PDFS_DIR = "pdfs"
os.makedirs(PDFS_DIR, exist_ok=True)


def download_pdf(url: str, path: str):
    r = requests.get(url, stream=True)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return path


# Download the pdfs
for pdf in pdfs:
    download_pdf(pdf["url"], pdf["path"])
```

PDFS_DIR = "pdfs" os.makedirs(PDFS_DIR, exist_ok=True) def download_pdf(url: str, path: str): r = requests.get(url, stream=True) with open(path, "wb") as f: for chunk in r.iter_content(chunk_size=1024): if chunk: f.write(chunk) return path

# Download the pdfs

for pdf in pdfs: download_pdf(pdf["url"], pdf["path"])

## 3. Convert PDFs to Images[¶](#3-convert-pdfs-to-images)

In \[ \]:

Copied!

```
def get_pdf_images(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)
    # Convert to PIL images
    images = convert_from_path(pdf_path)
    assert len(images) == len(page_texts)
    return images, page_texts


pdf_folder = "pdfs"
pdf_pages = []
for pdf in tqdm(pdfs):
    pdf_file = pdf["path"]
    title = os.path.splitext(os.path.basename(pdf_file))[0]
    images, texts = get_pdf_images(pdf_file)
    for page_no, (image, text) in enumerate(zip(images, texts)):
        pdf_pages.append(
            {
                "title": title,
                "year": pdf["year"],
                "url": pdf["url"],
                "path": pdf_file,
                "image": image,
                "text": text,
                "page_no": page_no,
            }
        )
```

def get_pdf_images(pdf_path): reader = PdfReader(pdf_path) page_texts = [] for page_number in range(len(reader.pages)): page = reader.pages[page_number] text = page.extract_text() page_texts.append(text)

# Convert to PIL images

images = convert_from_path(pdf_path) assert len(images) == len(page_texts) return images, page_texts pdf_folder = "pdfs" pdf_pages = [] for pdf in tqdm(pdfs): pdf_file = pdf["path"] title = os.path.splitext(os.path.basename(pdf_file))[0] images, texts = get_pdf_images(pdf_file) for page_no, (image, text) in enumerate(zip(images, texts)): pdf_pages.append( { "title": title, "year": pdf["year"], "url": pdf["url"], "path": pdf_file, "image": image, "text": text, "page_no": page_no, } )

In \[ \]:

Copied!

```
len(pdf_pages)
```

len(pdf_pages)

In \[ \]:

Copied!

```
MAX_PAGES = 10  # Set to None to use all pages
pdf_pages = pdf_pages[:MAX_PAGES] if MAX_PAGES else pdf_pages
```

MAX_PAGES = 10 # Set to None to use all pages pdf_pages = pdf_pages[:MAX_PAGES] if MAX_PAGES else pdf_pages

We now have 176 pages, which will be the entity we define as one document in Vespa.

Let us look at the extracted text from the pages displayed above.

In \[ \]:

Copied!

```
pdf_pages[8]["image"]
```

pdf_pages[8]["image"]

In \[ \]:

Copied!

```
print(pdf_pages[8]["text"])
```

print(pdf_pages[8]["text"])

In \[ \]:

Copied!

```
# print(pdf_pages[95]["text"])
```

# print(pdf_pages[95]["text"])

As we can see, the extracted text fails to capture the visual information we see in the image, and it would be difficult for an LLM to correctly answer questions such as *'Price development in Technology sector from April 2023?'* based on the text alone.

## 4. Generate Queries[¶](#4-generate-queries)

In this step, we want to generate queries for each page image. These will be useful for 2 reasons:

1. We can use these queries as typeahead suggestions in the search bar.
1. We could potentially use the queries to generate an evaluation dataset. See [Improving Retrieval with LLM-as-a-judge](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/) for a deeper dive into this topic. This will not be within the scope of this notebook though.

The prompt for generating queries is adapted from [this](https://danielvanstrien.xyz/posts/post-with-code/colpali/2024-09-23-generate_colpali_dataset.html#an-update-retrieval-focused-prompt) wonderful blog post by Daniel van Strien.

We have modified the prompt to also generate keword based queries, in addition to the question based queries.

We will use the Gemini API to generate these queries, with `gemini-flash-lite-latest` as the model.

In \[ \]:

Copied!

```
from pydantic import BaseModel


class GeneratedQueries(BaseModel):
    broad_topical_question: str
    broad_topical_query: str
    specific_detail_question: str
    specific_detail_query: str
    visual_element_question: str
    visual_element_query: str


def get_retrieval_prompt() -> Tuple[str, GeneratedQueries]:
    prompt = (
        prompt
    ) = """You are an investor, stock analyst and financial expert. You will be presented an image of a document page from a report published by the Norwegian Government Pension Fund Global (GPFG). The report may be annual or quarterly reports, or policy reports, on topics such as responsible investment, risk etc.
Your task is to generate retrieval queries and questions that you would use to retrieve this document (or ask based on this document) in a large corpus.
Please generate 3 different types of retrieval queries and questions.
A retrieval query is a keyword based query, made up of 2-5 words, that you would type into a search engine to find this document.
A question is a natural language question that you would ask, for which the document contains the answer.
The queries should be of the following types:
1. A broad topical query: This should cover the main subject of the document.
2. A specific detail query: This should cover a specific detail or aspect of the document.
3. A visual element query: This should cover a visual element of the document, such as a chart, graph, or image.

Important guidelines:
- Ensure the queries are relevant for retrieval tasks, not just describing the page content.
- Use a fact-based natural language style for the questions.
- Frame the queries as if someone is searching for this document in a large corpus.
- Make the queries diverse and representative of different search strategies.

Format your response as a JSON object with the structure of the following example:
{
    "broad_topical_question": "What was the Responsible Investment Policy in 2019?",
    "broad_topical_query": "responsible investment policy 2019",
    "specific_detail_question": "What is the percentage of investments in renewable energy?",
    "specific_detail_query": "renewable energy investments percentage",
    "visual_element_question": "What is the trend of total holding value over time?",
    "visual_element_query": "total holding value trend"
}

If there are no relevant visual elements, provide an empty string for the visual element question and query.
Here is the document image to analyze:
Generate the queries based on this image and provide the response in the specified JSON format.
Only return JSON. Don't return any extra explanation text. """

    return prompt, GeneratedQueries


prompt_text, pydantic_model = get_retrieval_prompt()
```

from pydantic import BaseModel class GeneratedQueries(BaseModel): broad_topical_question: str broad_topical_query: str specific_detail_question: str specific_detail_query: str visual_element_question: str visual_element_query: str def get_retrieval_prompt() -> Tuple\[str, GeneratedQueries\]: prompt = ( prompt ) = """You are an investor, stock analyst and financial expert. You will be presented an image of a document page from a report published by the Norwegian Government Pension Fund Global (GPFG). The report may be annual or quarterly reports, or policy reports, on topics such as responsible investment, risk etc. Your task is to generate retrieval queries and questions that you would use to retrieve this document (or ask based on this document) in a large corpus. Please generate 3 different types of retrieval queries and questions. A retrieval query is a keyword based query, made up of 2-5 words, that you would type into a search engine to find this document. A question is a natural language question that you would ask, for which the document contains the answer. The queries should be of the following types:

1. A broad topical query: This should cover the main subject of the document.
1. A specific detail query: This should cover a specific detail or aspect of the document.
1. A visual element query: This should cover a visual element of the document, such as a chart, graph, or image. Important guidelines:

- Ensure the queries are relevant for retrieval tasks, not just describing the page content.
- Use a fact-based natural language style for the questions.
- Frame the queries as if someone is searching for this document in a large corpus.
- Make the queries diverse and representative of different search strategies. Format your response as a JSON object with the structure of the following example: { "broad_topical_question": "What was the Responsible Investment Policy in 2019?", "broad_topical_query": "responsible investment policy 2019", "specific_detail_question": "What is the percentage of investments in renewable energy?", "specific_detail_query": "renewable energy investments percentage", "visual_element_question": "What is the trend of total holding value over time?", "visual_element_query": "total holding value trend" } If there are no relevant visual elements, provide an empty string for the visual element question and query. Here is the document image to analyze: Generate the queries based on this image and provide the response in the specified JSON format. Only return JSON. Don't return any extra explanation text. """ return prompt, GeneratedQueries prompt_text, pydantic_model = get_retrieval_prompt()

In \[ \]:

Copied!

```
gemini_model = genai.GenerativeModel("gemini-flash-lite-latest")


def generate_queries(image, prompt_text, pydantic_model):
    try:
        response = gemini_model.generate_content(
            [image, "\n\n", prompt_text],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=pydantic_model,
            ),
        )
        queries = json.loads(response.text)
    except Exception as _e:
        print(_e)
        queries = {
            "broad_topical_question": "",
            "broad_topical_query": "",
            "specific_detail_question": "",
            "specific_detail_query": "",
            "visual_element_question": "",
            "visual_element_query": "",
        }
    return queries
```

gemini_model = genai.GenerativeModel("gemini-flash-lite-latest") def generate_queries(image, prompt_text, pydantic_model): try: response = gemini_model.generate_content( [image, "\\n\\n", prompt_text], generation_config=genai.GenerationConfig( response_mime_type="application/json", response_schema=pydantic_model, ), ) queries = json.loads(response.text) except Exception as \_e: print(\_e) queries = { "broad_topical_question": "", "broad_topical_query": "", "specific_detail_question": "", "specific_detail_query": "", "visual_element_question": "", "visual_element_query": "", } return queries

In \[ \]:

Copied!

```
for pdf in tqdm(pdf_pages):
    image = pdf.get("image")
    pdf["queries"] = generate_queries(image, prompt_text, pydantic_model)
```

for pdf in tqdm(pdf_pages): image = pdf.get("image") pdf["queries"] = generate_queries(image, prompt_text, pydantic_model)

Let's take a look at the queries and questions generated for the page displayed above.

In \[ \]:

Copied!

```
pdf_pages[8]["queries"]
```

pdf_pages[8]["queries"]

## 5. Generate embeddings[¶](#5-generate-embeddings)

Now that we have the queries, we can use the ColPali model to generate embeddings for each page image.

In \[ \]:

Copied!

```
def generate_embeddings(images, model, processor, batch_size=1) -> np.ndarray:
    """
    Generate embeddings for a list of images.
    Move to CPU only once per batch.

    Args:
        images (List[PIL.Image]): List of PIL images.
        model (nn.Module): The model to generate embeddings.
        processor: The processor to preprocess images.
        batch_size (int, optional): Batch size for processing. Defaults to 64.

    Returns:
        np.ndarray: Embeddings for the images, shape
                    (len(images), processor.max_patch_length (1030 for ColPali), model.config.hidden_size (Patch embedding dimension - 128 for ColPali)).
    """

    def collate_fn(batch):
        # Batch is a list of images
        return processor.process_images(batch)  # Should return a dict of tensors

    dataloader = DataLoader(
        images,
        shuffle=False,
        collate_fn=collate_fn,
    )

    embeddings_list = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings_batch = model(**batch)
            # Convert tensor to numpy array and append to list
            embeddings_list.extend(
                [t.cpu().numpy() for t in torch.unbind(embeddings_batch)]
            )

    # Stack all embeddings into a single numpy array
    all_embeddings = np.stack(embeddings_list, axis=0)
    return all_embeddings
```

def generate_embeddings(images, model, processor, batch_size=1) -> np.ndarray: """ Generate embeddings for a list of images. Move to CPU only once per batch. Args: images (List[PIL.Image]): List of PIL images. model (nn.Module): The model to generate embeddings. processor: The processor to preprocess images. batch_size (int, optional): Batch size for processing. Defaults to 64. Returns: np.ndarray: Embeddings for the images, shape (len(images), processor.max_patch_length (1030 for ColPali), model.config.hidden_size (Patch embedding dimension - 128 for ColPali)). """ def collate_fn(batch):

# Batch is a list of images

return processor.process_images(batch) # Should return a dict of tensors dataloader = DataLoader( images, shuffle=False, collate_fn=collate_fn, ) embeddings_list = [] for batch in tqdm(dataloader): with torch.no_grad(): batch = {k: v.to(model.device) for k, v in batch.items()} embeddings_batch = model(\*\*batch)

# Convert tensor to numpy array and append to list

embeddings_list.extend( [t.cpu().numpy() for t in torch.unbind(embeddings_batch)] )

# Stack all embeddings into a single numpy array

all_embeddings = np.stack(embeddings_list, axis=0) return all_embeddings

In \[ \]:

Copied!

```
# Generate embeddings for all images
images = [pdf["image"] for pdf in pdf_pages]
embeddings = generate_embeddings(images, model, processor)
```

# Generate embeddings for all images

images = \[pdf["image"] for pdf in pdf_pages\] embeddings = generate_embeddings(images, model, processor)

Now, we have one embedding vector of dimension 128 for each patch of each image (1024 patches + some special tokens).

In \[ \]:

Copied!

```
embeddings.shape
```

embeddings.shape

In \[ \]:

Copied!

```
assert len(pdf_pages) == embeddings.shape[0]
assert embeddings.shape[1] > 1028  # Number of patches (including special tokens)
assert embeddings.shape[2] == 128  # Embedding dimension per patch
```

assert len(pdf_pages) == embeddings.shape[0] assert embeddings.shape[1] > 1028 # Number of patches (including special tokens) assert embeddings.shape[2] == 128 # Embedding dimension per patch

## 6. Prepare Data on Vespa Format[¶](#6-prepare-data-on-vespa-format)

Now, that we have all the data we need, all that remains is to make sure it is in the right format for Vespa.

We now convert the embeddings to Vespa JSON format so we can store (and index) them in Vespa. Details in [Vespa JSON feed format doc](https://docs.vespa.ai/en/reference/document-json-format.html).

We use binary quantization (BQ) of the page level ColPali vector embeddings to reduce their size by 32x.

Read more about binarization of multi-vector representations in the [colbert blog post](https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/).

The binarization step maps 128 dimensional floats to 128 bits, or 16 bytes per vector. Reducing the size by 32x. On the [DocVQA benchmark](https://huggingface.co/datasets/vidore/docvqa_test_subsampled), binarization results in only a small drop in ranking accuracy.

In \[ \]:

Copied!

```
def float_to_binary_embedding(float_query_embedding: dict) -> dict:
    """Utility function to convert float query embeddings to binary query embeddings."""
    binary_query_embeddings = {}
    for k, v in float_query_embedding.items():
        binary_vector = (
            np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
        )
        binary_query_embeddings[k] = binary_vector
    return binary_query_embeddings
```

def float_to_binary_embedding(float_query_embedding: dict) -> dict: """Utility function to convert float query embeddings to binary query embeddings.""" binary_query_embeddings = {} for k, v in float_query_embedding.items(): binary_vector = ( np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist() ) binary_query_embeddings[k] = binary_vector return binary_query_embeddings

We also need a couple of image processing helper functions. These are borrowed from [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark/blob/v4.0.0/src/vidore_benchmark/utils/image_utils.py) repo.

In \[ \]:

Copied!

```
import base64
import io
from pathlib import Path
from typing import Union
from PIL import Image


def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    # Calculate the scaling factor
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))
    return scaled_image


def get_base64_image(img: Union[str, Image.Image], add_url_prefix: bool = True) -> str:
    """
    Convert an image (from a filepath or a PIL.Image object) to a JPEG-base64 string.
    """
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise ValueError("`img` must be a path to an image or a PIL Image object.")

    buffered = io.BytesIO()
    img.save(buffered, format="jpeg")
    b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_data}" if add_url_prefix else b64_data
```

import base64 import io from pathlib import Path from typing import Union from PIL import Image def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image: """ Scale an image to a new height while maintaining the aspect ratio. """

# Calculate the scaling factor

width, height = image.size aspect_ratio = width / height new_width = int(new_height * aspect_ratio)

# Resize the image

scaled_image = image.resize((new_width, new_height)) return scaled_image def get_base64_image(img: Union[str, Image.Image], add_url_prefix: bool = True) -> str: """ Convert an image (from a filepath or a PIL.Image object) to a JPEG-base64 string. """ if isinstance(img, str): img = Image.open(img) elif isinstance(img, Image.Image): pass else: raise ValueError("`img` must be a path to an image or a PIL Image object.") buffered = io.BytesIO() img.save(buffered, format="jpeg") b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8") return f"data:image/jpeg;base64,{b64_data}" if add_url_prefix else b64_data

Note that we also store a scaled down (blurred) version of the image in Vespa. The purpose of this is to return this fast on first results to the frontend, to provide a snappy user experience, and then load the full resolution image async in the background.

In \[ \]:

Copied!

```
vespa_feed = []
for pdf, embedding in zip(pdf_pages, embeddings):
    url = pdf["url"]
    year = pdf["year"]
    title = pdf["title"]
    image = pdf["image"]
    text = pdf.get("text", "")
    page_no = pdf["page_no"]
    query_dict = pdf["queries"]
    questions = [v for k, v in query_dict.items() if "question" in k and v]
    queries = [v for k, v in query_dict.items() if "query" in k and v]
    base_64_image = get_base64_image(
        scale_image(image, 32), add_url_prefix=False
    )  # Scaled down image to return fast on search (~1kb)
    base_64_full_image = get_base64_image(image, add_url_prefix=False)
    embedding_dict = {k: v for k, v in enumerate(embedding)}
    binary_embedding = float_to_binary_embedding(embedding_dict)
    # id_hash should be md5 hash of url and page_number
    id_hash = hashlib.md5(f"{url}_{page_no}".encode()).hexdigest()
    page = {
        "id": id_hash,
        "fields": {
            "id": id_hash,
            "url": url,
            "title": title,
            "year": year,
            "page_number": page_no,
            "blur_image": base_64_image,
            "full_image": base_64_full_image,
            "text": text,
            "embedding": binary_embedding,
            "queries": queries,
            "questions": questions,
        },
    }
    vespa_feed.append(page)
```

vespa_feed = [] for pdf, embedding in zip(pdf_pages, embeddings): url = pdf["url"] year = pdf["year"] title = pdf["title"] image = pdf["image"] text = pdf.get("text", "") page_no = pdf["page_no"] query_dict = pdf["queries"] questions = [v for k, v in query_dict.items() if "question" in k and v] queries = [v for k, v in query_dict.items() if "query" in k and v] base_64_image = get_base64_image( scale_image(image, 32), add_url_prefix=False ) # Scaled down image to return fast on search (~1kb) base_64_full_image = get_base64_image(image, add_url_prefix=False) embedding_dict = {k: v for k, v in enumerate(embedding)} binary_embedding = float_to_binary_embedding(embedding_dict)

# id_hash should be md5 hash of url and page_number

id_hash = hashlib.md5(f"{url}\_{page_no}".encode()).hexdigest() page = { "id": id_hash, "fields": { "id": id_hash, "url": url, "title": title, "year": year, "page_number": page_no, "blur_image": base_64_image, "full_image": base_64_full_image, "text": text, "embedding": binary_embedding, "queries": queries, "questions": questions, }, } vespa_feed.append(page)

### [Optional] Saving the feed file[¶](#optional-saving-the-feed-file)

If you have a large dataset, you can optionally save the file, and feed it using the Vespa CLI, which is more performant than the pyvespa client. See [Feeding to Vespa Cloud](https://vespa-engine.github.io/pyvespa/examples/feed_performance_cloud.md) for more details. Uncomment the cell below if you want to save the feed file.

In \[ \]:

Copied!

```
# os.makedirs("output", exist_ok=True)
# with open("output/vespa_feed.jsonl", "w") as f:
#     vespa_feed_to_save = []
#     for page in vespa_feed:
#         document_id = page["id"]
#         put_id = f"id:{VESPA_APPLICATION_NAME}:{VESPA_SCHEMA_NAME}::{document_id}"
#         vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})
#     json.dump(vespa_feed_to_save, f)
```

# os.makedirs("output", exist_ok=True)

# with open("output/vespa_feed.jsonl", "w") as f:

# vespa_feed_to_save = []

# for page in vespa_feed:

# document_id = page["id"]

# put_id = f"id:{VESPA_APPLICATION_NAME}:{VESPA_SCHEMA_NAME}::{document_id}"

# vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})

# json.dump(vespa_feed_to_save, f)

## 7. Prepare Vespa Application[¶](#7-prepare-vespa-application)

### Configuring the application package[¶](#configuring-the-application-package)

[PyVespa](https://vespa-engine.github.io/pyvespa/) helps us build the [Vespa application package](https://docs.vespa.ai/en/application-packages.html). A Vespa application package consists of configuration files, schemas, models, and code (plugins).

Here are some of the key components of this application package:

1. We store images (and a scaled down version of the image) as a `raw` field.
1. We store the binarized ColPali embeddings as a `tensor<int8>` field.
1. We store the queries and questions as a `array<str>` field.
1. We define 3 different ranking profiles:
   - `default` Uses BM25 for first phase ranking and MaxSim for second phase ranking.
   - `bm25` Uses `bm25(title) + bm25(text)` (first phase only) for ranking.
   - `retrieval-and-rerank` Uses `nearestneighbor` of the query embedding over the document embeddings for retrieval, `binary_max_sim` for first phase ranking, and `max_sim` of the query-embeddings as float for second phase ranking. Vespa's [phased ranking](https://docs.vespa.ai/en/phased-ranking.html) allows us to use different ranking strategies for retrieval and reranking, to choose attractive trade-offs between latency, cost, and accuracy.
1. We also calculate dot product between the query and each document, so that it can be returned with the results, to generate the similarity maps, which show which patches of the image is most similar to the query token embeddings.

First, we define a [Vespa schema](https://docs.vespa.ai/en/schemas.html) with the fields we want to store and their type.

In \[ \]:

Copied!

```
colpali_schema = Schema(
    name=VESPA_SCHEMA_NAME,
    document=Document(
        fields=[
            Field(
                name="id",
                type="string",
                indexing=["summary", "index"],
                match=["word"],
            ),
            Field(name="url", type="string", indexing=["summary", "index"]),
            Field(name="year", type="int", indexing=["summary", "attribute"]),
            Field(
                name="title",
                type="string",
                indexing=["summary", "index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(name="page_number", type="int", indexing=["summary", "attribute"]),
            Field(name="blur_image", type="raw", indexing=["summary"]),
            Field(name="full_image", type="raw", indexing=["summary"]),
            Field(
                name="text",
                type="string",
                indexing=["summary", "index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(
                name="embedding",
                type="tensor<int8>(patch{}, v[16])",
                indexing=[
                    "attribute",
                    "index",
                ],
                ann=HNSW(
                    distance_metric="hamming",
                    max_links_per_node=32,
                    neighbors_to_explore_at_insert=400,
                ),
            ),
            Field(
                name="questions",
                type="array<string>",
                indexing=["summary", "attribute"],
                summary=Summary(fields=["matched-elements-only"]),
            ),
            Field(
                name="queries",
                type="array<string>",
                indexing=["summary", "attribute"],
                summary=Summary(fields=["matched-elements-only"]),
            ),
        ]
    ),
    fieldsets=[
        FieldSet(
            name="default",
            fields=["title", "text"],
        ),
    ],
    document_summaries=[
        DocumentSummary(
            name="default",
            summary_fields=[
                Summary(
                    name="text",
                    fields=[("bolding", "on")],
                ),
                Summary(
                    name="snippet",
                    fields=[("source", "text"), "dynamic"],
                ),
            ],
            from_disk=True,
        ),
        DocumentSummary(
            name="suggestions",
            summary_fields=[
                Summary(name="questions"),
            ],
            from_disk=True,
        ),
    ],
)

# Define similarity functions used in all rank profiles
mapfunctions = [
    Function(
        name="similarities",  # computes similarity scores between each query token and image patch
        expression="""
                sum(
                    query(qt) * unpack_bits(attribute(embedding)), v
                )
            """,
    ),
    Function(
        name="normalized",  # normalizes the similarity scores to [-1, 1]
        expression="""
                (similarities - reduce(similarities, min)) / (reduce((similarities - reduce(similarities, min)), max)) * 2 - 1
            """,
    ),
    Function(
        name="quantized",  # quantizes the normalized similarity scores to signed 8-bit integers [-128, 127]
        expression="""
                cell_cast(normalized * 127.999, int8)
            """,
    ),
]

# Define the 'bm25' rank profile
bm25 = RankProfile(
    name="bm25",
    inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
    first_phase="bm25(title) + bm25(text)",
    functions=mapfunctions,
)


# A function to create an inherited rank profile which also returns quantized similarity scores
def with_quantized_similarity(rank_profile: RankProfile) -> RankProfile:
    return RankProfile(
        name=f"{rank_profile.name}_sim",
        first_phase=rank_profile.first_phase,
        inherits=rank_profile.name,
        summary_features=["quantized"],
    )


colpali_schema.add_rank_profile(bm25)
colpali_schema.add_rank_profile(with_quantized_similarity(bm25))


# Update the 'colpali' rank profile
input_query_tensors = []
MAX_QUERY_TERMS = 64
for i in range(MAX_QUERY_TERMS):
    input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

input_query_tensors.extend(
    [
        ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
        ("query(qtb)", "tensor<int8>(querytoken{}, v[16])"),
    ]
)

colpali = RankProfile(
    name="colpali",
    inputs=input_query_tensors,
    first_phase="max_sim_binary",
    second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    functions=mapfunctions
    + [
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(embedding)), v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
        Function(
            name="max_sim_binary",
            expression="""
                sum(
                    reduce(
                        1 / (1 + sum(
                            hamming(query(qtb), attribute(embedding)), v)
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
    ],
)
colpali_schema.add_rank_profile(colpali)
colpali_schema.add_rank_profile(with_quantized_similarity(colpali))

# Update the 'hybrid' rank profile
hybrid = RankProfile(
    name="hybrid",
    inputs=input_query_tensors,
    first_phase="max_sim_binary",
    second_phase=SecondPhaseRanking(
        expression="max_sim + 2 * (bm25(text) + bm25(title))", rerank_count=10
    ),
    functions=mapfunctions
    + [
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(embedding)), v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
        Function(
            name="max_sim_binary",
            expression="""
                sum(
                    reduce(
                        1 / (1 + sum(
                            hamming(query(qtb), attribute(embedding)), v)
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
    ],
)
colpali_schema.add_rank_profile(hybrid)
colpali_schema.add_rank_profile(with_quantized_similarity(hybrid))
```

colpali_schema = Schema( name=VESPA_SCHEMA_NAME, document=Document( fields=\[ Field( name="id", type="string", indexing=["summary", "index"], match=["word"], ), Field(name="url", type="string", indexing=["summary", "index"]), Field(name="year", type="int", indexing=["summary", "attribute"]), Field( name="title", type="string", indexing=["summary", "index"], match=["text"], index="enable-bm25", ), Field(name="page_number", type="int", indexing=["summary", "attribute"]), Field(name="blur_image", type="raw", indexing=["summary"]), Field(name="full_image", type="raw", indexing=["summary"]), Field( name="text", type="string", indexing=["summary", "index"], match=["text"], index="enable-bm25", ), Field( name="embedding", type="tensor<int8>(patch{}, v[16])", indexing=[ "attribute", "index", ], ann=HNSW( distance_metric="hamming", max_links_per_node=32, neighbors_to_explore_at_insert=400, ), ), Field( name="questions", type="array<string>", indexing=["summary", "attribute"], summary=Summary(fields=["matched-elements-only"]), ), Field( name="queries", type="array<string>", indexing=["summary", "attribute"], summary=Summary(fields=["matched-elements-only"]), ), \] ), fieldsets=\[ FieldSet( name="default", fields=["title", "text"], ), \], document_summaries=\[ DocumentSummary( name="default", summary_fields=\[ Summary( name="text", fields=[("bolding", "on")], ), Summary( name="snippet", fields=[("source", "text"), "dynamic"], ), \], from_disk=True, ), DocumentSummary( name="suggestions", summary_fields=[ Summary(name="questions"), ], from_disk=True, ), \], )

# Define similarity functions used in all rank profiles

mapfunctions = \[ Function( name="similarities", # computes similarity scores between each query token and image patch expression=""" sum( query(qt) * unpack_bits(attribute(embedding)), v ) """, ), Function( name="normalized", # normalizes the similarity scores to [-1, 1] expression=""" (similarities - reduce(similarities, min)) / (reduce((similarities - reduce(similarities, min)), max)) * 2 - 1 """, ), Function( name="quantized", # quantizes the normalized similarity scores to signed 8-bit integers [-128, 127] expression=""" cell_cast(normalized * 127.999, int8) """, ), \]

# Define the 'bm25' rank profile

bm25 = RankProfile( name="bm25", inputs=\[("query(qt)", "tensor<float>(querytoken{}, v[128])")\], first_phase="bm25(title) + bm25(text)", functions=mapfunctions, )

# A function to create an inherited rank profile which also returns quantized similarity scores

def with_quantized_similarity(rank_profile: RankProfile) -> RankProfile: return RankProfile( name=f"{rank_profile.name}\_sim", first_phase=rank_profile.first_phase, inherits=rank_profile.name, summary_features=["quantized"], ) colpali_schema.add_rank_profile(bm25) colpali_schema.add_rank_profile(with_quantized_similarity(bm25))

# Update the 'colpali' rank profile

input_query_tensors = [] MAX_QUERY_TERMS = 64 for i in range(MAX_QUERY_TERMS): input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])")) input_query_tensors.extend( \[ ("query(qt)", "tensor<float>(querytoken{}, v[128])"), ("query(qtb)", "tensor<int8>(querytoken{}, v[16])"), \] ) colpali = RankProfile( name="colpali", inputs=input_query_tensors, first_phase="max_sim_binary", second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10), functions=mapfunctions

- [ Function( name="max_sim", expression=""" sum( reduce( sum( query(qt) * unpack_bits(attribute(embedding)), v ), max, patch ), querytoken ) """, ), Function( name="max_sim_binary", expression=""" sum( reduce( 1 / (1 + sum( hamming(query(qtb), attribute(embedding)), v) ), max, patch ), querytoken ) """, ), ], ) colpali_schema.add_rank_profile(colpali) colpali_schema.add_rank_profile(with_quantized_similarity(colpali))

# Update the 'hybrid' rank profile

hybrid = RankProfile( name="hybrid", inputs=input_query_tensors, first_phase="max_sim_binary", second_phase=SecondPhaseRanking( expression="max_sim + 2 * (bm25(text) + bm25(title))", rerank_count=10 ), functions=mapfunctions

- [ Function( name="max_sim", expression=""" sum( reduce( sum( query(qt) * unpack_bits(attribute(embedding)), v ), max, patch ), querytoken ) """, ), Function( name="max_sim_binary", expression=""" sum( reduce( 1 / (1 + sum( hamming(query(qtb), attribute(embedding)), v) ), max, patch ), querytoken ) """, ), ], ) colpali_schema.add_rank_profile(hybrid) colpali_schema.add_rank_profile(with_quantized_similarity(hybrid))

### Configuring the `services.xml`[¶](#configuring-the-servicesxml)

[services.xml](https://docs.vespa.ai/en/reference/services.html) is the primary configuration file for a Vespa application, with a plethora of options to configure the application.

Since `pyvespa` version `0.50.0`, these configuration options are also available in `pyvespa`. See [Pyvespa - Advanced configuration](https://vespa-engine.github.io/pyvespa/advanced-configuration.md) for more details. (Note that configurating this is optional, and pyvespa will use basic defaults for you if you opt out).

We will use the advanced configuration to configure up [dynamic snippets](https://docs.vespa.ai/en/document-summaries.html#dynamic-snippets). This allows us to highlight matched terms in the search results and generate a `snippet` to display, rather than the full text of the document.

In \[ \]:

Copied!

```
from vespa.configuration.services import (
    services,
    container,
    search,
    document_api,
    document_processing,
    clients,
    client,
    config,
    content,
    redundancy,
    documents,
    node,
    certificate,
    token,
    document,
    nodes,
)
from vespa.configuration.vt import vt
from vespa.package import ServicesConfiguration

service_config = ServicesConfiguration(
    application_name=VESPA_APPLICATION_NAME,
    services_config=services(
        container(
            search(),
            document_api(),
            document_processing(),
            clients(
                client(
                    certificate(file="security/clients.pem"),
                    id="mtls",
                    permissions="read,write",
                ),
                client(
                    token(id=f"{VESPA_TOKEN_ID}"),
                    id="token_write",
                    permissions="read,write",
                ),
            ),
            config(
                vt("tag")(
                    vt("bold")(
                        vt("open", "<strong>"),
                        vt("close", "</strong>"),
                    ),
                    vt("separator", "..."),
                ),
                name="container.qr-searchers",
            ),
            id=f"{VESPA_APPLICATION_NAME}_container",
            version="1.0",
        ),
        content(
            redundancy("1"),
            documents(document(type="pdf_page", mode="index")),
            nodes(node(distribution_key="0", hostalias="node1")),
            config(
                vt("max_matches", "2", replace_underscores=False),
                vt("length", "1000"),
                vt("surround_max", "500", replace_underscores=False),
                vt("min_length", "300", replace_underscores=False),
                name="vespa.config.search.summary.juniperrc",
            ),
            id=f"{VESPA_APPLICATION_NAME}_content",
            version="1.0",
        ),
        version="1.0",
    ),
)
```

from vespa.configuration.services import ( services, container, search, document_api, document_processing, clients, client, config, content, redundancy, documents, node, certificate, token, document, nodes, ) from vespa.configuration.vt import vt from vespa.package import ServicesConfiguration service_config = ServicesConfiguration( application_name=VESPA_APPLICATION_NAME, services_config=services( container( search(), document_api(), document_processing(), clients( client( certificate(file="security/clients.pem"), id="mtls", permissions="read,write", ), client( token(id=f"{VESPA_TOKEN_ID}"), id="token_write", permissions="read,write", ), ), config( vt("tag")( vt("bold")( vt("open", "<strong>"), vt("close", "</strong>"), ), vt("separator", "..."), ), name="container.qr-searchers", ), id=f"{VESPA_APPLICATION_NAME}\_container", version="1.0", ), content( redundancy("1"), documents(document(type="pdf_page", mode="index")), nodes(node(distribution_key="0", hostalias="node1")), config( vt("max_matches", "2", replace_underscores=False), vt("length", "1000"), vt("surround_max", "500", replace_underscores=False), vt("min_length", "300", replace_underscores=False), name="vespa.config.search.summary.juniperrc", ), id=f"{VESPA_APPLICATION_NAME}\_content", version="1.0", ), version="1.0", ), )

In \[ \]:

Copied!

```
# Create the Vespa application package
vespa_application_package = ApplicationPackage(
    name=VESPA_APPLICATION_NAME,
    schema=[colpali_schema],
    services_config=service_config,
)
```

# Create the Vespa application package

vespa_application_package = ApplicationPackage( name=VESPA_APPLICATION_NAME, schema=[colpali_schema], services_config=service_config, )

## 8. Deploy Vespa Application[¶](#8-deploy-vespa-application)

In \[ \]:

Copied!

```
# This is only needed for CI.
VESPA_TEAM_API_KEY = os.getenv("VESPA_TEAM_API_KEY", None)
```

# This is only needed for CI.

VESPA_TEAM_API_KEY = os.getenv("VESPA_TEAM_API_KEY", None)

In \[ \]:

Copied!

```
vespa_cloud = VespaCloud(
    tenant=VESPA_TENANT_NAME,
    application=VESPA_APPLICATION_NAME,
    key_content=VESPA_TEAM_API_KEY,
    application_package=vespa_application_package,
)

# Deploy the application
vespa_cloud.deploy()

# Output the endpoint URL
endpoint_url = vespa_cloud.get_token_endpoint()
print(f"Application deployed. Token endpoint URL: {endpoint_url}")
```

vespa_cloud = VespaCloud( tenant=VESPA_TENANT_NAME, application=VESPA_APPLICATION_NAME, key_content=VESPA_TEAM_API_KEY, application_package=vespa_application_package, )

# Deploy the application

vespa_cloud.deploy()

# Output the endpoint URL

endpoint_url = vespa_cloud.get_token_endpoint() print(f"Application deployed. Token endpoint URL: {endpoint_url}")

Make sure to take note of the token endpoint_url. You need to put this in your `.env` file for your web application - `VESPA_APP_TOKEN_URL=https://abcd.vespa-app.cloud` - to access the Vespa application from your web application.

## 9. Feed Data to Vespa[¶](#9-feed-data-to-vespa)

We will need the `enpdoint_url` and `colpalidemo_write` token to feed the data to the Vespa application.

In \[ \]:

Copied!

```
# Instantiate Vespa connection using token
app = Vespa(url=endpoint_url, vespa_cloud_secret_token=VESPA_CLOUD_SECRET_TOKEN)
app.get_application_status()
```

# Instantiate Vespa connection using token

app = Vespa(url=endpoint_url, vespa_cloud_secret_token=VESPA_CLOUD_SECRET_TOKEN) app.get_application_status()

Now, let us feed the data to Vespa. If you have a large dataset, you could also do this async, with `feed_async_iterable()`, see [Feeding Vespa cloud](https://vespa-engine.github.io/pyvespa/examples/feed_performance_cloud.md) for a detailed comparison.

In \[ \]:

Copied!

```
def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
        )


# Feed data into Vespa synchronously
app.feed_iterable(vespa_feed, schema=VESPA_SCHEMA_NAME, callback=callback)
```

def callback(response: VespaResponse, id: str): if not response.is_successful(): print( f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}" )

# Feed data into Vespa synchronously

app.feed_iterable(vespa_feed, schema=VESPA_SCHEMA_NAME, callback=callback)

## 10. Test a query to the Vespa application[¶](#10-test-a-query-to-the-vespa-application)

For now, we will just run a query with the default rank profile. We will need a utility function to generate embeddings for the query, and pass this to Vespa to use for calculating MaxSim. In the web application, we also provide function to generate binary embeddings, allowing the user to choose different rank profiles at query time.

In \[ \]:

Copied!

```
query = "Price development in Technology sector from April 2023?"
```

query = "Price development in Technology sector from April 2023?"

In \[ \]:

Copied!

```
def get_q_embs_vespa_format(query: str):
    inputs = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        embeddings_query = model(**inputs)
        q_embs = embeddings_query.to("cpu")[0]  # Extract the single embedding
    return {idx: emb.tolist() for idx, emb in enumerate(q_embs)}
```

def get_q_embs_vespa_format(query: str): inputs = processor.process_queries([query]).to(model.device) with torch.no_grad(): embeddings_query = model(\*\*inputs) q_embs = embeddings_query.to("cpu")[0] # Extract the single embedding return {idx: emb.tolist() for idx, emb in enumerate(q_embs)}

In \[ \]:

Copied!

```
q_emb = get_q_embs_vespa_format(query)
```

q_emb = get_q_embs_vespa_format(query)

In \[ \]:

Copied!

```
with app.syncio() as sess:
    response = sess.query(
        body={
            "yql": (
                f"select id, url, title, year, full_image, quantized  from {VESPA_SCHEMA_NAME} where userQuery();"
            ),
            "ranking": "default",
            "query": query,
            "timeout": "10s",
            "hits": 3,
            "input.query(qt)": q_emb,
            "presentation.timing": True,
        }
    )
```

with app.syncio() as sess: response = sess.query( body={ "yql": ( f"select id, url, title, year, full_image, quantized from {VESPA_SCHEMA_NAME} where userQuery();" ), "ranking": "default", "query": query, "timeout": "10s", "hits": 3, "input.query(qt)": q_emb, "presentation.timing": True, } )

In \[ \]:

Copied!

```
assert len(response.json["root"]["children"]) == 3
```

assert len(response.json["root"]["children"]) == 3

Great. You have now deployed the Vespa application and fed the data to it, and made sure you are able to query it using the vespa endpoint and a token.

### Saving the generated key/cert files[¶](#saving-the-generated-keycert-files)

A key and cert file is generated for you as an alternative to using tokens for authentication. We advise you to save these files in a secure location, in case you want to use them for authentication in the future.

In \[ \]:

Copied!

```
key_path = Path(
    f"~/.vespa/{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.default/data-plane-private-key.pem"
).expanduser()
cert_path = Path(
    f"~/.vespa/{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.default/data-plane-public-cert.pem"
).expanduser()

assert key_path.exists(), cert_path.exists()
```

key_path = Path( f"~/.vespa/{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.default/data-plane-private-key.pem" ).expanduser() cert_path = Path( f"~/.vespa/{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.default/data-plane-public-cert.pem" ).expanduser() assert key_path.exists(), cert_path.exists()

## 11. Deploying your web app[¶](#11-deploying-your-web-app)

To deploy a frontend to let users interact with the Vespa application. you can clone the sample app from [sample-apps repo](https://github.com/vespa-engine/sample-apps/blob/master/visual-retrieval-colpali/README.md). It includes instructions for running and connecting your web application to your vespa app.

In \[ \]:

Copied!

```
!git clone --depth 1 --filter=blob:none --sparse https://github.com/vespa-engine/sample-apps.git src && cd src && git sparse-checkout set visual-retrieval-colpali
```

!git clone --depth 1 --filter=blob:none --sparse https://github.com/vespa-engine/sample-apps.git src && cd src && git sparse-checkout set visual-retrieval-colpali

Now, you have the code for the webapp in your `src/visual-retrieval-colpali`-directory

In \[ \]:

Copied!

```
os.listdir("src/visual-retrieval-colpali")
```

os.listdir("src/visual-retrieval-colpali")

### Setting environment variables for your web app[¶](#setting-environment-variables-for-your-web-app)

Now, you need to set the following variables in the `src/.env.example`-file:

```
VESPA_APP_TOKEN_URL=https://abcde.z.vespa-app.cloud # Your token endpoint url you got after deploying your Vespa app.
VESPA_CLOUD_SECRET_TOKEN=vespa_cloud_xxxxxxxx # The value of the token that your created in this notebook. 
GEMINI_API_KEY=your_api_key # The same as GOOGLE_API_KEY in this notebook
HF_TOKEN=hf_xxxx # If you want to deploy your web app to huggingface spaces - https://huggingface.co/settings/tokens
```

After, that, rename your file to .env.

In \[ \]:

Copied!

```
# rename src/visual-retrieval-colpali/.env.example
os.rename(
    "src/visual-retrieval-colpali/.env.example", dst="src/visual-retrieval-colpali/.env"
)
```

# rename src/visual-retrieval-colpali/.env.example

os.rename( "src/visual-retrieval-colpali/.env.example", dst="src/visual-retrieval-colpali/.env" )

And you're ready to spin up your web app locally, and deploy to huggingface spaces if you want. Navigate to `src/visual-retrieval-colpali/` directory and follow the instructions in the `README.md` to continue. 🚀

## Cleanup[¶](#cleanup)

As this notebook runs in CI, we will delete the Vespa application after running the notebook. DO NOT run the cell below unless you are sure you want to delete the Vespa application.

In \[ \]:

Copied!

```
if os.getenv("CI", "false") == "true":
    vespa_cloud.delete()
```

if os.getenv("CI", "false") == "true": vespa_cloud.delete()
