# -*- coding: utf-8 -*-
"""RAG_CHATBOT_without_UI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fomRbbuTUVOD1Tm8X7OLYp-9tduEJBLP

## **INSTALL REQUIRED PACKAGES**
"""

!pip install together openai langchain tiktoken faiss-cpu python-dotenv

pip install -U langchain-community

!apt install poppler-utils
!pip install pdf2image pytesseract pillow

!pip install pypdf

!apt install poppler-utils tesseract-ocr -y
!pip install pdf2image pytesseract together openai langchain faiss-cpu tiktoken python-dotenv

!pip install -q together openai tiktoken langchain chromadb

"""# **IMPORT AND SETUP**"""

import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import together
from google.colab import files
import warnings
warnings.filterwarnings("ignore")


os.environ['TOGETHER_API_KEY'] = "tgp_v1_hLcY1a2QF8ynIFQE6QSuDujFr65xF6CckK_DzhMroIE"
together.api_key = os.environ['TOGETHER_API_KEY']

"""# **UPLOAD A DOCUMENT**"""

uploaded = files.upload()
pdf_path = next(iter(uploaded))
print(f"Uploaded: {pdf_path}")

"""## **EXTRACTING TEXT FROM DOCUMENT**"""

def extract_text_from_pdf(pdf_path):
    try:
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        if all(len(p.page_content.strip()) == 0 for p in pages):
            raise ValueError("Empty pages from loader.")
        print("Extracted using PyPDFLoader")
        return [Document(page_content=p.page_content) for p in pages]
    except:
        print("Falling back to OCR (image-based PDF)...")
        images = convert_from_path(pdf_path)
        texts = []
        for img in images:
            text = pytesseract.image_to_string(img)
            texts.append(Document(page_content=text))
        print("Extracted using OCR")
        return texts

docs = extract_text_from_pdf(pdf_path)

"""# **CHUNKING THE DOCUMENT**"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(docs)
print(f"Total Chunks: {len(docs)}")

"""# **EMBEDDINGS + STORING IN FAISS**"""

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db=FAISS.from_documents(docs, embedding_model)

"""# **RETRIEVAL & LLAMA-3 CHAT**"""

!pip install -q together
import together

def get_top_k_chunks(query, k=3):
    results = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def generate_llama_response(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the given context to answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]


    client = together.Client(api_key=os.environ.get("TOGETHER_API_KEY"))

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=messages,
        max_tokens=512,
        temperature=0.5,
        top_p=0.9
    )

    return response.choices[0].message.content.strip()

"""# **USER INPUT**"""

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    context = get_top_k_chunks(query)
    response = generate_llama_response(query, context)
    print("\nAnswer:")
    print(response)
    print("-"*50)

