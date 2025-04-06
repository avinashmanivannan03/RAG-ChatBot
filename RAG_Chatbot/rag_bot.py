import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import fitz  # PyMuPDF
import traceback



def extract_text_from_pdf(pdf_path):
    try:
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        if all(len(p.page_content.strip()) == 0 for p in pages):
            raise ValueError("Empty pages from PyPDFLoader")
        print("Extracted using PyPDFLoader")
        return [Document(page_content=p.page_content) for p in pages]
    except Exception as e:
        print(f"PyPDFLoader failed: {e}")

    try:
        print("Trying OCR (pdf2image + pytesseract)...")
        images = convert_from_path(pdf_path)
        docs = [Document(page_content=pytesseract.image_to_string(img)) for img in images]
        if all(len(doc.page_content.strip()) == 0 for doc in docs):
            raise ValueError("OCR returned empty text.")
        print("Extracted using OCR")
        return docs
    except Exception as e:
        print(f"OCR failed: {e}")

    try:
        print("Trying fallback: PyMuPDF...")
        doc = fitz.open(pdf_path)
        pages = [Document(page_content=page.get_text()) for page in doc]
        print("Extracted using PyMuPDF")
        return pages
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
        traceback.print_exc()

    return []


def prepare_db(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding_model)
    return db, chunks



def get_top_k_chunks(query, db, k=3):
    results = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def generate_llama_response(query, context, api_key):
    import together
    together.api_key = api_key

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the given context to answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]

    client = together.Client(api_key=api_key)

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=messages,
        max_tokens=512,
        temperature=0.5,
        top_p=0.9
    )

    return response.choices[0].message.content.strip()
