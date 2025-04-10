import os
import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
#from langchain_community.embeddings import TogetherEmbeddings
from langchain_together import TogetherEmbeddings
import together
import warnings
warnings.filterwarnings("ignore")

os.environ['HF_HOME'] = './hf_cache'

from dotenv import load_dotenv
load_dotenv()
os.environ['TOGETHER_API_KEY'] = os.getenv("TOGETHER_API_KEY")
together.api_key = os.environ['TOGETHER_API_KEY']

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("RAG Chatbot for PDF (LLaMA + FAISS)")

#PDF Extraction Function 
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


#LLaMA Response Generator
def generate_llama_response(query, context):
    messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "ONLY answer based on the given context. "
            "If the answer is not in the context, respond with: 'No relevant information found in the document.'"
        )
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{query}"
    }
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

#Chunk Retriever
def get_top_k_chunks(query, db, k=5):
    results = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

#Streamlit UI
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded successfully!")

    with st.spinner("Extracting and indexing text..."):
        docs = extract_text_from_pdf("temp.pdf")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(docs)
        embedding_model = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-32k-retrieval",
            together_api_key=os.environ["TOGETHER_API_KEY"]
        )

        db = FAISS.from_documents(docs, embedding_model)

    st.success(f"Text indexed. Total chunks: {len(docs)}")

    query = st.text_input("Ask a question based on the document:")
    if st.button("Get Answer") and query:
        with st.spinner("Generating answer..."):
            context = get_top_k_chunks(query, db)
            response = generate_llama_response(query, context)
        st.markdown("Answer:")
        st.write(response)

