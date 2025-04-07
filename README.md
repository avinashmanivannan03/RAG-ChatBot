# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for interacting with PDF documents using LLaMA-3 and FAISS vector search.

## Overview

This project implements a question-answering system that enables users to upload PDF documents and ask questions about their content. The system uses a RAG architecture to extract relevant passages from the document and generate accurate answers using the Meta-LLaMA-3-8B-Instruct-Turbo model.

## Features

- PDF text extraction with multiple fallback methods (PyPDF, OCR, PyMuPDF)
- Text chunking and embedding for efficient retrieval
- FAISS vector database for similarity search
- Integration with LLaMA-3 for natural language understanding and response generation
- Streamlit web interface for easy interaction
- Command-line interface option

## Tech Stack

- **Frontend**: Streamlit
- **Text Extraction**:
  - PyPDFLoader (primary)
  - Pytesseract + pdf2image (OCR fallback)
  - PyMuPDF (secondary fallback)
- **Text Processing**: LangChain for text splitting and document management
- **Embeddings**:
  - HuggingFace Embeddings (all-MiniLM-L6-v2)
  - Together Embeddings (togethercomputer/m2-bert-80M-32k-retrieval)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **LLM API**: Together API (using Meta-LLaMA-3-8B-Instruct-Turbo)

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR engine
- Poppler (for pdf2image)

### Setup

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd RAG_Chatbot
   ```

2. Install system dependencies (Linux/Ubuntu):
   ```bash
   sudo apt-get update
   sudo apt-get install -y poppler-utils tesseract-ocr
   ```

   For macOS:
   ```bash
   brew install poppler tesseract
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Together API key:
   ```
   TOGETHER_API_KEY=your_api_key_here
   ```

## Usage

### Streamlit Web Interface

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a PDF document using the file uploader

4. Ask questions about the document in the text input field

### Command Line Interface

Alternatively, you can use the command-line interface:

1. Run the script:
   ```bash
   python rag_chatbot_without_ui.py
   ```

2. Upload a PDF when prompted

3. Type your questions when prompted, or type 'exit' to quit

## How It Works

### PDF Processing Pipeline

1. **Document Upload**: The system accepts PDF files through the Streamlit interface or command-line.

2. **Text Extraction**: The system attempts to extract text using three methods in sequence:
   - PyPDFLoader (primary approach for text-based PDFs)
   - OCR with pytesseract (for scanned or image-based PDFs)
   - PyMuPDF (fallback method)

3. **Text Chunking**: The extracted text is split into smaller chunks (500 characters with 50-character overlap) for more effective retrieval.

4. **Embedding Generation**: Each chunk is converted into numerical vector embeddings using the specified embedding model.

5. **Vector Database Creation**: The embeddings are stored in a FAISS index for efficient similarity search.

### Query Processing Pipeline

1. **User Query**: The user submits a question about the document.

2. **Retrieval**: The system converts the question into an embedding and retrieves the top k most relevant chunks from the document.

3. **Context Formation**: The retrieved chunks are combined to form a context for the LLM.

4. **Response Generation**: The LLaMA-3 model generates an answer based on the provided context and question.

## Technical Details

### Embedding Models

Two embedding models are used in different parts of the codebase:

1. **HuggingFace Embeddings (all-MiniLM-L6-v2)**: A lightweight but effective embedding model used in the non-UI and fallback implementations.

2. **Together Embeddings (m2-bert-80M-32k-retrieval)**: A more powerful embedding model with longer context support, used in the main Streamlit app.

### Chunking Strategy

Documents are split using LangChain's RecursiveCharacterTextSplitter with:
- Chunk size: 500 characters
- Chunk overlap: 50 characters

This balances the need for context preservation while keeping chunks small enough for efficient retrieval.

### LLM Prompt Engineering

The system prompt instructs the LLaMA-3 model to:
- Only answer based on the provided context
- Respond with "No relevant information found in the document" when the answer isn't in the context

This helps ensure that answers are grounded in the document and prevents hallucination.

## Challenges and Solutions

### Challenge 1: Handling Different PDF Types

**Problem**: PDFs can be text-based, image-based, or a combination, making text extraction inconsistent.

**Solution**: Implemented a robust extraction pipeline with multiple fallback methods:
- First try PyPDFLoader for text-based PDFs
- If that fails or returns empty content, try OCR with pdf2image and pytesseract
- If OCR fails, try PyMuPDF as a final fallback

### Challenge 2: Balancing Retrieval Precision

**Problem**: Finding the optimal number of chunks to retrieve for context while keeping responses accurate.

**Solution**: 
- Used a top-k retrieval approach (k=3 in the non-UI version, k=5 in the Streamlit app)
- Implemented chunk overlap to maintain context across chunk boundaries
- Kept chunk sizes relatively small (500 characters) to improve retrieval precision

### Challenge 3: Response Quality

**Problem**: Ensuring generated responses remain faithful to the document content.

**Solution**:
- Added explicit system prompts to constrain the LLM to only use the provided context
- Included instructions for the model to explicitly state when information isn't found in the document
- Used temperature=0.5 for a balance between creativity and accuracy

## Future Improvements

- Add support for multiple document uploads
- Implement document caching to avoid reprocessing
- Add conversation history for multi-turn interactions
- Support for more document formats (DOCX, TXT, etc.)
- Implement metadata filtering for more targeted retrieval
- Add evaluation metrics for response quality



## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for document processing tools
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Together AI](https://www.together.ai/) for LLaMA-3 API access
- [Streamlit](https://streamlit.io/) for the web interface
