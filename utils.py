import os
from typing import List
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

def extract_text_from_pdfs(pdf_docs) -> str:
    """
    Extracts and merges plain text from a list of uploaded PDF files.
    """
    raw_text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
        except Exception as e:
            raise RuntimeError(f"Error parsing file {pdf.name}: {str(e)}")
    return raw_text


def get_text_chunks(text: str) -> List[str]:
    """
    Splits a single string of text into smaller, overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks: List[str]) -> FAISS:
    """
    Generates sentence-transformer embeddings and indexes them into a FAISS vector store.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to generate vector store: {str(e)}")


def generate_gemini_response(context: str, question: str) -> str:
    """
    Sends the compiled context and user question to the gemini-2.5-flash model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment configurations.")

    # Initialize the official Google GenAI client
    client = genai.Client(api_key=api_key)

    # Contextual prompt ensuring strict grounding
    system_prompt = (
        "You are a precise and helpful document analysis assistant.\n"
        "Answer the user's question using only the facts provided in the context section below.\n"
        "If the answer cannot be found or reasonably inferred from the context, state clearly: "
        "'I cannot find the answer to that question in the provided documents.'\n"
        "Do not make up facts, rely on outside knowledge, or extrapolate beyond the text.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt,
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"