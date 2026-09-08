import os
import tempfile
from typing import List, Tuple

import faiss 
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF, build a FAISS vector index, and ask questions using Groq.")


# -----------------------------
# Models
# -----------------------------
@st.cache_resource
def load_embedding_model():
    # Open-source embedding model.
    # SentenceTransformers handles tokenization internally.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


EMBEDDING_MODEL = load_embedding_model()

# Groq model. GPT-OSS is an open-weight model hosted by Groq.
DEFAULT_LLM = "openai/gpt-oss-20b"


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n\n".join(pages).strip()


# -----------------------------
# Chunking
# -----------------------------
def create_chunks(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[str]:
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_faiss_index(chunks: List[str]):
    embeddings = EMBEDDING_MODEL.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve_chunks(
    question: str,
    index,
    chunks: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    query_embedding = EMBEDDING_MODEL.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((chunks[int(idx)], float(score)))

    return results


# -----------------------------
# Groq answer generation
# -----------------------------
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        return None

    return Groq(api_key=api_key)


def generate_answer(question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets."
        )

    context = "\n\n".join(
        f"[Source {i + 1}]\n{chunk}"
        for i, (chunk, _) in enumerate(retrieved_chunks)
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, clearly say:
"I couldn't find that information in the uploaded PDF."

Do not invent facts.
Keep the answer clear and reasonably concise.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=DEFAULT_LLM,
        messages=[
            {
                "role": "system",
                "content": "You answer questions from retrieved document context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=5,
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=400,
        max_value=1600,
        value=900,
        step=100,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=300,
        value=150,
        step=50,
    )

    st.divider()

    st.write("**LLM:**", DEFAULT_LLM)
    st.write("**Embeddings:** all-MiniLM-L6-v2")
    st.write("**Vector DB:** FAISS")


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        with st.spinner("Reading PDF..."):
            text = extract_pdf_text(uploaded_file)

        if not text:
            st.error(
                "No extractable text was found. This app currently works with "
                "text-based PDFs; scanned/image-only PDFs need OCR."
            )
        else:
            chunks = create_chunks(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            with st.spinner("Creating embeddings and FAISS index..."):
                index = create_faiss_index(chunks)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []

            st.success(
                f"Indexed **{uploaded_file.name}** — "
                f"{len(chunks)} chunks created."
            )


# -----------------------------
# Chat
# -----------------------------
if st.session_state.index is None:
    st.info("Upload a PDF above to start asking questions.")
else:
    st.subheader(f"💬 Ask questions about: {st.session_state.file_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and generating an answer..."):
                try:
                    retrieved = retrieve_chunks(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        top_k=top_k,
                    )

                    answer = generate_answer(question, retrieved)
                    st.markdown(answer)

                    with st.expander("🔎 Retrieved context"):
                        for i, (chunk, score) in enumerate(retrieved, 1):
                            st.markdown(
                                f"**Chunk {i} — similarity: {score:.3f}**"
                            )
                            st.write(chunk)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"Error: {e}")


st.divider()
st.caption(
    "Pipeline: PDF → text extraction → chunks → tokenization inside "
    "SentenceTransformers → embeddings → FAISS → retrieval → Groq LLM."
)
