

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings

PERSIST_DIRECTORY = "./chroma_db"
PDF_PATH = "./data/document.pdf"
DEFAULT_MODEL = "gemma:2b"


@st.cache_data(show_spinner=False)
def load_pdf_pages(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Could not find the PDF at `{pdf_path}`. Please place the document before running the app."
        )
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def is_safe(prompt: str) -> bool:
    forbidden_words = [
        "illegal",
        "harmful",
        "unethical",
        "dangerous",
        "violent",
        "hate",
        "racist",
        "sexist",
        "porn",
        "nude",
        "explicit",
        "bomb",
        "kill",
        "murder",
        "suicide",
        "drugs",
        "weapon",
    ]
    return not any(word in prompt.lower() for word in forbidden_words)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str) -> Ollama:
    return Ollama(model=model_name)


@st.cache_resource(show_spinner=False)
def get_embedding_function() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    embedding_function = get_embedding_function()
    if not os.path.exists(PERSIST_DIRECTORY):
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(
                f"Could not find the PDF at `{PDF_PATH}`. Please place the document before running the app."
            )
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        Chroma.from_documents(
            chunks,
            embedding_function=embedding_function,
            persist_directory=PERSIST_DIRECTORY,
        )
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function,
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_retriever():
    return get_vectorstore().as_retriever()


@st.cache_resource(show_spinner=False)
def get_chain(model_name: str) -> RetrievalQA:
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer also mention page number of the respective content .

{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=get_llm(model_name),
        chain_type="stuff",
        retriever=get_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )


@st.cache_resource(show_spinner=False)
def initialize_default_resources(model_name: str) -> Dict[str, Any]:
    retriever = get_retriever()
    chain = get_chain(model_name)
    pages = load_pdf_pages(PDF_PATH)
    return {
        "retriever": retriever,
        "chain": chain,
        "pages": pages,
        "source": PDF_PATH,
        "label": f"Default PDF — {Path(PDF_PATH).name}",
        "model": model_name,
    }


def prepare_uploaded_resources(uploaded_file, model_name: str) -> Dict[str, Any]:
    if "uploaded_resources" not in st.session_state:
        st.session_state["uploaded_resources"] = {}

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    cache_key = f"uploaded_{model_name}_{file_hash}"
    stored = st.session_state["uploaded_resources"].get(cache_key)
    if stored:
        return stored

    temp_dir = Path(tempfile.mkdtemp(prefix="chroma_upload_"))
    filename = uploaded_file.name or f"document_{file_hash}.pdf"
    temp_pdf_path = temp_dir / filename
    temp_pdf_path.write_bytes(file_bytes)

    loader = PyPDFLoader(str(temp_pdf_path))
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding_function=get_embedding_function(),
        persist_directory=str(temp_dir),
    )
    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(model_name),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PromptTemplate(
            template=PROMPT.template,
            input_variables=PROMPT.input_variables,
        )},
    )

    resources = {
        "retriever": retriever,
        "chain": chain,
        "pages": pages,
        "source": str(temp_pdf_path),
        "label": f"Uploaded — {filename}",
        "persist_directory": str(temp_dir),
        "cache_key": cache_key,
        "model": model_name,
    }
    st.session_state["uploaded_resources"][cache_key] = resources
    return resources


def render_document_viewer(pages: List[Document], source_label: str) -> None:
    st.markdown("## Document viewer")
    if not pages:
        st.info("No pages available to preview.")
        return

    total_pages = len(pages)
    page_selector_key = f"page_selector_{abs(hash(source_label)) % 1_000_000}"
    page_number = st.number_input(
        "Page number",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=page_selector_key,
    )

    st.markdown(f"**Source:** {source_label}")
    st.caption("Browse the original document to understand the context.")

    page = pages[int(page_number) - 1]
    st.code(page.page_content, language="markdown")


def render_history(active_label: str | None = None) -> None:
    history = st.session_state.get("history", [])
    if not history:
        st.info("No questions asked yet. Submit a question to start the conversation.")
        return

    st.markdown("## Conversation history")
    for idx, item in enumerate(history, start=1):
        source_label = item.get("label") or item.get("source", "Unknown source")
        with st.chat_message("user"):
            st.markdown(f"**Q{idx}:** {item['question']}")
            st.caption(f"Document: {source_label}")
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            docs = item.get("docs") or []
            if docs:
                expand_default = source_label == active_label
                with st.expander("Retrieved context", expanded=expand_default):
                    for snippet_idx, doc in enumerate(docs, start=1):
                        metadata = getattr(doc, "metadata", {}) or {}
                        page = metadata.get("page", "?")
                        source = metadata.get("source", source_label)
                        st.markdown(
                            f"**Snippet {snippet_idx}** — page {page}, source: `{source}`"
                        )
                        content = getattr(doc, "page_content", "")
                        preview = content[:700]
                        if len(content) > 700:
                            preview += "..."
                        st.code(preview, language="markdown")


def main() -> None:
    st.set_page_config(page_title="Local LLM PDF QA", layout="wide")
    st.title("Local LLM with Retrieval-Augmented Generation")
    st.caption(
        "Ask questions about the PDF located at `./data/document.pdf` using your local Ollama model."
    )

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "uploaded_resources" not in st.session_state:
        st.session_state["uploaded_resources"] = {}
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = DEFAULT_MODEL

    active_resources: Dict[str, Any] | None = None

    with st.sidebar:
        st.header("Document source")
        model_name = st.selectbox(
            "Ollama model",
            options=["gemma:2b", "llama3:8b", "mistral"],
            index=["gemma:2b", "llama3:8b", "mistral"].index(
                st.session_state.get("model_name", DEFAULT_MODEL)
            ),
            help="Pick a lighter or faster model depending on availability.",
        )
        if model_name != st.session_state["model_name"]:
            st.session_state["model_name"] = model_name
            st.cache_resource.clear()
            st.rerun()

        source_choice = st.radio(
            "Select document source", ["Default PDF", "Upload PDF"], index=0
        )

        if source_choice == "Default PDF":
            try:
                active_resources = initialize_default_resources(model_name)
            except FileNotFoundError as e:
                st.error(str(e))
        else:
            uploaded_file = st.file_uploader(
                "Upload a PDF document", type="pdf", accept_multiple_files=False
            )
            if uploaded_file is not None:
                try:
                    active_resources = prepare_uploaded_resources(uploaded_file, model_name)
                except Exception as e:
                    st.error(f"Failed to process the uploaded file: {e}")
            else:
                st.info("Upload a PDF to start asking questions about it.")

        st.divider()
        st.header("Configuration")
        st.markdown(f"- **Model**: `{model_name}`")
        if active_resources:
            st.markdown(
                f"- **Active document**: `{Path(active_resources['source']).name}`"
            )
        else:
            st.markdown("- **Active document**: _None selected_")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state["history"] = []
            st.experimental_rerun()

    if not active_resources:
        st.warning("Select or upload a document from the sidebar to begin.")
        return

    render_document_viewer(active_resources.get("pages", []), active_resources["label"])

    retriever = active_resources["retriever"]
    chain = active_resources["chain"]

    st.markdown("## Ask a question")
    with st.form("qa_form"):
        query = st.text_area("Enter your question", height=120)
        submitted = st.form_submit_button("Submit")

    if submitted:
        cleaned_query = query.strip()
        if not cleaned_query:
            st.warning("Please enter a question before submitting.")
        elif not is_safe(cleaned_query):
            st.warning("Sorry, your question contains unsafe content. Please try again.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    retrieved_docs = retriever.invoke(cleaned_query)
                    if not isinstance(retrieved_docs, list):
                        retrieved_docs = [retrieved_docs]
                    result = chain.invoke({"query": cleaned_query})
                    answer = result.get("result", "I couldn't generate an answer.")
                    st.session_state["history"].insert(
                        0,
                        {
                            "question": cleaned_query,
                            "answer": answer,
                            "docs": retrieved_docs,
                            "source": active_resources.get("source"),
                            "label": active_resources.get("label"),
                        },
                    )
                    st.success(
                        "Answer generated! Scroll down to review the conversation and retrieved context."
                    )
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")

    render_history(active_resources.get("label"))


if __name__ == "__main__":
    main()