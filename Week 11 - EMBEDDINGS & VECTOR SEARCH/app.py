import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Knowledge Bot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# INITIALIZE CONNECTIONS
# ============================================================

def init_connections():
    supabase = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    )

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    groq_client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )

    return supabase, embedding_model, groq_client


supabase, embedding_model, groq_client = init_connections()


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        if end < len(text):

            last_period = text.rfind(
                ".",
                start,
                end
            )

            if last_period > start + chunk_size // 2:
                end = last_period + 1

        chunk = text[start:end].strip()

        if len(chunk) > 20:
            chunks.append(chunk)

        # Prevent the loop from repeating the last chunk
        if end == len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# INGEST DOCUMENT
# ============================================================

def ingest_document(title, content, source):

    print()
    print("Ingesting:", title)
    print("=" * 40)

    chunks = chunk_text(content)
    stored = 0

    for i, chunk in enumerate(chunks):

        embedding = embedding_model.encode(
            chunk
        ).tolist()

        supabase.table("documents").insert({
            "title": title,
            "content": chunk,
            "source": source,
            "page_number": i + 1,
            "embedding": embedding
        }).execute()

        stored += 1

    print("Stored", stored, "chunks from", title)

    return stored


# ============================================================
# RAG SEARCH
# ============================================================

def rag_search(question, top_k=5):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.30,
            "match_count": top_k
        }
    ).execute()

    return result.data


# ============================================================
# RAG GENERATION
# ============================================================

def rag_generate(question, documents):

    context = ""

    for i, doc in enumerate(documents):

        context += (
            f"[Document {i + 1}] "
            f"Source: {doc['title']}\n"
        )

        context += doc["content"]
        context += "\n\n"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions based ONLY on the provided "
                    "documents. Cite sources as [Document Title, "
                    "chunk X]. If the answer is not in the "
                    "provided documents, say you don't know."
                )
            },
            {
                "role": "user",
                "content": (
                    "DOCUMENTS:\n"
                    + context
                    + "\nQUESTION:\n"
                    + question
                )
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# DOCUMENT UPLOAD - SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Add Documents")

    uploaded_file = st.file_uploader(
        "Upload a text file",
        type=["txt"]
    )

    doc_title = st.text_input(
        "Document title"
    )

    if st.button("Ingest Document"):

        if uploaded_file and doc_title:

            content = uploaded_file.read().decode(
                "utf-8"
            )

            with st.spinner(
                "Chunking, embedding, and storing..."
            ):

                count = ingest_document(
                    doc_title,
                    content,
                    uploaded_file.name
                )

            st.success(
                f"Stored {count} chunks."
            )

        else:

            st.warning(
                "Please provide both a file and a title."
            )

    st.divider()

    try:

        count_result = (
            supabase
            .table("documents")
            .select("id", count="exact")
            .execute()
        )

        st.caption(
            f"Documents in database: "
            f"{count_result.count}"
        )

    except Exception:

        st.caption(
            "Database connection error"
        )


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

st.title("Company Knowledge Bot")

st.caption(
    "Ask questions about your policies, manuals, and documents."
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if "sources" in msg and msg["sources"]:

            with st.expander("Sources"):

                for source in msg["sources"]:

                    st.write(
                        "- "
                        + source["title"]
                        + " (similarity: "
                        + str(source["similarity"])
                        + ")"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question..."
)


if question:

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            docs = rag_search(question)


        if not docs:

            answer = (
                "I don't have information about that "
                "in my knowledge base."
            )

            sources = []

        else:

            with st.spinner(
                "Generating answer..."
            ):

                answer = rag_generate(
                    question,
                    docs
                )

            sources = []

            for doc in docs:

                sources.append({
                    "title": doc["title"],
                    "similarity": round(
                        doc["similarity"],
                        3
                    )
                })


        st.markdown(answer)


        if sources:

            with st.expander("Sources"):

                for source in sources:

                    st.write(
                        "- "
                        + source["title"]
                        + " (similarity: "
                        + str(source["similarity"])
                        + ")"
                    )


    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })