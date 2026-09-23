import os
import json
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ============================================================
# INITIALIZE
# ============================================================

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

print("Connected to Supabase")
print("Embedding model loaded")
print("Groq client ready")


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_period = text.rfind(".", start, end)

            if last_period > start + chunk_size // 2:
                end = last_period + 1

        chunks.append(text[start:end].strip())

        if end == len(text):
            break

        start = end - overlap

    return [chunk for chunk in chunks if chunk]


# ============================================================
# INGEST DOCUMENT
# ============================================================

def ingest_document(title, content, source):
    chunks = chunk_text(content)
    stored = 0

    for i, chunk in enumerate(chunks):

        existing = (
            supabase
            .table("documents")
            .select("id")
            .eq("title", title)
            .eq("source", source)
            .eq("content", chunk)
            .execute()
        )

        if existing.data:
            continue

        embedding = embedding_model.encode(chunk).tolist()

        supabase.table("documents").insert({
            "title": title,
            "content": chunk,
            "source": source,
            "page_number": i + 1,
            "embedding": embedding
        }).execute()

        stored += 1

    print(f"Stored {stored} chunk(s): {title}")

    return stored


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def rag_search(question, top_k=5, threshold=0.30):

    query_embedding = embedding_model.encode(question).tolist()

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": 20
        }
    ).execute()

    allowed_sources = {
        "AI Automation Guide",
        "Python Guide",
        "Cybersecurity Guide"
    }

    results = [
        document
        for document in result.data
        if document["source"] in allowed_sources
    ]

    return results[:top_k]


# ============================================================
# GENERATE ANSWER
# ============================================================

def rag_generate(question, documents):

    if not documents:
        return {
            "answer": "I don't know. I could not find relevant information in the documents.",
            "sources": []
        }

    context = ""

    for document in documents:
        context += (
            f"[Source: {document['title']}, chunk {document['page_number']}]\n"
            f"{document['content']}\n\n"
        )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the provided documents. "
                    "Combine information from multiple documents when necessary. "
                    "Cite sources like [Document Title, chunk X]. "
                    "Do not use outside knowledge. "
                    "If the documents do not contain the answer, say: "
                    "I don't know. I could not find relevant information "
                    "in the documents."
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
        temperature=0.2,
        max_tokens=400
    )

    answer = response.choices[0].message.content

    sources = [
        {
            "title": document["title"],
            "page": document["page_number"],
            "similarity": round(document["similarity"], 4)
        }
        for document in documents
    ]

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# COMPLETE RAG QUERY
# ============================================================

def rag_query(question):

    print("\nQUESTION:", question)
    print("-" * 60)

    documents = rag_search(question)

    if documents:
        print("RETRIEVED DOCUMENTS:")

        for document in documents:
            print(
                "-",
                document["title"],
                "| chunk",
                document["page_number"],
                "| similarity:",
                round(document["similarity"], 3)
            )

    result = rag_generate(question, documents)

    print("\nANSWER:")
    print(result["answer"])

    if result["sources"]:
        print("\nSOURCES:")

        for source in result["sources"]:
            print(
                "-",
                source["title"],
                "| chunk",
                source["page"],
                "| similarity:",
                source["similarity"]
            )

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"]
    }


# ============================================================
# THREE DOCUMENTS
# ============================================================

seed_documents = [
    (
        "AI Automation",
        "AI automation uses artificial intelligence to perform tasks that normally require repeated human effort. Large language models can classify information, summarize text, generate structured outputs, and call external tools. AI automation workflows can connect several steps together to process information and complete tasks. Good AI automation workflows should include clear instructions, validation, testing, and error handling.",
        "AI Automation Guide"
    ),
    (
        "Python Basics",
        "Python is a programming language commonly used for automation, data analysis, and artificial intelligence projects. Functions group reusable logic into named blocks of code. Lists store multiple values, dictionaries store key-value pairs, and libraries provide reusable functionality for projects. Python can be used to build automation workflows that connect different tools and services.",
        "Python Guide"
    ),
    (
        "Cybersecurity Basics",
        "API keys, passwords, and other credentials should be kept secret and should not be committed to GitHub. Environment variables and .env files can be used to store local secrets. A .gitignore file should include .env so the secret file is not tracked by Git. Keeping credentials outside source code helps reduce the risk of exposing sensitive information.",
        "Cybersecurity Guide"
    )
]


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("\nINGESTING DOCUMENTS")
    print("=" * 60)

    for title, content, source in seed_documents:
        ingest_document(title, content, source)

    results = []

    # Multi-document test 1
    results.append(
        rag_query(
            "How can Python be used to build an AI automation workflow?"
        )
    )

    # Multi-document test 2
    results.append(
        rag_query(
            "How can a Python-based AI automation project keep its API keys secure?"
        )
    )

    # Unanswerable test
    results.append(
        rag_query(
            "What is the recipe for jollof rice?"
        )
    )

    with open(
        "week11_vid3_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("\nVid 3 pipeline tests complete.")
    print("Results saved.")