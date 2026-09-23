import os
import json
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# Connect to Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Groq client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

print("Connected to Supabase")
print("Embedding model loaded")
print("Groq client ready")


# ============================================================
# STORE DOCUMENT
# ============================================================

def store_document(title, content, source, page_number=1):

    embedding = embedding_model.encode(content).tolist()

    supabase.table("documents").insert({
        "title": title,
        "content": content,
        "source": source,
        "page_number": page_number,
        "embedding": embedding
    }).execute()

    print("Stored:", title)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_similar(query, top_k=5, threshold=0.30):

    query_embedding = embedding_model.encode(query).tolist()

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": top_k
        }
    ).execute()

    return result.data


# ============================================================
# RAG QUERY
# ============================================================

def rag_query(question):

    print("\nQUESTION:", question)
    print("-" * 55)

    documents = search_similar(question)

    if not documents:
        answer = "I don't know. I could not find relevant information in the documents."

        print("ANSWER:")
        print(answer)

        return {
            "question": question,
            "answer": answer,
            "sources": []
        }

    print("RETRIEVED DOCUMENTS:")

    for document in documents:
        print(
            "-",
            document["title"],
            "| page",
            document["page_number"],
            "| similarity:",
            round(document["similarity"], 3)
        )

    context = ""

    for document in documents:
        context += (
            f"[{document['title']}, page {document['page_number']}]\n"
            f"{document['content']}\n\n"
        )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the provided documents. "
                    "Do not use outside knowledge. "
                    "Include a citation like [Title, page X]. "
                    "If the documents do not contain the answer, say: "
                    "I don't know. I could not find relevant information "
                    "in the documents."
                )
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENTS:\n{context}\n"
                    f"QUESTION:\n{question}"
                )
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    answer = response.choices[0].message.content

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")

    sources = []

    for document in documents:
        print(
            "-",
            document["title"],
            "| page",
            document["page_number"]
        )

        sources.append({
            "title": document["title"],
            "page_number": document["page_number"],
            "similarity": round(document["similarity"], 4)
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


# ============================================================
# KNOWLEDGE BASE
# ============================================================

documents = [
    (
        "Machine Learning",
        "Machine learning is a branch of artificial intelligence that allows computers to learn patterns from examples and use those patterns to make predictions or decisions. A machine learning system learns from data instead of relying only on explicitly written rules. The quality of the training data and the way a model is evaluated can affect how well it performs on new data.",
        "AI Guide",
        1
    ),
    (
        "Prompt Engineering",
        "Prompt engineering is the practice of giving a language model clear instructions, useful context, examples, and constraints so that it produces a desired result. A well-designed prompt can specify the role of the model, the task it should perform, the information it should use, and the format expected in the response. Structured prompts can make AI workflows more consistent and easier to control.",
        "AI Guide",
        2
    ),
    (
        "Embeddings",
        "Embeddings convert text into numerical vectors that represent the meaning of the text. Texts with similar meanings tend to have similar vector representations even when they use different words. Embeddings are useful for semantic search because a system can compare the vector of a user query with stored vectors to find related information. This allows searches based on meaning rather than exact keyword matches.",
        "AI Guide",
        3
    ),
    (
        "RAG",
        "Retrieval Augmented Generation, or RAG, allows an AI system to look up relevant information from a knowledge base before generating an answer. The system searches for relevant document chunks, provides those chunks to the language model as context, and generates an answer based on the retrieved information. RAG helps ground answers in the provided documents and can include citations showing the source of the information. If the required information is not in the documents, the system should say that it does not know instead of making up an answer.",
        "AI Guide",
        4
    ),
    (
        "Function Calling",
        "Function calling allows a language model to request external tools or functions to complete tasks. These tools can include databases, APIs, support systems, calculators, or other services. Instead of only generating text, the model can decide that it needs a tool, provide the required arguments, receive the tool result, and then continue the task. Function calling is useful when an AI system needs access to information or actions outside the language model itself.",
        "AI Guide",
        5
    ),
    (
        "Temperature",
        "Temperature controls how much variation or randomness a language model can use when generating responses. Lower temperature values generally make responses more consistent and predictable, while higher values can produce more varied outputs. Temperature can therefore affect how repeatable an AI system is, especially when consistency matters in an automated workflow.",
        "AI Guide",
        6
    ),
    (
        "Token Usage",
        "Tokens are the units of text processed by a language model. Input text and generated responses are made up of tokens, and the amount of token usage can affect cost and available context. Tracking token usage is useful when building AI applications because long prompts and large responses can consume more tokens and may reach model limits.",
        "AI Guide",
        7
    ),
    (
        "Document AI",
        "Document AI uses artificial intelligence to extract and process information from documents such as invoices, receipts, forms, and other business files. Document processing can involve extracting text, identifying important fields, converting information into structured data, and handling image-based or scanned documents. This can make large amounts of document information easier to search, organize, and use in automated workflows.",
        "AI Guide",
        8
    ),
    (
        "Vector Databases",
        "Vector databases store numerical embeddings and support similarity search across collections of information. Instead of searching only for exact keywords, a vector database compares the embedding of a query with stored embeddings and returns the most semantically similar records. This makes vector databases useful for semantic search and retrieval augmented generation systems where relevant document chunks need to be found quickly.",
        "AI Guide",
        9
    ),
    (
        "Model Validation",
        "Model validation is the process of checking how well a machine learning system performs on data that was not used to train the model. Validation helps determine whether a model generalizes well to new data and can reveal problems such as overfitting. Different evaluation measures can be used depending on the machine learning task and the type of prediction being made.",
        "AI Guide",
        10
    )
]


# ============================================================
# REPLACE AI GUIDE DOCUMENTS
# ============================================================

print("\nRESETTING AI GUIDE DOCUMENTS")
print("=" * 55)

supabase.table("documents").delete().eq(
    "source", "AI Guide"
).execute()

print("Old AI Guide documents removed.")


# ============================================================
# STORE DOCUMENTS
# ============================================================

print("\nSTORING DOCUMENTS")
print("=" * 55)

for document in documents:
    store_document(*document)


# ============================================================
# RAG TESTS
# ============================================================

results = []

# Positive test 1
results.append(
    rag_query(
        "How does RAG use a knowledge base to answer questions?"
    )
)

# Positive test 2
results.append(
    rag_query(
        "How can a vector database find information with similar meaning?"
    )
)


# Negative test 1
results.append(
    rag_query(
        "What is the recipe for jollof rice?"
    )
)

# Negative test 2
results.append(
    rag_query(
        "What are the symptoms of malaria?"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "week11_vid2_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

print("\nVid 2 complete. Results saved.")