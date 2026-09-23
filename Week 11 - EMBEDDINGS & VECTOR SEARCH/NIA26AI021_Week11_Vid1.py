from sentence_transformers import SentenceTransformer
import numpy as np
import json


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SimpleVectorDB:
    # In-memory vector database for semantic search
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, text, metadata=None):
        embedding = model.encode(text)
        self.documents.append({"text": text, "metadata": metadata or {}})
        self.embeddings.append(embedding)
        print("Added:", text[:60] + "...")

    def search(self, query, top_k=3):
        query_embedding = model.encode(query)
        similarities = []

        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))

        similarities.sort(reverse=True, key=lambda x: x[0])

        results = []
        for similarity, index in similarities[:top_k]:
            results.append({
                "similarity": round(similarity, 4),
                "text": self.documents[index]["text"],
                "metadata": self.documents[index]["metadata"]
            })

        return results


def keyword_search(documents, query, top_k=3):
    query_words = set(query.lower().split())
    results = []

    for doc in documents:
        doc_words = set(doc["text"].lower().split())
        score = len(query_words.intersection(doc_words))
        if score > 0:
            results.append((score, doc))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]


# Build a knowledge base with 10 documents
print("Building Knowledge Base")
print("=" * 40)

db = SimpleVectorDB()

documents = [
    ("Machine Learning", "Machine learning allows computers to learn patterns from examples and use those patterns to make predictions.", {"source": "AI Guide", "page": 1}),
    ("Prompt Engineering", "Prompt engineering uses clear instructions, context, examples, and constraints to guide a language model toward useful outputs.", {"source": "AI Guide", "page": 2}),
    ("Embeddings", "Embeddings represent text as numerical vectors so that semantically similar pieces of information can be found even when they use different words.", {"source": "AI Guide", "page": 3}),
    ("RAG", "Retrieval augmented generation combines document retrieval with a language model so the model can answer questions using a selected knowledge base.", {"source": "AI Guide", "page": 4}),
    ("Function Calling", "Function calling allows a language model to request external tools such as databases, APIs, or support systems to complete a task.", {"source": "AI Guide", "page": 5}),
    ("Temperature", "Temperature controls how much variation or randomness a language model can use when generating responses.", {"source": "AI Guide", "page": 6}),
    ("Token Usage", "Tokens are the units of text processed by a language model, and tracking token usage helps manage cost and context limits.", {"source": "AI Guide", "page": 7}),
    ("Document AI", "Document AI can extract useful information from documents such as invoices, receipts, and forms and turn it into structured data.", {"source": "AI Guide", "page": 8}),
    ("Vector Databases", "Vector databases store embeddings and make similarity search efficient across large collections of information.", {"source": "AI Guide", "page": 9}),
    ("Model Validation", "Model validation checks how well a machine learning system performs on data that was not used to train the model.", {"source": "AI Guide", "page": 10}),
]

for title, text, metadata in documents:
    db.add(text, {"title": title, **metadata})


# Test semantic search with five different queries
queries = [
    "How do computers learn from examples?",
    "How can I give an AI clearer instructions?",
    "How can a system find information with different wording?",
    "How can an AI answer questions from my own documents?",
    "How can a model trigger an external service?",
]

all_results = []

print("\nSemantic Search Tests")
print("=" * 40)

for query in queries:
    print("\nQuery:", query)
    results = db.search(query, top_k=3)
    for result in results:
        print(f"{result['similarity']} - {result['metadata']['title']}")
        print("  ", result["text"])

    keyword_results = keyword_search(db.documents, query, top_k=3)
    keyword_titles = [doc["metadata"]["title"] for _, doc in keyword_results]

    print("Keyword matches:", keyword_titles if keyword_titles else "None")

    all_results.append({
        "query": query,
        "semantic_results": results,
        "keyword_matches": keyword_titles
    })


# Industry use case
industry_use_case = {
    "industry": "AI Automation / Technology",
    "use_case": "Semantic search can help an AI automation team search project documentation by meaning instead of exact keywords.",
    "documents_to_index": [
        "API documentation",
        "automation workflows",
        "troubleshooting guides",
        "project notes",
        "standard operating procedures"
    ]
}

print("\nIndustry Use Case")
print("=" * 40)
print(industry_use_case["use_case"])
print("Documents to index:", ", ".join(industry_use_case["documents_to_index"]))


with open("week11_vid1_results.json", "w", encoding="utf-8") as file:
    json.dump({
        "semantic_search": all_results,
        "industry_use_case": industry_use_case
    }, file, indent=2)

print("\nVid 1 complete. Results saved.")
