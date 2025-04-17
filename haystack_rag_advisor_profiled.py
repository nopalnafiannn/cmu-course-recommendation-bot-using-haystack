#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pathlib import Path
import json

from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import ChatMessage

# ─── 1. Load env ───────────────────────────────────────────────────────────────
load_dotenv()  # expects OPENAI_API_KEY in your .env

def build_index():
    """Build and save the document store with course documents"""
    md_paths = [
        Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/heinz_courses_md"),
        Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/syllabi_heinz_courses_md"),
    ]
    md_docs = []
    for folder in md_paths:
        for md_file in folder.rglob("*.md"):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                code = md_file.stem.replace("-", "_")
                doc = Document(
                    content=content,
                    meta={
                        "course": code,
                        "source": str(md_file),
                        "type": "markdown"
                    }
                )
                md_docs.append(doc)
    print(f"✅ Loaded {len(md_docs)} Markdown docs")

    desc_file = Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/structured_output_courseDesc.json")
    syll_file = Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/structured_output_Syllabi.json")
    course_desc = json.loads(desc_file.read_text(encoding="utf8"))
    syllabi_data = json.loads(syll_file.read_text(encoding="utf8"))
    print(f"✅ Loaded {len(course_desc)} course‑desc entries and {len(syllabi_data)} syllabi entries")

    json_docs = []
    for fn, cd in course_desc.items():
        if isinstance(cd, dict):
            nested_keys = [k for k, v in cd.items() if isinstance(v, dict)]
        else:
            continue
        for nk in nested_keys:
            nested = cd[nk]
            desc = nested.get("Description", "")
            prereq = nested.get("Prerequisites", cd.get("Prerequisites", ""))
            lo = nested.get("Learning Outcomes", cd.get("Learning Outcomes", []))
            parts = [f"Description: {desc}"]
            if prereq:
                parts.append(f"Prerequisites: {prereq}")
            if isinstance(lo, dict):
                parts += [f"{k}: {v}" for k, v in lo.items()]
            else:
                parts += [f"Outcome: {item}" for item in lo]
            content = "\n".join(parts)
            meta = nested.get("Metadata", {})
            course_code = meta.get("Course Number", Path(fn).stem).replace("-", "_")
            title = meta.get("Title", nk)
            json_docs.append(Document(
                content=content,
                meta={
                    "course": course_code,
                    "title": title,
                    "type": "course_desc",
                    "source": fn
                }
            ))
    for fn, sy in syllabi_data.items():
        if isinstance(sy, dict):
            for main in [k for k in sy if k != "Document Metadata"]:
                content = json.dumps(sy[main], indent=2)
                course_code = Path(fn).stem.split()[0].replace("-", "_")
                json_docs.append(Document(
                    content=content,
                    meta={
                        "course": course_code,
                        "title": main,
                        "type": "syllabus_json",
                        "source": fn
                    }
                ))
        else:
            continue
    print(f"✅ Converted to {len(json_docs)} JSON‑based docs")

    document_store = InMemoryDocumentStore()
    document_embedder = OpenAIDocumentEmbedder(
        model="text-embedding-3-small",
        meta_fields_to_embed=["title", "course"]
    )
    md_docs_with_embeddings = document_embedder.run(md_docs)["documents"]
    document_store.write_documents(md_docs_with_embeddings)
    json_docs_with_embeddings = document_embedder.run(json_docs)["documents"]
    document_store.write_documents(json_docs_with_embeddings)
    print(f"✅ Indexed {document_store.count_documents()} documents with embeddings")
    return document_store

def create_rag_pipeline(document_store):
    """Create the RAG pipeline with user profile support"""
    text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=10,
        scale_score=True
    )
    generator = OpenAIGenerator(model="gpt-4o-mini")
    prompt_builder = PromptBuilder(
        template="""
You are a friendly and helpful course advisor. Respond in a positive, friendly, and encouraging tone.

User Profile:
{{ profile }}

Based on the following documents:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}

Answer the question: {{ query }}

Provide a comprehensive and specific answer that directly addresses the question.
If the question asks about a specific course, mention the course code and title.
If the information isn't available in the documents, state that clearly.
""",
        required_variables=["profile", "query", "documents"]
    )
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", generator)
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
    return rag_pipeline

def extract_course_number(query):
    import re
    course_patterns = [r"\b\d{2}[_-]\d{3}\b", r"\b\d{5}\b"]
    for pattern in course_patterns:
        matches = re.findall(pattern, query)
        if matches:
            return matches[0].replace('-', '_')
    return None

def answer_query(document_store, pipeline, query, profile):
    """Process a query through the RAG pipeline including user profile"""
    course_code = extract_course_number(query)
    retriever_params = {}
    if course_code:
        retriever_params["filters"] = {"field": "course", "operator": "==", "value": course_code}
    result = pipeline.run({
        "text_embedder": {"text": query},
        "retriever": retriever_params,
        "prompt_builder": {"profile": profile, "query": query}
    })
    answer = result["generator"]["replies"][0]
    text_embedder = pipeline.get_component("text_embedder")
    query_embedding = text_embedder.run(text=query)["embedding"]
    similar_docs = document_store.embedding_retrieval(
        query_embedding=query_embedding,
        top_k=10,
        filters={"field": "course", "operator": "==", "value": course_code} if course_code else None
    )
    sources = []
    for doc in similar_docs:
        src = doc.meta.get("source")
        if src and src not in sources:
            sources.append(src)
    return answer, sources[:5]

def run_chat():
    """Interactive chat mode with user profiling"""
    print("💬 CMU Course Advisor Chatbot (Personalized)")
    print("Enter 'exit', 'quit', or 'q' to end the session\n")
    interest = input("What is your interest field? ")
    level = input("What is your current level (e.g., beginner, intermediate, expert)? ")
    user_profile = f"Interest field: {interest}. Current level: {level}."
    print(f"\nUser Profile: {user_profile}\n")
    document_store = build_index()
    pipeline = create_rag_pipeline(document_store)
    while True:
        question = input("\nYour question: ")
        if question.lower() in ("exit", "quit", "q"):
            break
        answer, sources = answer_query(document_store, pipeline, question, user_profile)
        print("\nAnswer:", answer)
        print("\nSources:")
        for source in sources:
            print(f"- {source}")
        print("\n" + "-"*40)

def run_test_query():
    """Non-interactive test mode with a sample query and default profile"""
    print("💬 CMU Course Advisor Chatbot (Test Mode)")
    document_store = build_index()
    pipeline = create_rag_pipeline(document_store)
    test_query = "What is course 95-865 about?"
    default_profile = "Interest field: general. Current level: beginner."
    print(f"\nTest user profile: {default_profile}")
    print(f"\nTest question: {test_query}")
    answer, sources = answer_query(document_store, pipeline, test_query, default_profile)
    print("\nAnswer:", answer)
    print("\nSources:")
    for source in sources:
        print(f"- {source}")

if __name__ == "__main__":
    interactive_mode = True
    if interactive_mode:
        run_chat()
    else:
        run_test_query()