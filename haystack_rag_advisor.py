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
    # ─── 2. Load & tag Markdown docs ───────────────────────────────────────────────
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

    # ─── 3. Load structured JSON ───────────────────────────────────────────────────
    desc_file = Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/structured_output_courseDesc.json")
    syll_file = Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/structured_output_Syllabi.json")

    course_desc = json.loads(desc_file.read_text(encoding="utf8"))
    syllabi_data = json.loads(syll_file.read_text(encoding="utf8"))
    print(f"✅ Loaded {len(course_desc)} course‑desc entries and {len(syllabi_data)} syllabi entries")

    # ─── 4. Convert JSON → Documents ───────────────────────────────────────────
    json_docs = []

    # 4a. Course descriptions
    for fn, cd in course_desc.items():
        # find any dict entries (there will be the nested course info and possibly LO)
        if isinstance(cd, dict):
            nested_keys = [k for k, v in cd.items() if isinstance(v, dict)]
        else:
            # Skip if cd is not a dictionary
            continue
        for nk in nested_keys:
            nested = cd[nk]
            # description & prereq
            desc = nested.get("Description", "")
            prereq = nested.get("Prerequisites", cd.get("Prerequisites", ""))
            # learning outcomes
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

    # 4b. Full syllabus JSON
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
            # Skip if sy is not a dictionary
            continue

    print(f"✅ Converted to {len(json_docs)} JSON‑based docs")

    # ─── 5. Initialize document store and add documents ─────────────────────────
    document_store = InMemoryDocumentStore()
    
    # ─── 6. Create embeddings and index documents ───────────────────────────────
    document_embedder = OpenAIDocumentEmbedder(
        model="text-embedding-3-small",
        meta_fields_to_embed=["title", "course"]
    )
    
    # First embed and index the Markdown docs
    md_docs_with_embeddings = document_embedder.run(md_docs)["documents"]
    document_store.write_documents(md_docs_with_embeddings)
    
    # Then embed and index the JSON docs
    json_docs_with_embeddings = document_embedder.run(json_docs)["documents"]
    document_store.write_documents(json_docs_with_embeddings)
    
    print(f"✅ Indexed {document_store.count_documents()} documents with embeddings")
    
    return document_store

def create_rag_pipeline(document_store):
    """Create the RAG pipeline for answering questions"""
    
    # Create the text embedder for queries
    text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    
    # Create retriever with no course filter initially
    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=10,
        scale_score=True
    )
    
    # Create the generative model
    generator = OpenAIGenerator(model="gpt-4o-mini")
    
    # Create prompt builder to format prompt for the generator
    prompt_builder = PromptBuilder(
        template="""
        Based on the following documents:
        {% for doc in documents %}
          {{ doc.content }}
        {% endfor %}
        
        Answer the question: {{ query }}
        
        Provide a comprehensive and specific answer that directly addresses the question. 
        If the question asks about a specific course, mention the course code and title.
        If the information isn't available in the documents, state that clearly.
        """,
        required_variables=["query", "documents"]
    )
    
    # Build the RAG pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)  
    rag_pipeline.add_component("generator", generator)
    
    # Connect the components
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
    
    return rag_pipeline

def extract_course_number(query):
    """
    Attempt to extract course numbers from the query.
    This is a simple approach that could be improved with NLP techniques.
    """
    import re
    # Look for patterns like 95_891, 95-891, 95891
    course_patterns = [
        r'\b\d{2}[_-]\d{3}\b',  # Match 95_891 or 95-891
        r'\b\d{5}\b'            # Match 95891
    ]
    
    for pattern in course_patterns:
        matches = re.findall(pattern, query)
        if matches:
            # Normalize to using underscore
            return matches[0].replace('-', '_')
    
    return None

def answer_query(document_store, pipeline, query):
    """Process a query through the RAG pipeline and return the answer"""
    
    # Extract course number from query if available
    course_code = extract_course_number(query)
    
    # Run the embedder on the query
    retriever_params = {}
    
    # Add filters if a course code is extracted
    if course_code:
        retriever_params["filters"] = {"field": "course", "operator": "==", "value": course_code}
    
    result = pipeline.run({
        "text_embedder": {"text": query},
        "retriever": retriever_params,
        "prompt_builder": {"query": query}
    })
    
    # Get the generated response
    answer = result["generator"]["replies"][0]
    
    # Let's get the embedding directly from the embedder
    # Since we can't get it from the pipeline result
    text_embedder = pipeline.get_component("text_embedder")
    query_embedding_result = text_embedder.run(text=query)
    query_embedding = query_embedding_result["embedding"]
    
    # Get similar documents using the embedding
    similar_docs = document_store.embedding_retrieval(
        query_embedding=query_embedding,
        top_k=10,
        filters={"field": "course", "operator": "==", "value": course_code} if course_code else None
    )
    
    sources = []
    for doc in similar_docs:
        if doc.meta.get("source") not in sources:
            sources.append(doc.meta.get("source"))
    
    return answer, sources[:5]  # Limit to 5 sources

def run_chat():
    """Interactive chat mode"""
    print("💬 CMU Course Advisor Chatbot")
    print("Enter 'exit', 'quit', or 'q' to end the session\n")
    
    # Initialize document store and pipeline
    document_store = build_index()
    pipeline = create_rag_pipeline(document_store)
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ("exit", "quit", "q"):
            break
            
        answer, sources = answer_query(document_store, pipeline, question)
        
        print("\nAnswer:", answer)
        print("\nSources:")
        for source in sources:
            print(f"- {source}")
        print("\n" + "-"*40)

def run_test_query():
    """Non-interactive test mode with a sample query"""
    print("💬 CMU Course Advisor Chatbot (Test Mode)")
    
    # Initialize document store and pipeline
    document_store = build_index()
    pipeline = create_rag_pipeline(document_store)
    
    # Test with a sample query
    test_query = "What is course 95-865 about?"
    print(f"\nTest question: {test_query}")
    
    answer, sources = answer_query(document_store, pipeline, test_query)
    
    print("\nAnswer:", answer)
    print("\nSources:")
    for source in sources:
        print(f"- {source}")

if __name__ == "__main__":
    # Set to True for interactive mode, False for test mode
    interactive_mode = True
    
    if interactive_mode:
        run_chat()
    else:
        run_test_query()