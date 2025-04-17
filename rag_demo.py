#!/usr/bin/env python3
import os
import json
import re
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# ─── 1. Load env ───────────────────────────────────────────────────────────────
load_dotenv()  # expects OPENAI_API_KEY in your .env

INDEX_PATH = "faiss_index"

def build_index():
    """Build and save the FAISS index from course documents"""
    # ─── 2. Load & tag Markdown docs ───────────────────────────────────────────────
    md_paths = [
        Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/heinz_courses_md"),
        Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/syllabi_heinz_courses_md"),
    ]

    md_docs = []
    for folder in md_paths:
        for md_file in folder.rglob("*.md"):
            loader = TextLoader(str(md_file), encoding="utf8")
            for doc in loader.load():
                code = md_file.stem.replace("-", "_")
                doc.metadata.update({
                    "course": code,
                    "source": str(md_file),
                    "type": "markdown"
                })
                md_docs.append(doc)
    print(f"✅ Loaded {len(md_docs)} Markdown docs")

    # ─── 3. Load structured JSON ───────────────────────────────────────────────────
    desc_file = Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/structured_output_courseDesc.json")
    syll_file = Path("/Users/macbookairm1/Desktop/PythonLearn/LLM/llm_engineering/project/haystack/knowledge-base-course/structured_output_Syllabi.json")

    course_desc = json.loads(desc_file.read_text(encoding="utf8"))
    syllabi_data = json.loads(syll_file.read_text(encoding="utf8"))
    print(f"✅ Loaded {len(course_desc)} course‑desc entries and {len(syllabi_data)} syllabi entries")

    # ─── 4. Convert JSON → Documents (robust) ───────────────────────────────────────
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
                page_content=content,
                metadata={
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
                    page_content=content,
                    metadata={
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

    # ─── 5. Chunk Markdown & Combine ──────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    md_chunks = splitter.split_documents(md_docs)
    all_docs = md_chunks + json_docs
    print(f"📚 Total docs for indexing: {len(all_docs)}")

    # ─── 6. Build FAISS index ─────────────────────────────────────────────────────
    emb = OpenAIEmbeddings()
    db = FAISS.from_documents(all_docs, emb)
    
    # Save the index to disk
    db.save_local(INDEX_PATH)
    print(f"✅ FAISS index built and saved to {INDEX_PATH}")

def load_index():
    """Load the pre-built FAISS index"""
    if not os.path.exists(INDEX_PATH):
        print(f"❌ Error: Index not found at {INDEX_PATH}. Run with --build first.")
        sys.exit(1)
        
    emb = OpenAIEmbeddings()
    db = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS index from {INDEX_PATH}")
    return db

# ─── 7. Retriever factory ──────────────────────────────────────────────────────
def make_retriever(db, course: str, doc_type: str = None):
    filt = {"course": course}
    if doc_type:
        filt["type"] = doc_type
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.5},
        filter=filt
    )

# ─── 8. RAG QA function ───────────────────────────────────────────────────────
def answer_query(db, query: str, course_code: str):
    if re.search(r"\bprerequisite", query, re.I):
        retr = make_retriever(db, course_code, "course_desc")
    else:
        retr = make_retriever(db, course_code)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0, top_p=1.0),
        chain_type="refine",
        retriever=retr,
        return_source_documents=True
    )
    return qa({"query": query})

def run_chat(db):
    """Interactive chat mode"""
    print("💬 CMU Course Chatbot")
    print("Enter 'exit', 'quit', or 'q' to end the session\n")
    
    # Ask for course code first
    course_code = input("Enter course code (e.g. 95_891): ")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ("exit", "quit", "q"):
            break
            
        res = answer_query(db, question, course_code)
        print("\nAnswer:", res["result"])
        print("\nSources:")
        for doc in res["source_documents"]:
            print(f"- {doc.metadata['source']}")
        print("\n" + "-"*40)

def run_example_query(db):
    """Run a single example query"""
    print("💬 CMU Course Chatbot Example Query\n")
    
    # Example query with fixed values
    code = "95_891"  # Example course code
    q = "Who is the professor of introduction to AI class?"
    
    print(f"Course code: {code}")
    print(f"Question: {q}")
    
    res = answer_query(db, q, code)
    print("\nAnswer:", res["result"])
    print("Sources:", [d.metadata["source"] for d in res["source_documents"]])
    print("\n" + "-"*40 + "\n")

# ─── Main function with command line arguments ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMU Course Chatbot with separate embedding and chat modes")
    parser.add_argument("--build", action="store_true", help="Build and save the FAISS index")
    parser.add_argument("--example", action="store_true", help="Run an example query")
    
    args = parser.parse_args()
    
    if args.build:
        build_index()
    elif args.example:
        db = load_index()
        run_example_query(db)
    else:
        db = load_index()
        run_chat(db)
