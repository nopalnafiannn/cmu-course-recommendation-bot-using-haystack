# CMU Course Advisor Chatbot (Streamlit UI & CLI)

This project demonstrates how to build a profile‑aware Retrieval‑Augmented Generation (RAG) chatbot for course advising using Haystack.

## Prerequisites

- Python 3.9 or higher
- An OpenAI API key (set via the `OPENAI_API_KEY` environment variable)
- Local course data in the `knowledge-base-course/` directory (contains Markdown and JSON course files)

## Setup

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/your_username/your_repo.git
cd your_repo

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install Haystack locally and necessary dependencies
pip install -e .
pip install streamlit python-dotenv
```

## Environment Variables

Create a `.env` file in the project root with:
```ini
OPENAI_API_KEY=your_openai_api_key_here
```
Or set the variable directly in your shell:
```bash
export OPENAI_API_KEY=your_openai_api_key_here  # Linux/macOS
set OPENAI_API_KEY=your_openai_api_key_here     # Windows
```

## Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

- This will start a local web server (default: http://localhost:8501).
- Enter your profile (interest field and level) and then ask course‑related questions.

## Running the CLI Chat (Conversation Process)

```bash
python haystack_rag_advisor_profiled.py
```

- Follow the interactive prompts to input your profile and questions.
- Type `exit`, `quit`, or `q` to end the session.

### Non-Interactive Test Mode

To run a predefined test query without manual input:
1. Open `haystack_rag_advisor_profiled.py`.
2. Change `interactive_mode = True` to `interactive_mode = False` at the bottom of the file.
3. Run:
   ```bash
   python haystack_rag_advisor_profiled.py
   ```
This will execute `run_test_query()` and print the test question, answer, and source documents.

## Programmatic Usage

You can also import and use the core functions in your own Python scripts:

```python
from haystack_rag_advisor_profiled import build_index, create_rag_pipeline, answer_query

# Build the document index and RAG pipeline
document_store = build_index()
pipeline = create_rag_pipeline(document_store)

# Ask a question with a user profile
profile = "Interest field: AI. Current level: intermediate."
query = "What is course 95_865 about?"
answer, sources = answer_query(document_store, pipeline, query, profile)

print("Answer:", answer)
print("Sources:", sources)
```

## License

See the [LICENSE](LICENSE) file in the project root for license details.