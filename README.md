# AIScores Scholarship Recommender

A hybrid AI-powered system that helps students discover scholarships—especially those with no application fees—through both rule-based filtering and semantic matching.

---

## 🚀 Project Overview

* **Goal:** Automate discovery of zero-fee scholarship opportunities to lower financial barriers for underprivileged students.
* **Approach:**

  1. **Rule-Based Filtering**: Fast, 100% precision filters on explicit fields (application fee, country, field of study).
  2. **Semantic Matching**: Embedding-based search using `all-mpnet-base-v2` to understand free-text queries and scholarship descriptions.
  3. **LLM Paraphrasing**: Local LLaMA 3 (via Ollama CLI) normalizes user queries for better embeddings.
  4. **User Interface**: Two modes:

     * **CLI** for terminal usage.
     * **Streamlit Web App** with Quick Search and Refine forms.

---

## 📁 Repository Structure

```
├── app/
│   ├── data_utils.py    # Data loading & preprocessing
│   ├── recommender.py   # Filtering, semantic search, explanations
│   └── __init__.py
├── data/
│   └── raw/
│       └── scholarships_mock.csv  # Mock dataset (300+ listings)
├── requirements.txt     # Python dependencies
├── main.py              # CLI entry point
├── app.py     # Streamlit application
└── README.md            # Project documentation
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Hassanraza512-red/scholarship-recommender.git
   cd scholarship-recommender
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources** (if running first time)

   ```bash
   python - <<EOF
   ```

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
EOF

````

5. **Ensure Ollama CLI is installed** for local LLaMA 3 paraphrasing: https://ollama.com/docs

---

## ▶️ Usage

### 1. Command-Line Interface (CLI)

```bash
python main.py
````

* **Mode 1: Rule-Based**

  * Answer prompts for degree, field, country, and no-fee preference.
  * Displays matching scholarships in a table.

* **Mode 2: Semantic (AI) Matching**

  * Enter free-text query (e.g., "I want scholarships in the USA").
  * System paraphrases via LLaMA, returns quick country-based results.
  * Prompts for refinement: field, fee, deadline, degree.
  * Displays top 5 recommendations with explanations.

### 2. Web App (Streamlit)

```bash
streamlit run streamlit_app.py
```

* **Rule-Based Tab**: Text inputs for field, country, no-fee filter.
* **Semantic Tab**: Quick Search form & Refine form with live results display.

---

## 🔎 Key Components

* **Data Loading (`data_utils.py`)**

  * Validates schema, reads `scholarships_mock.csv`.
  * Handles missing fields and date parsing.

* **Recommender (`recommender.py`)**

  * `recommend_scholarships`: Pandas filters on strings.
  * `semantic_recommend`: Embedding, cosine similarity, heuristic boosts, explanation.
  * `generate_explanation`: Builds human-readable reasons for each recommendation.

* **LLM Paraphrasing**

  * Uses `query_ollama_cli()` to normalize user queries via LLaMA 3.

* **Streamlit Interface**

  * Cached data/model loading for speed.
  * Forms for quick search and refinement, with session state.

---



## 📈 Future Improvements

* Integrate live scraping for real-time scholarship data.
* Fine-tune embeddings on scholarship-specific text.
* Add OCR for PDF brochures and NER for better date parsing.
* Containerize (Docker + Kubernetes) with a FastAPI backend.
* Collect user feedback for reinforcement learning of preference weights.

---


