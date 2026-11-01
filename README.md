# Intelligent Resume Screening System – Prototype

A modular, explainable, and evidence-grounded Resume Screening System in Python. Designed as an HR decision-support tool, the system evaluates candidate resumes against a Job Description (JD) using a hybrid matching approach that combines exact canonical terms, known aliases, fuzzy matching, local semantic sentence embeddings, taxonomy relationships, and contextual implementation evidence.

---

## 🌟 Key Features

- **No Keyword Hallucination / Black Box:** Does not just count keywords. Matches candidate evidence sentences directly to JD requirements, identifying the source section, evidence depth, and match confidence.
- **Strict Distinction between Equivalency and Related Skills:** 
  - Recognizes synonyms and aliases (e.g., `NLP` ↔ `Natural Language Processing`, `K8s` ↔ `Kubernetes`, `SFDC` ↔ `Salesforce`).
  - Correctly tags related but distinct skills as **`Related Skill – Not Equivalent`** (e.g., candidate having `Angular` when `React` is required, or `MySQL` instead of `PostgreSQL`, or `Flask` instead of `FastAPI`).
- **Contextual Evidence Depth:** Differentiates between a bare skill mention in a skills list (`Listed Only`) versus execution in work history or projects with quantifiable metrics (`Strong Implementation`).
- **Completely Local & Free / Open-Source:** No paid APIs (no OpenAI, Gemini, Claude). Zero Ollama or external database dependencies. Runs locally on CPU/Apple Silicon.
- **Explainable Multi-Factor Scoring:** Fully configurable weights for Required Skills Coverage, Preferred Skills Coverage, Evidence Depth, and Experience/Project Relevance. Every score includes a mathematical formula and human recruiter rationale.
- **Interactive Streamlit Interface:** 
  1. *JD Ingestion & Review:* Paste or upload JD, extract requirements into Required vs Preferred, and edit/confirm before screening.
  2. *Resume Batch Screening:* Upload multiple PDF resumes and track progress.
  3. *Candidate Comparison & Reports:* Comparative overview table, requirement-level evidence drilldown, and exports to CSV and JSON.

---

## 🏗️ System Architecture

```
resume-screening-system/
│
├── app.py                                # Streamlit Web UI (3 core workflow sections)
│
├── modules/
│   ├── knowledge_base/                   # Multi-industry skill taxonomy & relationships
│   │   ├── models.py                     # SkillEntry, SkillRelationship, RelationshipType
│   │   └── taxonomy.py                   # SkillTaxonomy manager (lookup, aliases, logging)
│   │
│   ├── resume_processing/                # Multi-stage PDF ingestion
│   │   ├── models.py                     # ProcessedResume, ResumeSection, EvidenceSentence
│   │   └── extractor.py                  # PyMuPDF extractor, OCR fallback, section segmenter
│   │
│   ├── jd_processing/                    # Structured JD requirement extractor
│   │   ├── models.py                     # JobDescription, RequirementItem
│   │   └── parser.py                     # Rule-based/NLP parser with editable requirements
│   │
│   ├── matching_engine/                  # Central matching intelligence
│   │   ├── models.py                     # MatchClassification, RequirementMatchResult
│   │   ├── exact_matcher.py              # Word-boundary canonical and alias matcher
│   │   ├── fuzzy_matcher.py              # RapidFuzz similarity with length guards
│   │   ├── semantic_matcher.py           # SentenceTransformers (BAAI/bge-small-en-v1.5)
│   │   ├── contextual_matcher.py         # Evidence depth & action verb analyzer
│   │   └── hybrid_engine.py              # Central orchestrator combining all 6 methods
│   │
│   ├── scoring/                          # Multi-factor explainable scoring
│   │   ├── models.py                     # CandidateEvaluation, ScoringWeights, Recommendation
│   │   └── evaluator.py                  # Formula calculator and HR recommendation generator
│   │
│   └── reporting/                        # Report generation & exports
│       └── reporter.py                   # Single candidate report, comparison DF, CSV/JSON
│
├── data/
│   ├── knowledge_base/
│   │   └── skills_taxonomy.json          # Structured taxonomy across 8 industries
│   ├── evaluation_dataset/
│   │   └── labeled_pairs.json            # 20 ground-truth labeled evaluation pairs
│   └── test_resumes/                     # Synthetic test PDFs across matching levels
│       ├── alex_morgan_ai_engineer.pdf   # Strong match
│       ├── jordan_lee_backend_dev.pdf    # Partial match
│       ├── taylor_reed_frontend_dev.pdf  # Related skill (Angular vs React)
│       ├── sam_rivera_marketing.pdf      # Unrelated domain
│       └── morgan_vance_minimal.pdf      # Minimal text / extraction warning
│
├── evaluation/
│   └── benchmark_embeddings.py           # Benchmark script for embedding models
│
├── output/                               # Generated reports and benchmark results
│   ├── embedding_benchmark_results.json
│   ├── example_screening_reports.json
│   └── example_candidate_comparison.csv
│
├── scripts/
│   └── run_sample_screening.py           # End-to-end headless CLI screening script
│
├── tests/
│   └── test_screening_system.py          # 11 automated test cases for all requirements
│
├── requirements.txt
└── README.md
```

---

## 🔬 Embedding Model Evaluation Results

To select the most accurate and efficient embedding model for local execution, we benchmarked candidate models on `data/evaluation_dataset/labeled_pairs.json` (20 labeled requirement-evidence pairs across AI/ML, Software, PM, Marketing, Sales, Operations, and Finance).

Results at Decision Threshold `0.65`:

| Model | Precision | Recall | F1 Score | Avg Latency / Pair | False Positives | False Negatives | Selection Decision |
|---|---|---|---|---|---|---|---|
| **`BAAI/bge-small-en-v1.5`** | **94.12%** | **100.0%** | **96.97%** | **4.3 ms** | 1 (Resolved by Taxonomy) | 0 | **Selected as Default** |
| `sentence-transformers/all-MiniLM-L6-v2` | 100.0% | 25.0% | 40.0% | 37.1 ms | 0 | 12 | Rejected (High FN rate) |

### Key Findings & Hybrid Resolution
- `all-MiniLM-L6-v2` compresses cosine similarities for semantic paraphrases between `0.45` and `0.60`, resulting in 12 false negatives at threshold `0.65`.
- `BAAI/bge-small-en-v1.5` delivers near-instant inference (4.3ms/pair) with 100% recall.
- **The Single Raw Semantic False Positive:** The sentence *"Built enterprise web dashboards using Angular 14..."* scored `0.737` cosine similarity against *"Experience building frontend applications using React and TypeScript"*.
- **How our Hybrid Engine Solves It:** The Hybrid Engine checks the Knowledge Base before finalizing semantic matches. Recognizing that `Angular` is an explicit `related_not_equivalent` alternative to `React`, it intercepts the semantic match and correctly classifies it as **`Related Skill – Not Equivalent`**.

---

## 🚀 Setup and Execution Instructions

### 1. Prerequisites
- Python 3.10 or 3.11
- Operating System: macOS, Linux, or Windows

### 2. Environment Setup
```bash
# Clone or navigate to the project directory
cd /Users/apple/resume_screening

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Pytest Suite
Run the 11 automated test cases verifying canonical matches, aliases, typos (fuzzy matching), related skills (non-equivalent), project evidence, missing skills, and cross-industry parsing:
```bash
.venv/bin/pytest tests/ -v
```

### 4. Run Embedding Model Benchmark
Benchmark local SentenceTransformer models on the labeled dataset:
```bash
.venv/bin/python evaluation/benchmark_embeddings.py
```
Results are saved to `output/embedding_benchmark_results.json`.

### 5. Run Sample Screening via CLI
Run end-to-end screening of the 5 test resumes against a sample AI Engineer JD:
```bash
.venv/bin/python scripts/run_sample_screening.py
```
Deliverables generated in `output/`:
- `output/example_screening_reports.json`
- `output/example_candidate_comparison.csv`

### 6. Launch the Streamlit Web Application
```bash
.venv/bin/streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Using the Streamlit Interface

1. **Step 1: Job Description**
   - Select a sample JD template or paste your own.
   - Click **"Extract Structured Requirements"**.
   - Review the extracted requirements in the interactive data editor. Edit priority (`Required` vs `Preferred`), requirement types, or mapped canonical skills.
   - Click **"Confirm Job Description"**.
2. **Step 2: Resume Screening**
   - Upload candidate PDF resumes or select pre-generated test resumes.
   - Click **"Start Screening Resumes"** to run batch extraction and hybrid matching.
3. **Step 3: Candidate Comparison & Reports**
   - View the ranked candidate comparison table with overall scores, required/preferred coverage, and recommendations.
   - Download the comparison table as CSV or detailed candidate reports as JSON.
   - Select any candidate to drill down into requirement-by-requirement evidence sentences, source resume sections, confidence indicators, and recruiter decision rationales.

---

## ⚖️ Scoring Formula & Shortlisting Logic

Scoring is transparent and configurable via the sidebar:
$$\text{Overall Score} = (C_{\text{req}} \times W_{\text{req}}) + (C_{\text{pref}} \times W_{\text{pref}}) + (Q_{\text{evid}} \times W_{\text{evid}}) + (R_{\text{exp}} \times W_{\text{exp}})$$

- **$C_{\text{req}}$ (Required Coverage):** Weighted score of required items. Exact matches receive 1.0, Strong Semantic 0.90, Fuzzy 0.85, Related Skill 0.40, Not Found 0.0.
- **$C_{\text{pref}}$ (Preferred Coverage):** Weighted score of bonus / preferred items.
- **$Q_{\text{evid}}$ (Evidence Quality):** Evaluates depth (`Strong Implementation` with action verbs and metrics = 1.0, `Experience Evidence` = 0.85, `Project Evidence` = 0.75, `Listed Only` = 0.30).
- **$R_{\text{exp}}$ (Experience / Project Relevance):** Proportion of matches grounded in professional work history or projects rather than bare keyword lists.

### Recommendations for Human Recruiters
- **Strongly Recommended for Interview:** Required Coverage $\ge 80\%$ with solid evidence depth ($\ge 50\%$).
- **Recommended for Human Review:** Required Coverage $\ge 60\%$.
- **Borderline - Additional Screening Required:** Required Coverage $40\% - 59\%$ or presence of related non-equivalent competencies.
- **Low Alignment with Requirements:** Required Coverage $< 40\%$.

---

## ⚠️ Known Limitations & Future Extensibility

1. **OCR on Low-Quality Scans:** Basic OCR fallback via `pytesseract` is integrated. If Tesseract is not installed on the host OS, a warning is logged. Integrating an embedded lightweight OCR engine like `easyocr` or `PaddleOCR` can improve handling of heavily scanned PDFs.
2. **Complex Multi-Column PDF Formatting:** Multi-column layouts can occasionally interleave sentence fragments across columns. Enhancing PyMuPDF with reading-order layout analysis (`pymupdf-layout`) is planned for company-grade scale.
3. **Future Extensibility:**
   - Database layer: Seamless migration to SQLite or PostgreSQL by swapping `data/knowledge_base/skills_taxonomy.json` for an ORM repository pattern.
   - Multi-JD screening sessions: The matching engine accepts `(ProcessedResume, JobDescription)` independently of state, making multi-JD routing ready out of the box.
   - Optional local SLM (e.g., Llama 3.2 3B or Mistral 7B) for generating conversational executive summaries.
