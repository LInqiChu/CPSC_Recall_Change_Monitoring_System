# CPSC_Recall_Change_Monitoring_System
An end-to-end automated pipeline for tracking changes in U.S. Consumer Product Safety Commission (CPSC) recall data, with a novel contrastive adaptive RAG framework for risk-aware summary generation.

---

## 📌 Project Overview
This system monitors updates to CPSC product recall data, identifies changes across versions, and generates accurate, actionable safety summaries.
Key contributions:
- A robust pipeline for change detection in structured recall datasets
- **Contrastive Adaptive RAG (CARAG)**: A change-aware retrieval-augmented generation method that leverages structured old/new value pairs to produce high-quality, fact-grounded summaries
- A multi-dimensional evaluation framework with a human-annotated gold standard for reliable performance comparison across baselines
- This work is the first to introduce large language models (LLMs) for automated summarization of CPSC product recall change tasks. We establish a simple prompt-only paradigm as a strong baseline for fair and standardized downstream evaluation.
---

## 📂 Repository Structure
```text
CPSC_Recall_Change_Monitoring_System/
├── README.md                # Project overview, setup guide, and usage instructions
├── requirements.txt         # Python dependencies for reproducibility
├── .gitignore               # Specifies intentionally untracked files to ignore
│ 
├── src/                     # Core implementation
│   ├── config.py            # Configuration file
│   ├── CPSC_Recall_System_Final.py  # Main pipeline: change detection, generation, and evaluation
│   ├── exact_merged_original_patches.py  # code fix
│   └── debug/
│       ├── debug_gold_evidence_summary_by_method.csv
│       └── debug_gold_evidence_summary_by_recall.csv
│ 
├── data/                    # Data directory
│   ├── input/                 # Sample CPSC recall snapshots
│   │   ├── cpsc_recalls_20260304.csv
│   │   └── cpsc_recalls_20260430.csv
│   ├── gold/               # Human-annotated gold standard for evaluation
│   │   ├── gold_change_units_for_annotation.csv
│   │   └── frozen_human_gold_current_window_v2.csv
│   └── outputs/            # Generated results, metrics, and debug logs
│       ├── intermediate_statistical_result/
│       │   ├── change_stats.json
│       │   ├── route_digest.json
│       │   └── contrastive_verify_stats.json
│       │
│       ├── mode_outputs/
│       │   ├── summaries.json
│       │   ├── best_final_summary_agentic_rag.txt
│       │   ├── best_final_summary_contrastive_agentic_rag.txt
│       │   ├── final_meta_summary.md
│       │   └── final_meta_summary_meta.json
│       │
│       ├── evaluation_results/
│       │   ├── auto_eval.csv
│       │   ├── claim_eval_sheet.csv
│       │   └── summary_eval_sheet.csv
│
├── demos/
│       ├── cpsc_recall_dashboard.html # Visualization of pipeline flow and results
└──     └── presentation_demo_notes.md
---

```
## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/LInqiChu/CPSC_Recall_Change_Monitoring_System.git
cd CPSC_Recall_Change_Monitoring_System
```

### 2. Install dependencies
Create a virtual environment and install required packages:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure paths
Edit `src/config.py` to point to your local data directories. Example:
```python
DATA_DIR = "./data/"
RAW_DATA_PATH = DATA_DIR + "raw/"
GOLD_DATA_PATH = DATA_DIR + "gold/frozen_human_gold_current_window_v2.csv"
OUTPUT_PATH = DATA_DIR + "outputs/"
```

## 🚀 Quick Start

Run the full pipeline (change detection + summary generation + evaluation)
```bash
python src/CPSC_Recall_System_Final.py
```

## Explore the demo

Open `demos/cpsc_recall_dashboard.html` directly in your browser.


## 🧠 Core Method: Contrastive Adaptive RAG (CARAG)
CARAG is designed for change-aware summary generation, with three key components:

- **Contrastive Structured Generation**  
  Uses old/new value pairs from recall snapshots to generate "Before vs. After" summaries.

- **Three-Tier Self-Verification**  
  Classifies claims into *Fully Supported*, *Partially Supported*, or *Unsupported* based on token overlap with retrieved evidence.

- **Recall Coverage Enforcement**  
  Automatically supplements missing high-priority recalls to maximize coverage.

This design directly addresses the limitations of standard RAG methods for safety monitoring tasks.

---

## 📊 Evaluation Results
All methods are evaluated against a human-annotated gold standard (16 key recall changes).

| Method                      | Final Recall Coverage | Final F1 Score | CU-Weighted Coverage | Enforce Delta (Recall Gain) |
|-----------------------------|-----------------------|----------------|----------------------|------------------------------|
| Prompt-only                 | 0.5000                | 0.6154         | 0.5000               | 0.0000                       |
| Standard RAG                | 0.6875                | 0.6286         | 0.6875               | 0.0625                       |
| Agentic RAG                 | 0.8750                | 0.7179         | 0.8750               | 0.1250                       |
| Contrastive Adaptive RAG    | 0.9375                | 0.7692         | 0.9375               | 0.2500                       |

### Key observations:
- All methods achieve a **100% grounding rate** (no hallucinations) and **route coverage rate**.
- The proposed CARAG outperforms all baselines in recall coverage, F1 score, and risk-focused coverage.
- CARAG shows the largest improvement from post-processing (`enforce_delta = 0.25`), validating the effectiveness of its self-verification loop.

---

## 📁 Reproducibility Notes
- **Sample data**: The `data/` folder contains small subsets of CPSC recall data for demonstration purposes.
- **Full data**: The complete CPSC recall snapshots are not included due to size constraints, but can be downloaded from the CPSC SaferProducts database.
- **Fixed random seed**: All experiments use a fixed random seed to ensure reproducible results.
- **Evaluation framework**: The multi-dimensional evaluation logic is integrated into the main script and outputs all metrics to `data/outputs/`.

---

## 📜 Commit History (Example)
- feat: add core change detection pipeline
- feat: implement all four RAG baselines (Prompt-only, RAG, Agentic RAG, CARAG)
- feat: add three-tier self-verification and coverage enforcement
- eval: implement multi-dimensional evaluation metrics
- docs: complete README with setup and usage instructions
---
## 📄 License
This project is for academic use only.
If you have any questions, please contact the author.
