# AI Challenge 2026 — Dynamic Multi-Agent Routing via Queuing Theory

**Research Topic 3.1:** How can heterogeneous multimodal queries be dynamically routed to specialized agent worker pools using queuing theory (M/M/c model) to minimize end-to-end latency in large-scale interactive video retrieval systems?

---

## Architecture Overview

```
Multimodal Query (text / image / audio)
          │
          ▼
┌─────────────────────┐
│  Query Classifier   │  Lightweight MLP — classifies query type & complexity
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Dynamic Dispatcher │  M/M/c queuing model — routes to optimal agent pool
└──┬──────┬─────┬─────┘
   │      │     │
   ▼      ▼     ▼
┌──────┐ ┌─────┐ ┌────────┐
│ OCR  │ │ ASR │ │ Visual │
│Agent │ │Agent│ │ Agent  │
│Gemini│ │Whis.│ │ SigLIP │
└──────┘ └─────┘ └────┬───┘
                       ▼
              ┌─────────────────┐
              │  FAISS Vector   │
              │  Store (GPU)    │
              └─────────────────┘
```

---

## Directory Structure

```
├── data/
│   ├── raw/
│   │   ├── videos/             # Raw video files (.mp4)
│   │   └── queries/            # Query dataset (JSON)
│   └── processed/
│       ├── embeddings/         # SigLIP frame embeddings
│       ├── transcripts/        # Whisper ASR output
│       └── ocr/                # Gemini OCR output
├── notebooks/
│   ├── 01_eda_queries.ipynb    # Query distribution analysis
│   └── 02_queuing_analysis.ipynb  # M/M/c model prototyping
├── src/
│   ├── agents/                 # Specialized agents (OCR, ASR, Visual)
│   ├── routing/                # Query classifier + dynamic dispatcher
│   ├── retrieval/              # FAISS vector store + video indexer
│   ├── data_loader.py          # Dataset & preprocessing
│   ├── model.py                # Model architectures
│   ├── train.py                # Training pipeline
│   └── inference.py            # End-to-end inference
├── configs/
│   └── config.yaml             # All hyperparameters (no hardcoding)
├── weights/                    # Model checkpoints
├── requirements.txt
└── README.md
```

---

## Team Members & Responsibilities

| # | Name | Module | Sprint 1 Task |
|:---:|:---|:---|:---|
| 1 | Le Nguyen Khoi | Routing & Dispatcher | EDA on query types; prototype M/M/c model |
| 2 | Pham Viet Truong | Video Retrieval | Set up FAISS index; frame sampling pipeline |
| 3 | Truong Hoang Thong | Agents (OCR/ASR/Visual) | Integrate Gemini, Whisper, SigLIP APIs |
| 4 | Pham Huu Huy | Data Pipeline & Eval | Build dataset loader; query labeling script |

### Git Workflow

- Each member works on `feature/<name>-<module>` branch
- PR into `develop` requires at least 1 reviewer
- All config goes into `configs/config.yaml` — no hardcoded values in source files

### Commit Convention

Format: `<type>(<scope>): <short description>`

| Type | When to use | Example |
|:---|:---|:---|
| `feat` | Add new feature or module | `feat(routing): add Erlang-C dispatcher` |
| `fix` | Bug fix | `fix(asr): handle empty audio file` |
| `data` | Data scripts, labeling, preprocessing | `data(loader): implement QueryDataset` |
| `exp` | Experiment, notebook, EDA | `exp(eda): query type distribution analysis` |
| `refactor` | Code restructure, no behavior change | `refactor(agents): extract base timing logic` |
| `chore` | Config, deps, tooling | `chore(deps): add decord to requirements` |
| `docs` | README, docstrings | `docs(readme): update sprint plan` |

**Scope** = module name: `routing`, `agents`, `retrieval`, `loader`, `model`, `train`, `config`

```bash
# Examples
git commit -m "feat(agents): implement OCRAgent with Gemini Vision API"
git commit -m "data(loader): add query labeling script for sprint 1"
git commit -m "exp(queuing): prototype M/M/c E[W] simulation notebook"
git commit -m "fix(vector-store): normalize embeddings before FAISS add"
```

---

## Setup

```bash
git clone https://github.com/AIVIETNAM-AIO-herk30/AI_Challenge_2026.git
cd AI_Challenge_2026
python -m venv venv
source venv/bin/activate      # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

Set API keys:
```bash
export GOOGLE_API_KEY="your-gemini-key"
```

---

## Sprint Plan

| Sprint | Goal | Duration |
|:---|:---|:---|
| **Sprint 1** | Foundation & EDA | Week 1–2 |
| Sprint 2 | Core agents + classifier training | Week 3–4 |
| Sprint 3 | Dispatcher integration + benchmarking | Week 5–6 |
| Sprint 4 | End-to-end evaluation + paper writing | Week 7–8 |

---

## References

- Kleinrock, L. (1975). *Queueing Systems, Vol. 1* — M/M/c model
- Zhai et al. (2023). *Sigmoid Loss for Language Image Pre-Training* — SigLIP
- Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* — Whisper
