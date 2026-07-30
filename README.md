# AI Challenge 2026 — Dynamic Multi-Agent Routing

---

## Documentation

For a detailed breakdown of the system architecture and implementation plan, please see:
- [Architecture (English)](ARCHITECTURE.md) | [Kiến trúc (Tiếng Việt)](ARCHITECTURE_VI.md)
- [Implementation Plan (English)](IMPLEMENTATION_PLAN.md) | [Kế hoạch Triển khai (Tiếng Việt)](IMPLEMENTATION_PLAN_VI.md)

---

## Architecture Overview

This project implements an **Agent-guided Multimodal Pipeline** with **Temporal Event Reasoning** to solve the AIC 2026 Known-Item Search (KIS) and Conversational KIS challenges.

The system is split into two phases:
1. **Offline Indexing:** Videos are chunked using TransNet V2, audio is transcribed by Whisper, and keyframes are embedded by dual encoders (SigLIP and BEiT-3) and OCR'd by Gemini.
2. **Online Retrieval:** A GPT-4o Agent Router expands text queries and routes weights across Vector Stores (TurboVec/FAISS) and Text Stores (Elasticsearch). Temporal Beam Search and BLIP-2 cross-encoding rerank the final results.

> **Note:** For the full Mermaid diagrams, data flows, and tech stack details, please see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Directory Structure

```text
├── configs/
│   └── config.yaml             # All hyperparameters (no hardcoding)
├── src/
│   ├── agents/                 # Shared wrappers: asr, ocr, visual, base
│   ├── retrieval/              # Indexing: video_indexer, vector_store
│   ├── routing/                # Online: classifier, dispatcher
│   ├── ui/                     # Streamlit frontend (app.py)
│   ├── eval.py                 # Evaluation harness & metrics
│   └── inference.py            # End-to-end online search entry point
├── Dockerfile                  # Containerization
├── docker-compose.yml          # Multi-container orchestration (ES, etc.)
├── pyproject.toml              # Modern Python packaging
└── README.md
```

---

## Team Responsibilities

| Team | Module | Sprint 1 Task |
|:---|:---|:---|
| **Team 1 (Data & Indexing)** | Data Pipeline, Eval, Video Retrieval | Build dataset loader; query labeling script; Set up TurboVec/FAISS indices; frame sampling pipeline |
| **Team 2 (Retrieval & Serving)** | Routing, Dispatcher, Agents | EDA on query types; build dynamic dispatcher; Integrate Gemini, Whisper, SigLIP APIs |

### Git Workflow

- Each member works on `feature/<name>-<module>` branch
- PR into `develop` requires at least 1 reviewer
- All config goes into `configs/config.yaml` — no hardcoded values in source files

### Commit Convention

Format: `<type>(<scope>): <short description>`

| Type | When to use | Example |
|:---|:---|:---|
| `feat` | Add new feature or module | `feat(routing): add dynamic dispatcher` |
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
git commit -m "exp(routing): prototype dispatcher logic in notebook"
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
pip install -e .              # Install via pyproject.toml
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

- Zhai et al. (2023). *Sigmoid Loss for Language Image Pre-Training* — SigLIP
- Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* — Whisper
