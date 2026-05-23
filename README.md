# LINGUA

**Bridging the Grounding Gap in VideoQA via Typed Memory for Language-based Belief-State Reasoning**

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-b31b1b.svg)](https://openreview.net/forum?id=TBD)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Backbone-Gemma3--4B-orange.svg)](https://ollama.com/library/gemma3)

LINGUA is a memory-based agent for grounded Video Question Answering. It performs the entire reasoning process in an explicit **linguistic belief state**, replacing opaque visual embeddings with timestamped narratives, frame-semantic affordances, and procedural scripts. Hypotheses are checked against video evidence at inference time through a **Belief–Action–Verification (BAV)** loop, and procedural reliability is tracked with Bayesian updates — enabling continual learning without gradient descent.

Built on a single **Gemma3-4B** backbone (4-bit, served locally via Ollama), LINGUA matches or beats much larger systems on five VideoQA benchmarks while running ~2.6× faster than dense-frame methods.

<p align="center">
  <img src="Figures/Framework.png" width="780" alt="LINGUA architecture"/>
</p>

---

## Highlights

- **Closes the grounding gap.** 42.3 % `Acc@GQA` on NExT-GQA (correct answer **and** IoU ≥ 0.5 temporal localization).
- **Strong long-video reasoning.** 68.5 % overall / 69.4 % long-subset on Video-MME with a 4 B model.
- **Continual learning, no gradients.** Accuracy improves from 45.2 % (first 10 videos) to 82.4 % at 1 000 videos and 84.2 % at 2 000 videos, with no catastrophic forgetting.
- **Fully interpretable.** Every step of inference is a natural-language artifact: percepts, beliefs, hypotheses, verification outcomes, and reflection traces.
- **Runs locally.** No proprietary APIs; the entire pipeline is open-weights and self-hosted.

---

## How it works

LINGUA combines five paper-faithful mechanisms (Section 3 of the paper):

| Component | Role |
|---|---|
| **Event-Driven Perception** | VideoMAE-v2 picks frames with semantic change; YOLOv8 triggers affordance-bearing frames. Retains 8–12 % of frames while preserving 94 % of question-relevant events. |
| **Typed Memory** | Episodic narratives `⟨Agent, Action, Patient, [tₛ,tₑ], Goal, Outcome, …⟩`, semantic affordances (FrameNet-style), and procedural scripts `⟨𝒢, Ψ, Π, Φ, {μᵢ,σᵢ}⟩`. |
| **Belief–Action–Verification loop** | Retrieves evidence, ranks scripts by expected utility `EU = Rel·E[ρ] − Risk + λ·H(Beta)`, then verifies postcondition coverage and temporal consistency. |
| **Meta Reflection** | Triggers on 3 + consecutive failures, postcondition coverage < 0.3, or semantic drift > 0.7; diagnoses linguistic / temporal / causal failures. |
| **Bayesian Continual Learning** | Each script keeps a Beta(α,β) reliability posterior; verification outcomes drive α += 1 / β += 1. |

---

## Repository structure

```
.
├── LINGUA_Agentic_Model.py     # Reference implementation (single-file)
├── Figures/                    # Architecture and trace figures
├── data/                       # Benchmark loaders (placeholders)
├── scripts/                    # Eval / continual-learning runners
├── requirements.txt
├── LICENSE
└── README.md
```

The implementation is intentionally kept in a single file so that every component (perception, memory, BAV, reflection, verification) can be inspected in one place. Each class header cites the paper section it implements.

---

## Installation

### 1. Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (recommended; CPU works but is slow)
- **Ollama** (for the Gemma3-4B backbone) — see <https://ollama.com>

### 2. Clone and install

```bash
git clone https://github.com/<org>/lingua.git
cd lingua

python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A minimal `requirements.txt`:

```text
torch>=2.1
transformers>=4.40
ultralytics>=8.1
sentence-transformers>=2.7
spacy>=3.7
opencv-python>=4.9
numpy>=1.26
pandas>=2.2
scipy>=1.11
scikit-learn>=1.4
Pillow>=10.0
ollama>=0.1.7
```

### 3. Pull the Gemma3-4B model

```bash
ollama pull gemma3:4b
ollama serve     # leave running in a separate terminal
```

> **Windows note.** The code includes a `setup_ollama_windows()` helper that locates the Ollama binary and starts the daemon automatically. On Linux / macOS, run `ollama serve` once and the agent will connect to it.

---

## Quick start

```python
from LINGUA_Agentic_Model import LINGUAAgent, LINGUAConfig

agent = LINGUAAgent(LINGUAConfig())

# 1. Process a video once — event-driven perception, episodic memory,
#    semantic consolidation, procedural mining.
agent.process_video("path/to/video.mp4")

# 2. Answer a question through the Belief–Action–Verification loop.
result = agent.answer_question_BAV("What is the person doing after opening the fridge?")

print(result["answer"])
print("Confidence (EU):", result["confidence"])
print("Postcondition coverage:", result["verification"]["post_coverage"])
print("Temporal consistency:",  result["verification"]["temporal_consistency"])
```

Every dictionary returned by the agent contains the **verification trace** (postcondition coverage, temporal-consistency score, grounded-or-not flag, and the selected script). These are the artifacts used to compute `Acc@GQA` in the paper.

---

## Reproducing paper results

### NExT-QA / NExT-GQA

```bash
python -m scripts.eval_nextgqa \
    --dataset-path ./data/nextgqa \
    --split val \
    --output ./runs/nextgqa
```

The script loads `val_clips.json`, runs LINGUA over each clip, and reports:

- Answer accuracy
- `Acc@GQA` (answer correct **and** IoU ≥ 0.5)
- Recall@IoU{0.3, 0.5}, mIoU, mIoP

### Video-MME

```bash
python -m scripts.eval_videomme \
    --dataset-path ./data/videomme \
    --split test \
    --output ./runs/videomme
```

### Continual learning over 1 000 / 2 000 videos

```bash
python -m scripts.continual_learning \
    --dataset-path ./data/nextqa \
    --num-videos 2000 \
    --bucket-size 10 \
    --output ./runs/continual
```

`reliability_tracker.json` and `continual_learning.json` are written incrementally so the run can be resumed after interruption.

---

## Configuration

All hyperparameters live in `LINGUAConfig` and **exactly match the paper**.

| Symbol | Value | Field |
|---|---|---|
| τ_Δ (semantic change) | 0.15 | `SEMANTIC_CHANGE_THRESHOLD` |
| γ_aff (affordance retrieval) | 0.75 | `FUZZY_MATCH_THRESHOLD` |
| γ_post (postcondition match) | 0.80 | `POSTCONDITION_THRESHOLD` |
| τ_EU (utility threshold) | 0.40 | `EU_THRESHOLD` |
| λ_info (EU exploration) | 0.10 | `LAMBDA_INFO` |
| Δt_merge (episodic merging) | 2.0 s | `TEMPORAL_GAP_THRESHOLD` |
| Episodic-merging similarity | 0.85 | `EPISODIC_MERGE_SIMILARITY` |
| n_min (schema validation) | 5 | `MIN_SCRIPT_INSTANCES` |
| σᵢ / μᵢ (temporal variance) | 0.5 | `MAX_TEMPORAL_VARIANCE` |
| n_contrast | 3 | `MIN_CONTRASTIVE_EXAMPLES` |
| Reflection failure window | 3 | `REFLECTION_FAILURE_COUNT` |
| Reflection coverage threshold | 0.30 | `REFLECTION_COVERAGE_THRESHOLD` |
| Semantic-drift threshold | 0.70 | `SEMANTIC_DRIFT_THRESHOLD` |
| Beta prior | (1, 1) | `PRIOR_ALPHA`, `PRIOR_BETA` |
| Temperature (VL + text) | 0.10 | `VLM_TEMPERATURE`, `LLM_TEMPERATURE` |

To run an ablation, override only the field of interest:

```python
config = LINGUAConfig()
config.LAMBDA_INFO = 0.0           # disable exploration
agent = LINGUAAgent(config)
```

---

## Results

### Main results

| Benchmark | Metric | LINGUA (4B) |
|---|---|---|
| NExT-QA | Accuracy | **82.4 %** |
| NExT-GQA | `Acc@GQA` (IoU ≥ 0.5) | **42.3 %** |
| Video-MME (all) | Accuracy | **68.5 %** |
| Video-MME (long) | Accuracy | **69.4 %** |

### Continual learning (no gradient updates)

| Stream position | Accuracy |
|---|---|
| First 10 videos | 45.2 % |
| 1 000 videos | 82.4 % |
| 2 000 videos | 84.2 % |

### Efficiency

- **2.6× faster** than dense-frame baselines at comparable or higher accuracy
- Single-GPU inference with 4-bit quantized Gemma3-4B (~3 GB VRAM)

Detailed per-task breakdowns, ablations (memory components, BAV loop, scale 4B → 11B), and execution traces are in the paper appendix.

---

## Citation

If you use LINGUA in academic work, please cite:

```bibtex
@inproceedings{forouzandeh2026lingua,
  title     = {Bridging the Grounding Gap in {VideoQA} via Typed Memory for Language-based Belief-State Reasoning},
  author    = {Forouzandeh, Saman and Peng, Wei and Yu, Han and Jalili, Mahdi},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

> Update the author order, affiliations, and ICML URL once the camera-ready DOI is assigned.

---

## License

This project is released under the **MIT License** — see [LICENSE](LICENSE).

Third-party components retain their respective licenses:

- **Gemma 3** — Google DeepMind, Gemma Terms of Use
- **VideoMAE-v2** — MIT
- **YOLOv8** — AGPL-3.0 (Ultralytics)
- **spaCy**, **sentence-transformers** — MIT / Apache-2.0

If your downstream use is incompatible with AGPL, replace YOLOv8 with an MIT-licensed detector (the affordance-attention module is the only point of contact).

---

## Acknowledgments

We thank the ICML 2026 reviewers and Area Chair for constructive feedback that materially improved the paper.

LINGUA builds on the open-weights work of the **Gemma 3** team, the **VideoMAE-v2** authors, and the **NExT-QA / NExT-GQA / Video-MME** dataset curators.

---

## Contact

Questions, issues, or reproduction requests are welcome via GitHub Issues. For research collaboration, please email the corresponding author listed in the paper.
