# Progression-aware Reasoning Origin (PRO)

This repository is the official implementation for the paper: **The Error You See Is Not the Error You Made: Progression-aware Reasoning Origin for Reasoning Error Localization** ([arXiv:2609.33297](https://arxiv.org/abs/2609.33297)).

PRO is a training-free framework that jointly models incoming support and outgoing compatibility to distinguish the true error origin from its propagated manifestations, enabling reliable first-error localization even under error propagation.

## Abstract

Verifying multi-step LLM reasoning requires more than determining whether a trace is correct: a useful verifier should identify where the reasoning first goes wrong. However, existing holistic methods provide little positional evidence, while forward sequential verification often treats the first rejected step as the error source. Under error propagation, this assumption can fail, since an earlier mistake may remain locally plausible and become observable only through its downstream consequences. We therefore rethink reasoning verification as a progression-aware error-source localization problem: rather than asking only where a reasoning trace first appears inconsistent, we ask which earlier step best explains how that inconsistency emerges along the trajectory. Based on this view, we propose **Progression-aware Reasoning Origin (PRO)**, a training-free framework for first-error localization. PRO jointly models incoming support from the preceding context and outgoing compatibility with subsequent reasoning, selectively refines regions where these signals disagree, and finally performs detector-conditioned source attribution with intervention-based evidence to distinguish the true error origin from its propagated manifestations. We further formalize the gap between forward rejection and structural exposure, showing why incoming-side evidence alone is insufficient for reliable localization under error propagation. Experiments across open-form, medical, and structured reasoning tasks demonstrate consistent improvements over strong verification baselines, supporting progression-aware source attribution as a more faithful formulation of reasoning verification.

## How PRO Works

PRO operationalizes progression-aware error-source localization through a three-stage pipeline that separates error exposure from error attribution.

Figure: The PRO Three-Stage Localization Pipeline.

1. **Bidirectional Verification**: Each reasoning step is evaluated with both incoming support (forward: is this step justified by the problem and verified prefix?) and outgoing compatibility (backward: can this step consistently support the reasoning that follows?). Disagreement between these signals flags structurally ambiguous regions.

2. **Mismatch-Triggered Refinement**: When incoming and outgoing evidence disagree, PRO selectively increases verification granularity through edge-level refinement, resolving ambiguity without prematurely committing to an error source.

3. **Detector-Conditioned Source Attribution**: Conditioned on the detector's verification pattern, PRO compares remaining source candidates with intervention-based evidence (rewrite-and-verify) to distinguish the step that introduces the error from downstream steps that merely inherit or expose it.

This framework separates **error exposure** from **error attribution** by reasoning over how verification evidence evolves along the reasoning trajectory, enabling reliable first-error localization even under error propagation.

## Methods

| Method | Script | Description |
|--------|--------|-------------|
| **Baseline** | `processbench_baseline.py` | Holistic review: LLM reviews all steps at once and outputs the first error index. |
| **GoV** | `processbench_GoV.py` | Graph of Verification: sequential step-by-step forward verification, stopping at the first detected error. |
| **PARC** | `processbench_PARC.py` | Premise-Augmented Reasoning Chains: predicts premises per step, then runs standalone error check + contextual consistency check; fuses both for error localization. |
| **PRO** | `processbench_PRO.py` | Bidirectional verification → LLM ranks top-5 suspicious steps → rewrites and verifies each candidate → source reranker selects the first error origin. |

## Repository Structure

```
.
├── ProcessBench/
│   ├── data_download.py              # Downloads and prepares the ProcessBench dataset
│   ├── processbench_baseline.py      # Baseline (holistic) verification
│   ├── processbench_GoV.py           # GoV (sequential) verification
│   ├── processbench_PARC.py          # PARC (premise-augmented) verification
│   └── processbench_PRO.py           # PRO (bidirectional + top-5 rewrite + source rerank)
│
├── .env.example                      # Example environment file for API keys
├── requirements.txt                  # Python package dependencies
└── README.md
```

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Wan1G1uo/Progression-aware-Reasoning-Origin.git
   cd Progression-aware-Reasoning-Origin
   ```

2. **Install dependencies:**
   Create a virtual environment and install the required packages.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file by copying the example file:

   ```bash
   cp .env.example .env
   ```

   Open the `.env` file and add your API key:

   ```env
   OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

   # Optional: If using a custom endpoint, specify the base URL
   OPENAI_API_BASE="YOUR_OPTIONAL_API_BASE_URL_HERE"
   ```

   > **Note**: The LLM model name (e.g., `Qwen2.5-32B-Instruct`) is hardcoded in the Python scripts. To use a different model, you must modify the `API_MODEL` variable inside each script.

## Running the Experiments

The paper evaluates PRO across three reasoning tasks. This repository provides the ProcessBench (open-form mathematical reasoning) experiments.

### ProcessBench: Open-Form Mathematical Reasoning

ProcessBench evaluates reasoning verification on natural-language math solutions generated by LLMs. All methods operate on paragraph-level reasoning blocks.

  * **Step 1: Download and Prepare Data**
    This script downloads the `gsm8k` split from the `Qwen/ProcessBench` dataset on Hugging Face and prepares it.

    ```bash
    python ProcessBench/data_download.py
    ```

    This will create the `gsm8k.csv` file in the `ProcessBench` directory.

  * **Step 2: Run Verification**
    Run the scripts to get verification results for the ProcessBench dataset.

    ```bash
    # Run Baseline (holistic verification)
    python ProcessBench/processbench_baseline.py

    # Run GoV (paragraph-by-paragraph verification)
    python ProcessBench/processbench_GoV.py

    # Run PARC (premise-augmented verification)
    python ProcessBench/processbench_PARC.py

    # Run PRO (bidirectional verification + top-5 rewrite + source rerank)
    python ProcessBench/processbench_PRO.py
    ```

    The scripts will generate output CSVs containing the detailed verification results and predicted first-error indices.

  * **Step 3: Evaluate Results**
    Compare the predicted error index against the ground-truth `label` column to compute Correct Accuracy, Error Accuracy, and F1 score.

## Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@misc{wang2025errorseeerror,
      title={The Error You See Is Not the Error You Made: Progression-aware Reasoning Origin for Reasoning Error Localization}, 
      author={Yiguo Wang and Ziyuan Yang and Yi Zou and Dan Lin and Rongsheng Li and Yi Zhang},
      year={2025},
      eprint={2609.33297},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.33297},
}
```

The Graph of Verification baseline is based on:

```bibtex
@misc{fang2025graphverificationstructuredverification,
      title={Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs}, 
      author={Jiwei Fang and Bin Zhang and Changwei Wang and Jin Wan and Zhiwei Xu},
      year={2025},
      eprint={2506.12509},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.12509}, 
}
```