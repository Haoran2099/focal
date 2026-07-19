# FOCAL Reproduction Code

This repository contains the clean experiment implementation for
[FOCAL: Filtered On-device Continuous Activity Logging for Efficient Personal
Desktop Summarization](https://arxiv.org/abs/2604.19541). It includes the FOCAL
session method, the two principal baselines, DesktopBench session loading, and
Task Acc, BS-F1, KIR, G-Eval, VCC, and TCS evaluation code.

Benchmark data and screenshots are not stored here. Derived DesktopBench
artifacts are versioned separately in the public
[`HaoranYin/desktopbench`](https://huggingface.co/datasets/HaoranYin/desktopbench)
dataset. The engineering demo is maintained in
[`Haoran2099/major-tom-demo`](https://github.com/Haoran2099/major-tom-demo).

## Methods

- `focal`: metadata-only session planning, selective VLM observations,
  task-isolated memory, and per-task/session summaries.
- `naive_llm_agent_session`: the naive session baseline.
- `brain_vlm_shared_memory_session`: the global/shared-memory baseline.

See [CODE_AUDIT.md](CODE_AUDIT.md) for the code-to-paper audit and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for experiment details.

## Environment

Python 3.11 is the reference environment. Ollama is the default on-device
backend.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
ollama pull qwen3:8b
ollama pull qwen3-vl:8b
ollama pull qwen3-embedding:8b
```

Use `requirements.txt` instead of the lock file when resolving newer compatible
versions. Install `requirements-openai.txt` only for an explicitly selected
OpenAI-compatible inference backend or judge.

## Prepare DesktopBench

The `v0.1.0` dataset is public and does not require a Hugging Face login. Run
the explicit preparation command:

```bash
python scripts/prepare_data.py --data-dir data/desktopbench --revision v0.1.0
```

This downloads the derived release, retrieves the two pinned upstream VideoGUI
revisions, reconstructs screenshots and raw records locally, and verifies the
release. `run.py` never contacts Hugging Face or downloads data.

The data lookup precedence is:

1. `--session-catalog` and `--session-raw-records` when supplied;
2. `--data-root`;
3. `DESKTOPBENCH_ROOT`;
4. `data/desktopbench`.

## Run

```bash
python run.py --backend ollama --method focal \
  --data-root data/desktopbench --output-file outputs/focal.json

python run.py --backend ollama --method naive_llm_agent_session \
  --data-root data/desktopbench --output-file outputs/naive.json

python run.py --backend ollama --method brain_vlm_shared_memory_session \
  --data-root data/desktopbench --output-file outputs/global-memory.json
```

Use `--session-limit 1` for a smoke run and `--resume-from` to continue a saved
result. OpenAI-compatible credentials are accepted only through
`OPENAI_API_KEY`; an optional endpoint can be set with `OPENAI_BASE_URL` or
`--openai-base-url`.

## Evaluate

```bash
python -m pip install -r requirements-eval.txt
python experiments_window_based/session_summary_eval/evaluate_session_summaries.py \
  --gt-file data/desktopbench/experiments_window_based/data/videogui_sessions/gt_summaries/session_catalog_320_gt_summary_gpt.json \
  --method-file focal=outputs/focal.json \
  --method-file naive_llm_agent_session=outputs/naive.json \
  --method-file brain_vlm_shared_memory_session=outputs/global-memory.json \
  --metrics task_acc bertscore

python experiments_window_based/session_summary_eval/report_efficiency.py \
  --method-file focal=outputs/focal.json \
  --method-file naive_llm_agent_session=outputs/naive.json \
  --method-file brain_vlm_shared_memory_session=outputs/global-memory.json \
  --output outputs/efficiency.json
```

Judge-based KIR and G-Eval additionally require `requirements-openai.txt` and a
judge API key in the environment.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall -q experiments_window_based src scripts run.py
```

Code is available under the [MIT License](LICENSE). Cite the paper using
[CITATION.cff](CITATION.cff).
