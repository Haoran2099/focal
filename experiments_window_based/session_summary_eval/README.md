# Session Summary Evaluation

This directory contains the unified evaluation entrypoint for session-level summaries.

Scripts:

- evaluate_session_summaries.py: unified entrypoint for task accuracy, BERTScore, task-aware KIR, and G-Eval
- report_efficiency.py: paper-aligned VCC, TCS, and Brain/VLM token breakdown
- common.py: shared loading, merge, and judge utilities
- eval_config.json: shared evaluation config

Important assumptions:

- The evaluated text is session_summary, not per-task summary.
- Task accuracy, BERTScore, KIR, and G-Eval are all computed on completed sessions only.
- The denominator for each method is its own completed-session count after GT alignment and optional eval_config filtering.
- If per-task summaries are missing from the result JSON, the unified script falls back to splitting session_summary by paragraphs for task-aware KIR matching.

Clean-package defaults:

- data/session_catalog_320_gt_summary_gpt.json

Default input files:

- focal:
  - outputs/focal.json
- naive_llm_agent_session:
  - outputs/naive.json
- brain_vlm_shared_memory_session:
  - outputs/global-memory.json

Default run after placing external data and generated results at the paths above:

```bash
python experiments_window_based/session_summary_eval/evaluate_session_summaries.py --metrics task_acc bertscore
```

Custom run on arbitrary result files:

```bash
python experiments_window_based/session_summary_eval/evaluate_session_summaries.py \
  --gt-file data/session_catalog_youtube_aba_100_gt_summary_gpt.json \
  --method-file focal=outputs/focal_aba.json \
  --method-file naive_llm_agent_session=outputs/naive_aba.json
```

Judge-based runs:

```bash
export OPENAI_API_KEY=your_key
python experiments_window_based/session_summary_eval/evaluate_session_summaries.py --metrics task_acc bertscore kir geval --judge-model gpt-5
```

Efficiency report:

```bash
python experiments_window_based/session_summary_eval/report_efficiency.py \
  --method-file focal=outputs/focal.json \
  --method-file naive_llm_agent_session=outputs/naive.json \
  --method-file brain_vlm_shared_memory_session=outputs/global-memory.json \
  --output outputs/efficiency.json
```

Outputs:

- output/focal_merged.json
- output/session_summary_metrics.json
- output/session_summary_metrics.md
- output/cache/
