# Reproducibility guide

## Reference setup

- Python: 3.11
- Inference backend: Ollama
- Brain model: `qwen3:8b`
- VLM: `qwen3-vl:8b`
- Embedding model: `qwen3-embedding:8b`
- DesktopBench release: `HaoranYin/desktopbench@v0.1.0`

The private dataset release pins `VideoGUI-Mid-Plan` and `VideoGUI-Action` by
full commit SHA. The preparation script verifies 2,572 reconstructed actions,
320 multitask sessions, 100 interruption sessions, and both GT sets before an
experiment starts.

## Paper mapping

| Paper component | Repository entry point |
|---|---|
| FOCAL | `python run.py --method focal` |
| Naive baseline | `python run.py --method naive_llm_agent_session` |
| FOCAL-GM/shared memory | `python run.py --method brain_vlm_shared_memory_session` |
| Task Acc, BS-F1, KIR, G-Eval | `evaluate_session_summaries.py` |
| VCC and TCS | `report_efficiency.py` |

The paper reports aggregate improvements, including 60.4% lower total token
consumption and 72.3% fewer VLM calls relative to its baseline. Exact stochastic
outputs can vary with model builds and hardware; compare aggregate metrics and
retain the emitted per-session traces, model names, dataset revision, and token
usage fields.

## Output contract

Each run writes a JSON object containing method/backend/model identifiers,
portable input paths, completed session count, per-session results, per-session
LLM/VLM traces, and aggregate token usage. `--save-every` controls checkpoints;
`--resume-from` resumes by session ID without rerunning completed sessions.

## Failure modes

- Missing data produces an actionable error and never triggers an implicit
  download.
- Missing Ollama models fail at the model call; pull all three reference models
  before a full run.
- KIR/G-Eval fail closed when the configured judge-key environment variable is
  absent.
- Do not compare runs that use different dataset tags, upstream revisions, or
  model versions without recording that difference.
