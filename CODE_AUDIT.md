# 论文—代码一致性核对

## 结论

论文主实现对应实验包中的 `datasets/supplement_focal/experiments_window_based`，而不是顶层 `src_experiments/experiments/benchmark`。本目录仅提取前者。

## 方法映射

| 论文组件 | 本目录实现 | 核对结果 |
|---|---|---|
| Filter Agent | 预处理后的 `SessionAction` 流与截图有效性检查 | 会话代码接收已清洗动作；原始捕获器不在补充材料中 |
| Brain Agent | `FocalSessionMethod._plan_session` | 一次 text-only 调用读取完整有序 app/title 元数据，输出 task routing 与 sampling plan |
| Record Agent | `_observe_action` | 仅对 `should_sample` 的动作调用 VLM，并注入当前任务记忆 |
| Memory Agent | `TaskState.memory_items` | 每个推断任务独立存储；A→B→A 可恢复先前任务状态 |
| Summary Agent | `_summarize_tasks` / `_summarize_session` | 从各任务隔离记忆生成任务级总结，再按任务首次出现顺序组合 |

## 实验映射

| 论文项目 | 代码 |
|---|---|
| FOCAL | `FocalSessionMethod` |
| Naive LLM Agent | `NaiveLLMAgentSessionBaseline` |
| FOCAL-GM | `BrainLLMVLMSharedMemorySessionBaseline` |
| DesktopBench-Multitask / Interruption | `session_loader.py` 读取 session catalog；数据需外置 |
| Task Acc / BS-F1 / KIR / G-Eval | `session_summary_eval/evaluate_session_summaries.py` |
| VCC / TCS 与 token 分解 | `session_summary_eval/report_efficiency.py`、`common.py` 与 `run.py` 的 session traces |

## 未纳入的代码

- 顶层旧 `HSRMethod`：逐事件调用 Brain，并由 evaluator 使用数据里的 `task_id` 切换记忆；这与论文“完整元数据一次规划、任务标签未知”不一致。
- `rebuttal/`、`rebuttal_results/`、`human_eval/`：属于答辩扩展与生成结果，不是论文主方法最小复现依赖。
- 硬编码表格脚本：引用特定机器的绝对结果路径，已排除。
- 数据、截图、日志、输出、虚拟环境和 IDE 配置：均与代码复现分离，避免夹带个人数据。

## 在副本中做的两项一致性修正

1. `common.py` 的 VCC 原实现使用 `VLM calls / predicted tasks`；论文 Eq. (13) 定义为 `VLM calls / sessions`，本副本已改为按会话平均。
2. 原 `eval_config.json` 是空文件，会触发 JSON 解析错误；本副本初始化为合法的 `{}`。

## 仍需注意

- Filter Agent 的原始捕获/验证代码未随论文补充材料提供；会话实验依赖已经清洗并材料化的动作流。
- 数据构造的截图与 GT summaries 未复制进来；它们体积大，且真实桌面流可能含隐私信息。
- 使用 OpenAI-compatible backend 会把输入发送到所配置的服务，不再满足论文的 on-device 隐私设定。
