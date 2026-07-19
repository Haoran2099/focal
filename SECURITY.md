# Security policy

Report vulnerabilities through GitHub private vulnerability reporting. Never
attach private DesktopBench access tokens, screenshots, model prompts, outputs,
or local paths to a public issue.

`run.py` never downloads data. `scripts/prepare_data.py` is the only supported
networked data-preparation entry point and uses the active Hugging Face login or
`HF_TOKEN`. OpenAI-compatible credentials are read from environment variables
and must not be placed in command lines or committed files.
