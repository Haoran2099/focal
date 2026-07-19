# Contributing

Use Python 3.11 and keep tests offline. Before opening a pull request, run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall -q experiments_window_based src scripts run.py
```

Do not commit DesktopBench files, screenshots, experiment outputs, credentials,
absolute user paths, virtual environments, or caches.
