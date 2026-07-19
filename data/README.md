# Prepared data directory

No benchmark data is committed to this repository. Prepare the private
DesktopBench release explicitly:

```bash
python scripts/prepare_data.py --data-dir data/desktopbench --revision v0.1.0
```

`run.py` reads `DESKTOPBENCH_ROOT` when set and otherwise uses
`data/desktopbench`. Explicit `--session-catalog` and `--session-raw-records`
arguments override both locations.
