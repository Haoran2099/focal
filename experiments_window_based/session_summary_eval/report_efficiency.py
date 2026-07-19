#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import compute_efficiency_metrics, load_single_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report paper-aligned VCC, TCS, and token breakdowns from FOCAL result files."
    )
    parser.add_argument(
        "--method-file",
        action="append",
        required=True,
        help="Result file in the form method_name=path/to/result.json. Repeatable.",
    )
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    return parser.parse_args()


def parse_method_file(spec: str) -> tuple[str, Path]:
    method_name, separator, raw_path = spec.partition("=")
    if not separator or not method_name.strip() or not raw_path.strip():
        raise ValueError(f"Invalid --method-file value: {spec!r}")
    return method_name.strip(), Path(raw_path).expanduser().resolve()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {"vcc_definition": "VLM calls per session", "methods": {}}

    for spec in args.method_file:
        method_name, result_path = parse_method_file(spec)
        dataset = load_single_method(method_name, result_path)
        report["methods"][method_name] = compute_efficiency_metrics(list(dataset.sessions.values()))

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
