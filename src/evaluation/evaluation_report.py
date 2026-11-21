from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


def save_evaluation_report(report: dict, output_dir="reports/evaluation"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"evaluation_{timestamp}.json"
    csv_path = output_path / f"evaluation_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    rows = []
    for result in report.get("results", []):
        row = {
            "id": result.get("id"),
            "question": result.get("question"),
            "answerable": result.get("answerable"),
            "answer": result.get("answer"),
        }
        row.update(result.get("metrics", {}))
        rows.append(row)

    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        if columns:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


class EvaluationReport:
    """Backward-compatible class interface for existing evaluation scripts."""

    def __init__(self, output_dir="reports/evaluation"):
        self.output_dir = output_dir

    def save(self, report: dict):
        return save_evaluation_report(report, self.output_dir)

    def save_report(self, report: dict):
        return self.save(report)

    def generate_report(self, report: dict):
        return self.save(report)
