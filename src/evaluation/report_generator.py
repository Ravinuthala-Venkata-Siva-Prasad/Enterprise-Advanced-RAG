import json
from pathlib import Path

import pandas as pd


class EvaluationReportGenerator:

    @staticmethod
    def flatten(results):

        rows = []

        for result in results:

            row = {
                "id": result["id"],
                "question": result["question"],
                "retrieval_latency_ms": result["retrieval_latency_ms"],
                "answer_latency_ms": result["answer_latency_ms"]
            }

            row.update(result["retrieval_metrics"])
            row.update(result["answer_metrics"])

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def summarize(df):

        summary = {}

        metric_columns = [
            column for column in df.columns
            if column.startswith(
                ("recall@", "precision@", "ndcg@")
            ) or column in ("mrr", "map", "exact_match", "token_f1")
        ]

        for column in metric_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if values.notna().any():
                summary[column] = float(values.mean())

        summary["average_retrieval_latency_ms"] = float(
            df["retrieval_latency_ms"].mean()
        )

        summary["average_answer_latency_ms"] = float(
            df["answer_latency_ms"].mean()
        )

        answerable = df[df["answerable"] == True]
        unanswerable = df[df["answerable"] == False]

        summary["answerable_questions"] = len(answerable)
        summary["unanswerable_questions"] = len(unanswerable)

        summary["correct_abstentions"] = int(
            unanswerable["abstention_correct"].fillna(False).sum()
        )

        summary["abstention_accuracy"] = (
            summary["correct_abstentions"] /
            len(unanswerable)
            if len(unanswerable) else None
        )

        return summary

    @classmethod
    def save(cls, results, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = cls.flatten(results)

        summary = cls.summarize(df)

        df.to_csv(
            output_dir / "evaluation_results.csv",
            index=False
        )

        with (output_dir / "evaluation_results.json").open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(results, file, indent=4, ensure_ascii=False)

        with (output_dir / "evaluation_summary.json").open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(summary, file, indent=4)

        return summary