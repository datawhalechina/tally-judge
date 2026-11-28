"""Score retrieved chunk IDs against labeled evidence IDs locally.

Run from the repository root:
    uv run examples/rag_retrieval_ids.py
"""

from tallyjudge import Dataset, Evaluator, create_metric


def main() -> None:
    dataset = Dataset.from_records(
        [
            {
                "case_id": "return-window",
                "user_input": "无理由退货的申请期限是多久？",
                "retrieved_context_ids": ["returns/window", "support/hours"],
                "reference_context_ids": ["returns/window"],
            },
            {
                "case_id": "return-shipping",
                "user_input": "申请无理由退货的期限和运费规则是什么？",
                "retrieved_context_ids": ["returns/window"],
                "reference_context_ids": ["returns/window", "returns/shipping"],
            },
        ]
    )
    metrics = [
        create_metric("metric.retrieval.id_context_precision"),
        create_metric("metric.retrieval.id_context_recall"),
    ]
    result = Evaluator(metrics).run(dataset)
    for row in result.rows:
        print(row.case_id)
        for name, score in row.metrics.items():
            print(" ", name, score.value)


if __name__ == "__main__":
    main()
