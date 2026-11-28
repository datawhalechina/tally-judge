"""Evaluate one RAG response and its retrieved passages with an OpenAI judge.

Run from the repository root:
    uv run --extra openai examples/rag_openai_judge.py
"""

import asyncio
import os

from openai import AsyncOpenAI

from tallyjudge import Dataset, Evaluator, create_language_model, create_metric

# Edit these example values or supply them in your shell.
os.environ.setdefault("OPENAI_API_KEY", "your-api-key")
os.environ.setdefault("JUDGE_MODEL", "gpt-4o-mini")
# For an OpenAI-compatible endpoint, uncomment and edit the URL.
# os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")


async def main() -> None:
    dataset = Dataset.from_records(
        [
            {
                "case_id": "return-shipping",
                "user_input": "签收后多久可以申请无理由退货？寄回运费由谁承担？",
                "reference": "签收后 7 天内可以申请无理由退货，寄回运费由买家承担。",
                # The system under test got the shipping policy wrong.
                "response": "签收后 7 天内可以申请无理由退货，寄回运费由商家承担。",
                "retrieved_contexts": [
                    "在线客服的工作时间为每天 9:00 至 18:00。",
                    "无理由退货申请须在签收后 7 天内提交。",
                    "无理由退货的寄回运费由买家承担。",
                ],
            }
        ]
    )
    async with AsyncOpenAI() as client:
        judge = create_language_model(
            "model.openai_compatible.chat_completions",
            client=client,
            model=os.environ["JUDGE_MODEL"],
            temperature=0.0,
        )
        metrics = [
            create_metric("metric.rag.faithfulness", llm=judge),
            create_metric("metric.rag.context_precision_with_reference", llm=judge),
            create_metric("metric.rag.context_recall", llm=judge),
        ]
        result = await Evaluator(metrics).arun(dataset)

    for name, score in result.row_for("return-shipping").metrics.items():
        print(name, score.status, score.value if score.error is None else score.error)


if __name__ == "__main__":
    asyncio.run(main())
