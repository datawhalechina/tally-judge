"""Generate a RAG answer from retrieved passages, then judge its accuracy.

Run from the repository root:
    uv run --extra openai examples/rag_answer_then_score.py
"""

import asyncio
import os

from openai import AsyncOpenAI

from tallyjudge import Dataset, create_language_model, create_metric, experiment

# Edit these example values or supply them in your shell.
os.environ.setdefault("OPENAI_API_KEY", "your-api-key")
os.environ.setdefault("JUDGE_MODEL", "gpt-4o-mini")
# For an OpenAI-compatible endpoint, uncomment and edit the URL.
# os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")


async def main() -> None:
    dataset = Dataset.from_records(
        [
            {
                "case_id": "refund",
                "user_input": "退款审核通过后，钱会退到哪里？大约多久能收到？",
                "retrieved_contexts": [
                    "退款将原路退回付款账户，审核通过后通常在 3 至 5 个工作日内到账。",
                    "发票可在订单详情页下载。",
                ],
                "reference": "款项退还至原支付账户，一般要等 3 到 5 个工作日。",
            }
        ]
    )
    model = os.environ["JUDGE_MODEL"]
    async with AsyncOpenAI() as client:

        @experiment
        async def answer(row):
            passages = "\n".join(row["retrieved_contexts"])
            completion = await client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "请依据提供的资料回答；资料不足时请说明。",
                    },
                    {
                        "role": "user",
                        "content": f"资料：\n{passages}\n\n问题：{row['user_input']}",
                    },
                ],
            )
            return {"response": completion.choices[0].message.content}

        judge = create_language_model(
            "model.openai_compatible.chat_completions",
            client=client,
            model=model,
            temperature=0.0,
        )
        accuracy = create_metric("metric.rag.answer_accuracy", llm=judge)
        result = await answer.arun(dataset, metrics=[accuracy])

    row = result.row_for("refund")
    assert row.task is not None
    if row.task.output is None:
        print("Answer generation:", row.task.status, row.task.error)
        return
    print("Generated answer:", row.task.output["response"])
    score = row.metrics[accuracy.spec.id]
    print(
        "Answer accuracy:",
        score.status,
        score.value if score.error is None else score.error,
    )


if __name__ == "__main__":
    asyncio.run(main())
