"""Generate up to two reference questions from knowledge-base passages.

Run from the repository root:
    uv run --extra openai examples/rag_generate_testset.py
"""

import asyncio
import os

from openai import AsyncOpenAI

from tallyjudge import create_language_model, create_synthesizer

# Edit these example values or supply them in your shell.
os.environ.setdefault("OPENAI_API_KEY", "your-api-key")
os.environ.setdefault("JUDGE_MODEL", "gpt-4o-mini")
# For an OpenAI-compatible endpoint, uncomment and edit the URL.
# os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")


async def main() -> None:
    chunks = [
        {
            "document_id": "returns-policy",
            "chunk_id": "returns-window",
            "text": "顾客可在签收后七天内申请退货，商品和包装须保持完好。",
        },
        {
            "document_id": "refund-guide",
            "chunk_id": "refund-timing",
            "text": "仓库签收退货后，退款将在三个工作日内退回原支付账户。",
        },
    ]
    async with AsyncOpenAI() as client:
        llm = create_language_model(
            "model.openai_compatible.chat_completions",
            client=client,
            model=os.environ["JUDGE_MODEL"],
            temperature=0.0,
        )
        generator = create_synthesizer("synthesis.testset.single_hop", llm=llm)
        result = await generator.arun(chunks, testset_size=2)

    print(f"生成成功 {result.generated_count}/{result.attempted_count} 题")
    for row in result.to_records():
        print(f"\n问题：{row['user_input']}")
        print(f"参考答案：{row['reference']}")
        print(
            f"参考块 [{row['reference_context_ids'][0]}] {row['reference_contexts'][0]}"
        )
    for candidate in result.candidates:
        if error := candidate.error:
            reason = error.code or error.exception_type or error.category
            print(f"{candidate.chunk_id} 生成失败：{reason}")


if __name__ == "__main__":
    asyncio.run(main())
