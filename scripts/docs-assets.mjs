import { copyFile, mkdir } from "node:fs/promises";

const source = new URL("../examples/", import.meta.url);
const destination = new URL("../docs/public/examples/", import.meta.url);

await mkdir(destination, { recursive: true });
for (const name of [
  "rag_openai_judge.py",
  "rag_retrieval_ids.py",
  "rag_generate_testset.py",
  "rag_answer_then_score.py",
]) {
  await copyFile(new URL(name, source), new URL(name, destination));
}
