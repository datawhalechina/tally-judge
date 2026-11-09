# TallyJudge Examples

Examples demonstrating how to use TallyJudge for evaluating different types of AI applications including RAG systems, agents, prompts, workflows, and LLM benchmarking. These examples might be unstable and are subject to change.

## Installation

### Local Development
Install both main ragas and examples packages in editable mode:

```bash
cd /path/to/ragas
uv pip install -e . -e ./examples
```

Or using regular pip:
```bash
cd /path/to/ragas  
pip install -e . -e ./examples
```

## Available Examples

- **`ragas_examples.agent_evals`** - Agent evaluation examples
- **`ragas_examples.benchmark_llm`** - LLM benchmarking and comparison examples  
- **`ragas_examples.prompt_evals`** - Prompt evaluation examples
- **`ragas_examples.rag_eval`** - RAG system evaluation examples
- **`ragas_examples.workflow_eval`** - Workflow evaluation examples

## Usage

### Set Environment Variables

Most examples require API keys to be set:

```bash
export OPENAI_API_KEY=your_key_here
```

For Google Drive examples, also install the gdrive extra:
```bash
pip install -e ".[gdrive]" -e ./examples
```

### Running Examples as Modules

After installation, you can run examples directly:

```bash
# Run benchmark LLM prompt example
python -m ragas_examples.benchmark_llm.prompt

# Run benchmark LLM evaluation
python -m ragas_examples.benchmark_llm.evals

# Run other examples
python -m ragas_examples.rag_eval.evals
python -m ragas_examples.agent_evals.evals
python -m ragas_examples.prompt_evals.evals
python -m ragas_examples.workflow_eval.evals
```

## Release process

Publish a GitHub Release from a version tag such as `vX.Y.Z`. The release workflow builds `ragas` and `ragas-examples` from that tag and attaches a wheel and source archive for each package to the release.

## Local Development & Testing

### Verify Installation
```bash

# Test module execution
python -m ragas_examples.benchmark_llm.prompt --help
```
