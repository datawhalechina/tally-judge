"""Local CLI using the same registry, evaluator and run store as the Python API."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tallyjudge import Dataset, Evaluator, create_metric, create_storage_backend
from tallyjudge.core.errors import PluginError
from tallyjudge.registry import registry


def main(argv: Sequence[str] | None = None) -> int:
    """Run a local command and return its process exit status."""
    parser = argparse.ArgumentParser(prog="tallyjudge")
    commands = parser.add_subparsers(dest="command", required=True)
    metrics = commands.add_parser("metrics").add_subparsers(
        dest="action", required=True
    )
    metrics.add_parser("list")
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--metric", action="append", required=True)
    run.add_argument("--store", type=Path, required=True)
    runs = commands.add_parser("runs").add_subparsers(dest="action", required=True)
    show = runs.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--store", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "metrics":
            print(
                json.dumps(
                    [
                        {"id": s.id, "dependencies": dict(s.dependencies)}
                        for s in registry.list(kind="metric")
                    ]
                )
            )
            return 0
        store = create_storage_backend("storage.local.jsonl", root=args.store)
        if args.command == "runs":
            print(store.load_run(args.run_id).to_json())
            return 0
        try:
            evaluator = Evaluator(
                [create_metric(key) for key in args.metric], backend=store
            )
        except (PluginError, ValueError) as exc:
            parser.error(str(exc))
        with args.input.open(encoding="utf-8") as stream:
            dataset = Dataset.from_records(json.loads(line) for line in stream)
        result = evaluator.run(dataset)
        print(result.to_json(include_inputs=False))
        return 0 if result.run_status == "completed" else 1
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
