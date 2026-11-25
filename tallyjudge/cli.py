import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tallyjudge import (
    Dataset,
    Evaluator,
    create_analyzer,
    create_metric,
    create_storage_backend,
)
from tallyjudge.api.suite import Suite
from tallyjudge.registry import registry
from tallyjudge.sdk.errors import PluginError
from tallyjudge.sdk.plugin import manifest_source


def _option(text: str) -> tuple[str, Any]:
    key, value = text.split("=", 1)
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tallyjudge")
    commands = parser.add_subparsers(dest="command", required=True)
    metrics = commands.add_parser("metrics").add_subparsers(
        dest="action", required=True
    )
    metrics.add_parser("list")
    manifest = commands.add_parser("manifest")
    manifest.add_argument("package")
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--metric", action="append", required=True)
    run.add_argument("--store", type=Path, required=True)
    runs = commands.add_parser("runs").add_subparsers(dest="action", required=True)
    show = runs.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--store", type=Path, required=True)
    analyze = runs.add_parser("analyze")
    analyze.add_argument("run_id", nargs="+")
    analyze.add_argument("--store", type=Path, required=True)
    analyze.add_argument("--analyzer", required=True)
    analyze.add_argument("--option", action="append", default=[])
    suite = commands.add_parser("suite").add_subparsers(dest="action", required=True)
    suite_run = suite.add_parser("run")
    suite_run.add_argument("--store", type=Path, required=True)
    for command in (suite_run, suite.add_parser("show")):
        command.add_argument("file", type=Path)
        command.add_argument("--overlay", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "metrics":
            print(
                json.dumps(
                    [
                        {
                            "id": s.id,
                            "dependencies": dict(s.dependencies),
                            **(
                                {"optional_dependencies": dict(s.optional_dependencies)}
                                if s.optional_dependencies
                                else {}
                            ),
                        }
                        for s in registry.list(kind="metric")
                    ]
                )
            )
            return 0
        if args.command == "manifest":
            print(manifest_source(args.package), end="")
            return 0
        if args.command == "suite" and args.action == "show":
            print(
                json.dumps(
                    Suite.load(args.file, *args.overlay).effective_config(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        store = create_storage_backend("storage.local.jsonl", root=args.store)
        if args.command == "suite":
            result = Suite.load(args.file, *args.overlay).run(backend=store)
            print(result.to_json(include_inputs=False))
            return 0 if result.run_status == "completed" else 1
        if args.command == "runs" and args.action == "analyze":
            options = dict(_option(option) for option in args.option)
            try:
                analyzer = create_analyzer(args.analyzer, **options)
            except PluginError as exc:
                parser.error(str(exc))
            report = analyzer.analyze([store.load_run(run) for run in args.run_id])
            print(json.dumps(report, ensure_ascii=False))
            return 0
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
