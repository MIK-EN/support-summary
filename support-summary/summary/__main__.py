import argparse
import time
from collections.abc import Callable
from datetime import datetime
from itertools import count, islice
from pathlib import Path

from summary import runner
from summary.runner import run_once


def main(
    argv: list[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_iterations: int | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--once", action="store_true")
    refresh = commands.add_parser("summary", help="Актуализировать")
    refresh.set_defaults(once=True)
    default_config = Path(__file__).resolve().parents[1] / "config.yaml"
    for command in (run, refresh):
        command.add_argument("--now", type=datetime.fromisoformat)
        command.add_argument("--config", type=Path, default=default_config)
    args = parser.parse_args(argv)
    if args.now is not None and not args.once:
        parser.error("--now requires --once with run")

    cfg = runner.load_config(args.config)
    client = runner.make_client(cfg)
    limit = 1 if args.once else max_iterations
    for iteration in islice(count(), limit):
        if iteration:
            sleep(cfg.interval_minutes * 60)
        now = args.now or datetime.now().replace(second=0, microsecond=0)
        result = run_once(now, cfg, client, refresh=args.command == "summary")
        if result.markdown:
            print(result.markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
