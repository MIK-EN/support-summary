import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from summary.adapters import load_cases
from summary.llm import LLMClient, OpenRouterClient, load_cache, parse_message, save_cache
from summary.models import ParsedMessage
from summary.render import render_full
from summary.rules import CaseResult, evaluate_all, load_rules
from summary.state import State, diff, is_full_run, load_state, render_changes, save_state, snapshot


class Paths(BaseModel):
    cases: Path
    llm_cache: Path
    rules: Path
    state: Path
    out: Path
    logs: Path


class Config(BaseModel):
    interval_minutes: int
    llm_model: str
    paths: Paths


def load_config(path: str | Path) -> Config:
    path = Path(path)
    cfg = Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if cfg.interval_minutes not in (15, 30, 60, 120):
        raise ValueError("interval_minutes must be 15, 30, 60, or 120")
    root = path.resolve().parent
    cfg.paths = Paths(**{name: (root / value).resolve() for name, value in cfg.paths})
    return cfg


def make_client(cfg: Config) -> LLMClient | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    return OpenRouterClient(key, cfg.llm_model) if key else None


class RunOutput(BaseModel):
    mode: Literal["full", "changes", "silent"]
    markdown: str
    out_path: Path | None


def _log_run(
    directory: Path,
    now: datetime,
    results: list[CaseResult],
    model: str,
    llm_events: list[dict],
    message_texts: dict[str, str],
    output: RunOutput,
) -> None:
    ts = now.isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{now:%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as log:
        for entry in llm_events:
            event = {"ts": ts, "event": "llm", **entry, "text": message_texts[entry["msg_id"]]}
            log.write(json.dumps(event, ensure_ascii=False) + "\n")

        for result in results:
            event = {
                "ts": ts,
                "event": "case",
                "case_id": result.case.case_id,
                "priority": result.priority,
                "fired": result.fired,
                "overdue": result.overdue,
                "model": model,
                "review": result.base_rule == "R0" or "C5" in result.fired,
            }
            log.write(json.dumps(event, ensure_ascii=False) + "\n")

        event = {
            "ts": ts,
            "event": "run",
            "mode": output.mode,
            "cases": len(results),
            "out": str(output.out_path) if output.out_path is not None else None,
        }
        log.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_once(
    now: datetime, cfg: Config, client: LLMClient | None, refresh: bool = False
) -> RunOutput:
    paths = cfg.paths
    cases = load_cases(paths.cases)
    cache = load_cache(paths.llm_cache)
    rules = load_rules(paths.rules)

    parse_client = None if refresh else client
    cache_size = len(cache)
    parsed: dict[str, ParsedMessage] = {}
    llm_events: list[dict] = []
    message_texts: dict[str, str] = {}
    for case in (case for case in cases if case.is_open):
        for message in case.chat_messages:
            parsed[message.msg_id] = parse_message(
                message, case.type, parse_client, cache, llm_events
            )
            message_texts[message.msg_id] = message.text
    if len(cache) > cache_size:
        save_cache(paths.llm_cache, cache)

    results = evaluate_all(cases, parsed, now, rules)
    prev = load_state(paths.state)
    new_snaps = snapshot(results, prev)
    out_path = None
    last_full_date = prev.last_full_date

    mode: Literal["full", "changes", "silent"]
    if refresh or is_full_run(prev, now):
        mode = "full"
        markdown = render_full(results, now)
        out_path = paths.out / f"summary_{now:%Y%m%d_%H%M}.md"
        last_full_date = now.date()
    elif changes := diff(prev, results):
        mode = "changes"
        markdown = render_changes(changes, results, new_snaps, now)
        out_path = paths.out / f"changes_{now:%Y%m%d_%H%M}.md"
    else:
        mode = "silent"
        markdown = ""

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    save_state(paths.state, State(last_run=now, last_full_date=last_full_date, cases=new_snaps))

    output = RunOutput(mode=mode, markdown=markdown, out_path=out_path)
    model = parse_client.model if parse_client is not None else "cache"
    _log_run(paths.logs, now, results, model, llm_events, message_texts, output)
    return output
