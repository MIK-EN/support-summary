import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from summary.llm import LLMClient, OpenRouterClient

# Keep the tests-only imports grouped before the CLI and runner modules exist.
# isort: split
from summary.__main__ import main
from summary.runner import Config, RunOutput, load_config, make_client, run_once

from .test_cases import GOLDEN

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIRST_RUN = "2026-11-23T10:00"
MODEL = "deepseek/deepseek-v4.1-flash-1017"
CONFIG_YAML = """\
interval_minutes: {interval}
llm_model: {model}
paths:
  cases: data/cases.json
  llm_cache: data/llm_cache.json
  rules: rules.yaml
  state: state/state.json
  out: out
  logs: logs
"""


class FakeClient:
    model = "fake-model"

    def __init__(self, response: str | None = None) -> None:
        self.response = response
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.response is None:
            raise AssertionError("The LLM must not be called during refresh")
        return self.response


def write_config(path: Path, interval: int = 60) -> None:
    path.write_text(CONFIG_YAML.format(interval=interval, model=MODEL), encoding="utf-8")


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def fail_on_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail("The CLI tests must not call the network")

    monkeypatch.setattr("urllib.request.urlopen", fail_on_network)
    (tmp_path / "data").mkdir()
    for name in ("data/cases.json", "data/llm_cache.json", "rules.yaml"):
        shutil.copyfile(PROJECT_DIR / name, tmp_path / name)
    shutil.copytree(PROJECT_DIR / "prompts", tmp_path / "prompts")

    path = tmp_path / "config.yaml"
    write_config(path)
    return path


def run_args(config_path: Path, now: str = FIRST_RUN) -> list[str]:
    return ["run", "--once", "--now", now, "--config", str(config_path)]


def add_courier_message(root: Path, text: str, deadline: str | None = None) -> None:
    path = root / "data/cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    case = next(case for case in cases if case["case_id"] == "П-08")
    if deadline is not None:
        case["deadline"] = deadline
    case["chat_messages"].append(
        {
            "msg_id": "p08-2",
            "author_role": "courier",
            "sent_at": "2026-11-23T10:03",
            "text": text,
        }
    )
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")


def read_logs(root: Path) -> list[dict[str, Any]]:
    log = root / "logs/2026-11-23.jsonl"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_first_run_prints_and_saves_golden_summary(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = config_path.parent
    cache = root / "data/llm_cache.json"
    original_cache = cache.read_bytes()

    assert main(run_args(config_path)) == 0

    assert capsys.readouterr().out == GOLDEN + "\n"
    assert (root / "out/summary_20261123_1000.md").read_text(encoding="utf-8") == GOLDEN
    state = json.loads((root / "state/state.json").read_text(encoding="utf-8"))
    assert state["last_full_date"] == "2026-11-23"
    assert cache.read_bytes() == original_cache


def test_identical_second_run_is_silent(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = run_args(config_path)
    assert main(args) == 0
    capsys.readouterr()

    assert main(args) == 0

    assert capsys.readouterr().out == ""
    assert list((config_path.parent / "out").glob("changes_*")) == []


def test_same_day_run_reports_a_changed_deadline_and_new_message(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(run_args(config_path)) == 0
    capsys.readouterr()
    root = config_path.parent
    add_courier_message(root, "Теперь буду к 14:30", deadline="2026-11-23T13:00")

    assert main(run_args(config_path, "2026-11-23T11:00")) == 0

    output = capsys.readouterr().out
    assert output.startswith("# Изменения · 23.11.2026 11:00")
    assert "**П-08**" in output
    assert "новое сообщение (курьер)" in output
    assert "срок 10:00 → 13:00 (1-е изменение)" in output
    assert "## Требуют внимания" in output
    assert "⚠️ Разбор неоднозначен, прочитать оригинал: «Теперь буду к 14:30»" in output
    assert (root / "out/changes_20261123_1100.md").is_file()


def test_next_day_run_prints_a_full_summary(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(run_args(config_path)) == 0
    capsys.readouterr()

    assert main(run_args(config_path, "2026-11-24T09:00")) == 0

    assert capsys.readouterr().out.startswith("# Сводка support · 24.11.2026 09:00")


def test_night_run_waits_for_the_morning_summary(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(run_args(config_path)) == 0
    capsys.readouterr()

    assert main(run_args(config_path, "2026-11-24T00:30")) == 0
    assert capsys.readouterr().out == ""

    assert main(run_args(config_path, "2026-11-24T09:00")) == 0
    assert capsys.readouterr().out.startswith("# Сводка support · 24.11.2026 09:00")


def test_refresh_does_not_hide_changes_from_the_next_run(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(run_args(config_path)) == 0
    add_courier_message(config_path.parent, "Теперь буду к 14:30")
    assert main(["summary", "--now", "2026-11-23T10:05", "--config", str(config_path)]) == 0
    capsys.readouterr()

    assert main(run_args(config_path, "2026-11-23T11:00")) == 0

    assert "новое сообщение (курьер)" in capsys.readouterr().out


def test_refresh_prints_a_full_summary_without_calling_the_llm(
    config_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(run_args(config_path)) == 0
    capsys.readouterr()
    add_courier_message(config_path.parent, "Теперь буду к 14:30")
    client = FakeClient()
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("summary.runner.make_client", lambda cfg: client)

    assert main(["summary", "--now", "2026-11-23T10:05", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert output.startswith("# Сводка support · 23.11.2026 10:05")
    assert "⚠️ Разбор неоднозначен, прочитать оригинал: «Теперь буду к 14:30»" in output
    assert client.calls == 0


def test_run_caches_and_logs_a_successful_llm_parse(
    config_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = config_path.parent
    text = "Теперь буду в 14:30"
    add_courier_message(root, text)
    raw = json.dumps(
        {
            "summary": "Курьер будет в 14:30",
            "quote": text,
            "times": [{"time": "14:30", "role": "new_eta"}],
            "labels": ["reports_problem"],
            "uncertain": False,
        },
        ensure_ascii=False,
    )
    client = FakeClient(raw)
    monkeypatch.setattr("summary.runner.make_client", lambda cfg: client)

    assert main(run_args(config_path, "2026-11-23T11:00")) == 0

    assert capsys.readouterr().out.startswith("# Сводка support · 23.11.2026 11:00")
    cache = json.loads((root / "data/llm_cache.json").read_text(encoding="utf-8"))
    assert cache["p08-2"]["extraction"] == json.loads(raw)
    events = read_logs(root)
    [llm_event] = [event for event in events if event["event"] == "llm"]
    assert llm_event["msg_id"] == "p08-2"
    assert llm_event["raw"] == raw
    assert llm_event["text"] == text
    [case_event] = [
        event for event in events if event["event"] == "case" and event["case_id"] == "П-08"
    ]
    assert case_event["model"] == client.model


def test_logs_report_fired_rules_and_flag_r0_for_review(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = config_path.parent
    assert main(run_args(config_path)) == 0
    assert capsys.readouterr().out == GOLDEN + "\n"

    events = read_logs(root)
    case_events = [event for event in events if event["event"] == "case"]
    assert len(case_events) == 4
    assert {event["case_id"]: event["fired"] for event in case_events} == {
        "С-17": ["R5", "C2"],
        "С-24": ["R4", "C2"],
        "П-08": ["R1", "C1", "C3"],
        "П-09": ["R3"],
    }
    assert all(event["review"] is False for event in case_events)
    [run_event] = [event for event in events if event["event"] == "run"]
    assert run_event["mode"] == "full"
    assert run_event["cases"] == 4
    assert run_event["out"] == str(root / "out/summary_20261123_1000.md")
    assert all(event["ts"] == "2026-11-23T10:00:00" for event in events)

    path = root / "data/cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    cases.append(
        {
            "case_id": "П-10",
            "type": "card_delivery",
            "crm_status": "pending",
            "crm_updated_at": FIRST_RUN,
        }
    )
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    run_once(datetime(2026, 11, 23, 11), load_config(config_path), None)

    updated_events = read_logs(root)
    assert updated_events[: len(events)] == events
    [r0_event] = [
        event
        for event in updated_events
        if event["event"] == "case" and event["case_id"] == "П-10"
    ]
    assert r0_event["fired"] == ["R0"]
    assert r0_event["review"] is True


@pytest.mark.parametrize("interval", [15, 30, 60, 120])
def test_load_config_accepts_allowed_intervals_and_resolves_paths(
    config_path: Path, interval: int
) -> None:
    write_config(config_path, interval)

    config = load_config(config_path)

    assert config.interval_minutes == interval
    root = config_path.parent
    assert config.paths.model_dump() == {
        "cases": root / "data/cases.json",
        "llm_cache": root / "data/llm_cache.json",
        "rules": root / "rules.yaml",
        "state": root / "state/state.json",
        "out": root / "out",
        "logs": root / "logs",
    }


def test_load_config_rejects_interval_45(config_path: Path) -> None:
    write_config(config_path, interval=45)

    with pytest.raises(ValueError):
        load_config(config_path)


def test_run_rejects_now_without_once(config_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--now", FIRST_RUN, "--config", str(config_path)])

    assert error.value.code == 2


def test_loop_sleeps_once_between_two_runs(
    config_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str | float] = []

    def record_run(
        now: datetime, cfg: Config, client: LLMClient | None, refresh: bool = False
    ) -> RunOutput:
        events.append("run")
        return RunOutput(mode="silent", markdown="", out_path=None)

    def fake_sleep(seconds: float) -> None:
        events.append(seconds)

    monkeypatch.setattr("summary.__main__.run_once", record_run)

    assert main(["run", "--config", str(config_path)], sleep=fake_sleep, max_iterations=2) == 0

    assert events == ["run", 3600, "run"]
    assert capsys.readouterr().out == ""


def test_make_client_uses_configured_model_and_env_key(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(config_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    client = make_client(cfg)

    assert isinstance(client, OpenRouterClient)
    assert client.model == MODEL
    assert client.api_key == "k"


@pytest.mark.parametrize("key", [None, ""])
def test_make_client_returns_none_without_nonempty_key(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
    if key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)

    assert make_client(load_config(config_path)) is None
