from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from summary.adapters import load_cases
from summary.llm import load_cache
from summary.models import Case, ChatMessage, Extraction, ParsedMessage
from summary.rules import CaseResult, RulesConfig, evaluate_all, load_rules

# Keep import ordering stable while the implementation modules are absent.
# isort: split
from summary.render import render_item
from summary.state import (
    CaseSnapshot,
    Change,
    State,
    diff,
    is_full_run,
    load_state,
    render_changes,
    save_state,
    snapshot,
)

PROJECT_DIR = Path(__file__).parent.parent
NOW = datetime(2026, 11, 23, 10)


@pytest.fixture
def cases() -> dict[str, Case]:
    return {case.case_id: case for case in load_cases(PROJECT_DIR / "data" / "cases.json")}


@pytest.fixture
def parsed() -> dict[str, ParsedMessage]:
    return load_cache(PROJECT_DIR / "data" / "llm_cache.json")


@pytest.fixture
def rules() -> RulesConfig:
    return load_rules(PROJECT_DIR / "rules.yaml")


@pytest.fixture
def results(
    cases: dict[str, Case], parsed: dict[str, ParsedMessage], rules: RulesConfig
) -> list[CaseResult]:
    return evaluate_all(list(cases.values()), parsed, NOW, rules)


@pytest.fixture
def prev(results: list[CaseResult]) -> State:
    return State(cases=snapshot(results, State()))


@pytest.fixture
def courier_message() -> ChatMessage:
    return ChatMessage(msg_id="p08-2", author_role="courier", sent_at=NOW, text="Маршрут уточнил")


@pytest.fixture
def onboarding_problem(cases: dict[str, Case]) -> tuple[Case, ParsedMessage]:
    text = "Не могу загрузить документы"
    message = ChatMessage(msg_id="c17-2", author_role="employee", sent_at=NOW, text=text)
    case = cases["С-17"]
    updated = case.model_copy(update={"chat_messages": [*case.chat_messages, message]})
    parse = ParsedMessage(
        msg_id=message.msg_id,
        extraction=Extraction(summary=text, quote=text, labels=["reports_problem"]),
    )
    return updated, parse


def evaluate_updated(
    cases: dict[str, Case],
    parsed: dict[str, ParsedMessage],
    rules: RulesConfig,
    updated: Case,
) -> list[CaseResult]:
    return evaluate_all(list((cases | {updated.case_id: updated}).values()), parsed, NOW, rules)


def test_load_state_missing_file(tmp_path: Path) -> None:
    state = load_state(tmp_path / "missing.json")

    assert state == State()
    assert (state.last_run, state.last_full_date, state.cases) == (None, None, {})


def test_save_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"
    state = State(
        last_run=NOW,
        last_full_date=NOW.date(),
        cases={
            "П-08": CaseSnapshot(
                priority="red",
                crm_status="in_transit",
                deadline=NOW,
                deadline_history=["10:00"],
                msg_ids=["p08-1"],
            )
        },
    )

    save_state(path, state)

    assert load_state(path) == state
    raw = path.read_text(encoding="utf-8")
    assert '"П-08"' in raw
    assert '\n  "last_run":' in raw


@pytest.mark.parametrize(
    ("last_full_date", "expected"),
    [(None, True), (NOW.date(), False), (NOW.date() - timedelta(days=1), True)],
)
def test_is_full_run(last_full_date: date | None, expected: bool) -> None:
    assert is_full_run(State(last_full_date=last_full_date), NOW) is expected


def test_snapshot_from_empty_state(results: list[CaseResult]) -> None:
    current = snapshot(results, State())

    assert current.keys() == {"С-17", "С-24", "П-08", "П-09"}
    assert current["П-08"] == CaseSnapshot(
        priority="red",
        crm_status="in_transit",
        deadline=NOW,
        deadline_history=["10:00"],
        msg_ids=["p08-1"],
    )


def test_snapshot_appends_only_changed_deadline(
    cases: dict[str, Case], parsed: dict[str, ParsedMessage], rules: RulesConfig, prev: State
) -> None:
    case = cases["П-08"].model_copy(update={"deadline": NOW.replace(hour=13)})
    results_at_13 = evaluate_updated(cases, parsed, rules, case)
    at_13 = State(cases=snapshot(results_at_13, prev))

    assert at_13.cases["П-08"].deadline_history == ["10:00", "13:00"]
    assert snapshot(results_at_13, at_13)["П-08"].deadline_history == ["10:00", "13:00"]

    case_at_1430 = case.model_copy(update={"deadline": NOW.replace(hour=14, minute=30)})
    results_at_1430 = evaluate_updated(cases, parsed, rules, case_at_1430)

    assert snapshot(results_at_1430, at_13)["П-08"].deadline_history == ["10:00", "13:00", "14:30"]


def test_diff_new_cases_follow_results_order(results: list[CaseResult]) -> None:
    assert diff(State(), results) == [
        Change(case_id=result.case.case_id, kind="new_case", detail="новая заявка")
        for result in results
    ]


def test_diff_does_not_repeat_changes(prev: State, results: list[CaseResult]) -> None:
    assert diff(prev, results) == []


def test_diff_closed_case(
    cases: dict[str, Case], parsed: dict[str, ParsedMessage], rules: RulesConfig, prev: State
) -> None:
    closed = cases["П-09"].model_copy(update={"crm_status": "done"})
    results = evaluate_updated(cases, parsed, rules, closed)

    assert diff(prev, results) == [Change(case_id="П-09", kind="closed", detail="закрыта по CRM")]


def test_diff_priority_increase(
    cases: dict[str, Case],
    parsed: dict[str, ParsedMessage],
    rules: RulesConfig,
    prev: State,
    onboarding_problem: tuple[Case, ParsedMessage],
) -> None:
    case, parse = onboarding_problem
    results = evaluate_updated(cases, parsed | {parse.msg_id: parse}, rules, case)

    assert diff(prev, results) == [
        Change(case_id="С-17", kind="priority_up", detail="приоритет 🟡 → 🔴"),
        Change(case_id="С-17", kind="new_message", detail="новое сообщение (сотрудник)"),
    ]


def test_diff_priority_decrease_reports_status_change(
    cases: dict[str, Case], parsed: dict[str, ParsedMessage], rules: RulesConfig, prev: State
) -> None:
    accepted = cases["П-08"].model_copy(update={"crm_status": "accepted", "chat_messages": []})
    results = evaluate_updated(cases, parsed, rules, accepted)

    assert next(result for result in results if result.case.case_id == "П-08").priority == "green"
    assert diff(prev, results) == [
        Change(case_id="П-08", kind="status_changed", detail="статус «В пути» → «Принято»")
    ]


def test_diff_new_courier_message(
    cases: dict[str, Case],
    parsed: dict[str, ParsedMessage],
    rules: RulesConfig,
    prev: State,
    courier_message: ChatMessage,
) -> None:
    case = cases["П-08"]
    updated = case.model_copy(update={"chat_messages": [*case.chat_messages, courier_message]})
    results = evaluate_updated(cases, parsed, rules, updated)

    assert diff(prev, results) == [
        Change(case_id="П-08", kind="new_message", detail="новое сообщение (курьер)")
    ]


def test_diff_deadline_change(
    cases: dict[str, Case], parsed: dict[str, ParsedMessage], rules: RulesConfig, prev: State
) -> None:
    updated = cases["П-08"].model_copy(update={"deadline": NOW.replace(hour=13)})
    results = evaluate_updated(cases, parsed, rules, updated)

    assert diff(prev, results) == [Change(case_id="П-08", kind="deadline_changed", detail="")]


def test_diff_orders_multiple_changes_for_one_case(
    cases: dict[str, Case],
    parsed: dict[str, ParsedMessage],
    rules: RulesConfig,
    prev: State,
    onboarding_problem: tuple[Case, ParsedMessage],
) -> None:
    case, parse = onboarding_problem
    updated = case.model_copy(
        update={"crm_status": "accepted", "deadline": NOW + timedelta(days=1)}
    )
    results = evaluate_updated(cases, parsed | {parse.msg_id: parse}, rules, updated)

    assert [(change.case_id, change.kind) for change in diff(prev, results)] == [
        ("С-17", "priority_up"),
        ("С-17", "status_changed"),
        ("С-17", "deadline_changed"),
        ("С-17", "new_message"),
    ]


def test_render_changes_is_silent_for_empty_changes(results: list[CaseResult], prev: State) -> None:
    assert render_changes([], results, prev.cases, NOW) == ""


def test_render_changes_groups_details_and_includes_attention_item(
    cases: dict[str, Case],
    parsed: dict[str, ParsedMessage],
    rules: RulesConfig,
    courier_message: ChatMessage,
) -> None:
    case = cases["П-08"]
    updated = case.model_copy(
        update={
            "deadline": NOW.replace(hour=14, minute=30),
            "chat_messages": [*case.chat_messages, courier_message],
        }
    )
    results = evaluate_updated(cases, parsed, rules, updated)
    delivery = next(result for result in results if result.case.case_id == "П-08")
    assert delivery.priority in ("red", "yellow")
    snapshots = snapshot(results, State())
    snapshots["П-08"] = snapshots["П-08"].model_copy(
        update={"deadline_history": ["10:00", "13:00", "14:30"]}
    )
    changes = [
        Change(case_id="П-08", kind="deadline_changed", detail=""),
        Change(case_id="П-08", kind="new_message", detail="новое сообщение (курьер)"),
    ]

    assert render_changes(changes, results, snapshots, NOW) == (
        "# Изменения · 23.11.2026 10:00\n\n"
        "- **П-08**: новое сообщение (курьер); срок 10:00 → 13:00 → 14:30 (2-е изменение)\n\n"
        "## Требуют внимания\n\n"
        f"{render_item(1, delivery, NOW)}\n"
    )


def test_render_changes_omits_attention_for_green_case(
    results: list[CaseResult], prev: State
) -> None:
    assert prev.cases["П-09"].priority == "green"
    changes = [Change(case_id="П-09", kind="new_message", detail="новое сообщение (получатель)")]

    assert render_changes(changes, results, prev.cases, NOW) == (
        "# Изменения · 23.11.2026 10:00\n\n- **П-09**: новое сообщение (получатель)\n"
    )


def test_render_changes_lists_closed_case_without_attention_item(
    cases: dict[str, Case], parsed: dict[str, ParsedMessage], rules: RulesConfig, prev: State
) -> None:
    closed = cases["П-09"].model_copy(update={"crm_status": "done"})
    results = evaluate_updated(cases, parsed, rules, closed)

    assert render_changes(diff(prev, results), results, snapshot(results, prev), NOW) == (
        "# Изменения · 23.11.2026 10:00\n\n- **П-09**: закрыта по CRM\n"
    )
