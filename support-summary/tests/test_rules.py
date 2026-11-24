from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

from summary.adapters import load_cases
from summary.models import (
    AuthorRole,
    Case,
    CaseType,
    ChatMessage,
    Confirmations,
    CrmStatus,
    Extraction,
    Label,
    ParsedMessage,
    TimeRef,
)

# Keep the rules import grouped here before and after implementation.
# isort: split
from summary.rules import Priority, RulesConfig, evaluate, evaluate_all, load_rules

PROJECT_DIR = Path(__file__).parent.parent
RULES_PATH = PROJECT_DIR / "rules.yaml"
NOW = datetime(2026, 11, 23, 10)
TODAY = NOW + timedelta(hours=8)
TOMORROW = NOW + timedelta(days=1)
BASE_SITUATIONS = (
    pytest.param("card_delivery", "in_transit", NOW, None, "R1", "red", id="due-now"),
    pytest.param(
        "card_delivery",
        "in_transit",
        NOW + timedelta(minutes=1),
        None,
        "R2",
        "yellow",
        id="one-minute",
    ),
    pytest.param(
        "card_delivery",
        "in_transit",
        NOW + timedelta(minutes=119),
        None,
        "R2",
        "yellow",
        id="just-under-two-hours",
    ),
    pytest.param(
        "card_delivery",
        "in_transit",
        NOW + timedelta(hours=2),
        None,
        "R0",
        "gray",
        id="exactly-two-hours",
    ),
    pytest.param(
        "card_delivery",
        "in_transit",
        NOW + timedelta(hours=3),
        None,
        "R0",
        "gray",
        id="three-hours",
    ),
    pytest.param("offboarding", "pending", TODAY, None, "R4", "red", id="manager-missing"),
    pytest.param("offboarding", "pending", TODAY, "manager", "R0", "gray", id="manager-confirmed"),
    pytest.param("onboarding", "pending", TOMORROW, None, "R5", "yellow", id="documents-tomorrow"),
    pytest.param(
        "onboarding",
        "pending",
        TOMORROW,
        "documents",
        "R0",
        "gray",
        id="documents-confirmed",
    ),
    pytest.param("onboarding", "pending", TODAY, None, "R6", "red", id="documents-today"),
    pytest.param("onboarding", "pending", None, None, "R0", "gray", id="onboarding-no-deadline"),
    pytest.param("card_delivery", "pending", TODAY, None, "R0", "gray", id="delivery-pending"),
    pytest.param("card_delivery", "in_transit", None, None, "R0", "gray", id="transit-no-deadline"),
    pytest.param("card_delivery", "accepted", None, None, "R3", "green", id="accepted-no-deadline"),
)


@pytest.fixture
def rules() -> RulesConfig:
    return load_rules(RULES_PATH)


def make_case(
    case_type: CaseType = "card_delivery",
    status: CrmStatus = "pending",
    deadline: datetime | None = TODAY,
    *,
    case_id: str = "case-1",
    confirmed: str | None = None,
    messages: tuple[ChatMessage, ...] = (),
) -> Case:
    return Case(
        case_id=case_id,
        type=case_type,
        crm_status=status,
        deadline=deadline,
        crm_updated_at=NOW,
        confirmations=Confirmations.model_validate({confirmed: NOW} if confirmed else {}),
        chat_messages=list(messages),
    )


def make_message(
    msg_id: str = "message-1",
    *,
    text: str = "Исходное сообщение",
    author: AuthorRole = "courier",
) -> ChatMessage:
    return ChatMessage(msg_id=msg_id, text=text, author_role=author, sent_at=NOW)


def make_parsed(
    message: ChatMessage,
    *,
    labels: tuple[Label, ...] = (),
    times: tuple[TimeRef, ...] = (),
    uncertain: bool = False,
    summary: str = "Краткое содержание",
) -> ParsedMessage:
    return ParsedMessage(
        msg_id=message.msg_id,
        extraction=Extraction(
            summary=summary,
            quote="Цитата из разбора",
            times=list(times),
            labels=list(labels),
            uncertain=uncertain,
        ),
    )


def test_fixture_results(rules: RulesConfig) -> None:
    data_dir = PROJECT_DIR / "data"
    cases = load_cases(data_dir / "cases.json")
    parsed = TypeAdapter(dict[str, ParsedMessage]).validate_json(
        (data_dir / "llm_cache.json").read_text(encoding="utf-8")
    )

    results = evaluate_all(cases, parsed, NOW, rules)

    assert [(r.case.case_id, r.fired, r.priority, r.overdue) for r in results] == [
        ("П-08", ["R1", "C1", "C3"], "red", True),
        ("С-24", ["R4", "C2"], "red", False),
        ("С-17", ["R5", "C2"], "yellow", False),
        ("П-09", ["R3"], "green", False),
    ]
    delivery = results[0]
    assert delivery.notes == ["⚠️ Расхождение: по CRM 10:00, по словам курьера 13:00."]
    assert delivery.action == "Предупредить точку, подтвердить новое время, обновить срок в CRM"


@pytest.mark.parametrize(
    ("case_type", "status", "deadline", "confirmed", "base_rule", "priority"),
    BASE_SITUATIONS,
)
def test_base_rules(
    rules: RulesConfig,
    case_type: CaseType,
    status: CrmStatus,
    deadline: datetime | None,
    confirmed: str | None,
    base_rule: str,
    priority: Priority,
) -> None:
    case = make_case(case_type, status, deadline, confirmed=confirmed)

    result = evaluate(case, {}, NOW, rules)

    assert (result.base_rule, result.fired, result.priority) == (base_rule, [base_rule], priority)
    assert result.action == rules.base[base_rule].action
    if base_rule == "R0":
        assert result.action == "Разобрать вручную"


def test_confirmed_acceptance_suppresses_c2(rules: RulesConfig) -> None:
    message = make_message()
    case = make_case(status="accepted", confirmed="acceptance", messages=(message,))
    parsed = make_parsed(message, labels=("claims_done",))

    result = evaluate(case, {message.msg_id: parsed}, NOW, rules)

    assert result.fired == ["R3"]
    assert result.notes == []


@pytest.mark.parametrize(
    ("status", "deadline", "base_rule", "priority"),
    [
        ("accepted", TODAY, "R3", "yellow"),
        ("in_transit", NOW + timedelta(hours=1), "R2", "red"),
        ("in_transit", NOW, "R1", "red"),
        ("pending", TODAY, "R0", "yellow"),
    ],
)
def test_problem_bumps_priority(
    rules: RulesConfig,
    status: CrmStatus,
    deadline: datetime,
    base_rule: str,
    priority: Priority,
) -> None:
    message = make_message()
    case = make_case(status=status, deadline=deadline, messages=(message,))
    parsed = make_parsed(message, labels=("reports_problem",))

    result = evaluate(case, {message.msg_id: parsed}, NOW, rules)

    assert result.fired == [base_rule, "C3"]
    assert result.priority == priority
    assert result.notes == []


@pytest.mark.parametrize(
    ("eta", "fires"),
    [("12:01", True), ("12:00", False), ("11:59", False)],
)
def test_c1_requires_eta_after_deadline(rules: RulesConfig, eta: str, fires: bool) -> None:
    message = make_message()
    case = make_case(
        status="accepted",
        deadline=NOW + timedelta(hours=2),
        messages=(message,),
    )
    parsed = make_parsed(message, times=(TimeRef(time=eta, role="new_eta"),))

    result = evaluate(case, {message.msg_id: parsed}, NOW, rules)

    if fires:
        assert result.fired == ["R3", "C1"]
        assert result.priority == "yellow"
    else:
        assert result.fired == ["R3"]
        assert result.priority == "green"
        assert result.notes == []


def test_combined_c1_and_c3_bump_green_only_once(rules: RulesConfig) -> None:
    message = make_message()
    case = make_case(status="accepted", messages=(message,))
    parsed = make_parsed(
        message,
        labels=("reports_problem",),
        times=(TimeRef(time="19:00", role="new_eta"),),
    )

    result = evaluate(case, {message.msg_id: parsed}, NOW, rules)

    assert result.fired == ["R3", "C1", "C3"]
    assert result.priority == "yellow"


def test_action_request_uses_extraction_summary(rules: RulesConfig) -> None:
    message = make_message()
    case = make_case(messages=(message,))
    parsed = make_parsed(message, labels=("asks_action",), summary="Уточнить адрес")

    result = evaluate(case, {message.msg_id: parsed}, NOW, rules)

    assert result.fired == ["R0", "C4"]
    assert result.notes == ["❓ В чате запрос, нужен ответ: Уточнить адрес"]


@pytest.mark.parametrize("parse_state", ["uncertain", "failed", "missing"])
def test_c5_uses_original_text(rules: RulesConfig, parse_state: str) -> None:
    message = make_message(text="Оригинал сообщения")
    case = make_case(messages=(message,))
    parsed = {}
    if parse_state == "uncertain":
        parsed[message.msg_id] = make_parsed(message, uncertain=True)
    elif parse_state == "failed":
        parsed[message.msg_id] = ParsedMessage(msg_id=message.msg_id, parse_failed=True)

    result = evaluate(case, parsed, NOW, rules)

    assert result.fired == ["R0", "C5"]
    assert result.notes == ["⚠️ Разбор неоднозначен, прочитать оригинал: «Оригинал сообщения»"]
    [view] = result.messages
    assert view.message == message
    assert view.parsed.msg_id == message.msg_id
    assert view.parsed.parse_failed is (parse_state != "uncertain")
    if parse_state == "missing":
        assert view.parsed.extraction is None


def test_chat_notes_group_by_rule_and_preserve_message_order(rules: RulesConfig) -> None:
    first = make_message("first", text="Первый оригинал", author="employee")
    second = make_message("second", text="Второй оригинал", author="manager")
    case = make_case(deadline=NOW.replace(hour=8), messages=(first, second))
    first_parse = make_parsed(
        first,
        labels=("asks_action", "claims_done"),
        times=(
            TimeRef(time="7:00", role="new_eta"),
            TimeRef(time="9:05", role="new_eta"),
            TimeRef(time="23:00", role="reference"),
        ),
        uncertain=True,
        summary="Первый запрос",
    )
    second_parse = make_parsed(
        second,
        labels=("claims_done", "asks_action"),
        times=(TimeRef(time="10:15", role="new_eta"),),
        uncertain=True,
        summary="Второй запрос",
    )

    result = evaluate(case, {second.msg_id: second_parse, first.msg_id: first_parse}, NOW, rules)

    assert result.fired == ["R0", "C1", "C2", "C4", "C5"]
    assert result.notes == [
        "⚠️ Расхождение: по CRM 08:00, по словам сотрудника 09:05.",
        "⚠️ Расхождение: по CRM 08:00, по словам руководителя 10:15.",
        "⚠️ В чате утверждается, что сделано. Слова подтверждением не считаются.",
        "❓ В чате запрос, нужен ответ: Первый запрос",
        "❓ В чате запрос, нужен ответ: Второй запрос",
        "⚠️ Разбор неоднозначен, прочитать оригинал: «Первый оригинал»",
        "⚠️ Разбор неоднозначен, прочитать оригинал: «Второй оригинал»",
    ]


def test_evaluate_all_drops_done_cases(rules: RulesConfig) -> None:
    done = make_case(status="done", case_id="closed")
    opened = make_case(case_id="open")

    assert [r.case.case_id for r in evaluate_all([done, opened], {}, NOW, rules)] == ["open"]


def test_evaluate_all_sorts_by_priority_overdue_deadline_and_id(rules: RulesConfig) -> None:
    cases = [
        make_case(deadline=None, case_id="gray-no-deadline"),
        make_case(status="accepted", deadline=NOW, case_id="green"),
        make_case("offboarding", deadline=TODAY, case_id="red-later"),
        make_case("offboarding", "accepted", NOW - timedelta(hours=1), case_id="red-earlier"),
        make_case("offboarding", deadline=NOW + timedelta(hours=1), case_id="red-b"),
        make_case(deadline=NOW - timedelta(hours=1), case_id="gray-overdue"),
        make_case("offboarding", deadline=NOW + timedelta(hours=1), case_id="red-a"),
        make_case(status="in_transit", deadline=NOW, case_id="red-overdue"),
        make_case(deadline=TODAY, case_id="gray-dated"),
        make_case(status="in_transit", deadline=NOW + timedelta(hours=1), case_id="yellow"),
    ]

    results = evaluate_all(cases, {}, NOW, rules)

    assert [r.case.case_id for r in results] == [
        "red-overdue",
        "red-earlier",
        "red-a",
        "red-b",
        "red-later",
        "yellow",
        "gray-overdue",
        "gray-dated",
        "gray-no-deadline",
        "green",
    ]


@pytest.mark.parametrize(("section", "rule_id"), [("base", "R3"), ("chat", "C4")])
def test_load_rules_rejects_missing_rules(tmp_path: Path, section: str, rule_id: str) -> None:
    payload = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    del payload[section][rule_id]
    path = tmp_path / "incomplete-rules.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        load_rules(path)
