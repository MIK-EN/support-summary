from datetime import datetime
from itertools import combinations

import pytest

from summary.models import CaseType, CrmStatus, Label, ParsedMessage, TimeRef

# Keep the rules import grouped here before and after implementation.
# isort: split
from summary.rules import SEVERITY, Priority, RulesConfig, evaluate, evaluate_all, load_rules

from .test_rules import BASE_SITUATIONS, NOW, RULES_PATH, make_case, make_message, make_parsed

LABELS: tuple[Label, ...] = ("claims_done", "reports_problem", "asks_action", "info")
LABEL_SETS = tuple(
    subset for size in range(len(LABELS) + 1) for subset in combinations(LABELS, size)
)


@pytest.fixture
def rules() -> RulesConfig:
    return load_rules(RULES_PATH)


@pytest.mark.parametrize(
    ("case_type", "status", "deadline", "confirmed", "base_rule", "priority"),
    BASE_SITUATIONS,
)
@pytest.mark.parametrize("labels", LABEL_SETS, ids=lambda labels: "+".join(labels) or "no-labels")
@pytest.mark.parametrize("uncertain", [False, True], ids=["certain", "uncertain"])
@pytest.mark.parametrize("has_eta", [False, True], ids=["no-eta", "new-eta"])
@pytest.mark.parametrize("parse_failed", [False, True], ids=["parsed-only", "with-failed"])
def test_chat_never_lowers_priority_or_closes_case(
    rules: RulesConfig,
    case_type: CaseType,
    status: CrmStatus,
    deadline: datetime | None,
    confirmed: str | None,
    base_rule: str,
    priority: Priority,
    labels: tuple[Label, ...],
    uncertain: bool,
    has_eta: bool,
    parse_failed: bool,
) -> None:
    case = make_case(case_type, status, deadline, confirmed=confirmed)
    baseline = evaluate(case, {}, NOW, rules)
    assert (baseline.base_rule, baseline.priority) == (base_rule, priority)

    message = make_message()
    messages = [message]
    times = (TimeRef(time="23:59", role="new_eta"),) if has_eta else ()
    parsed = {
        message.msg_id: make_parsed(message, labels=labels, times=times, uncertain=uncertain),
    }
    if parse_failed:
        # A failed parse cannot also have an extraction.
        failed = make_message("failed")
        messages.append(failed)
        parsed[failed.msg_id] = ParsedMessage(msg_id=failed.msg_id, parse_failed=True)

    with_chat = make_case(
        case_type, status, deadline, confirmed=confirmed, messages=tuple(messages)
    )
    [result] = evaluate_all([with_chat], parsed, NOW, rules)

    assert SEVERITY[result.priority] >= SEVERITY[baseline.priority]
    assert result.case.is_open
    assert result.case.crm_status == status
    assert result.case.confirmations == case.confirmations


def test_every_open_case_keeps_all_its_messages_in_order(rules: RulesConfig) -> None:
    info = make_message("info")
    empty = make_message("empty", text="")
    failed = make_message("failed")
    missing = make_message("missing")
    cases = [
        make_case(case_id="first", messages=(info, empty, failed)),
        make_case(case_id="second", status="accepted", messages=(missing,)),
    ]
    parsed = {
        failed.msg_id: ParsedMessage(msg_id=failed.msg_id, parse_failed=True),
        empty.msg_id: make_parsed(empty),
        info.msg_id: make_parsed(info, labels=("info",)),
    }

    results = evaluate_all(cases, parsed, NOW, rules)

    assert {
        result.case.case_id: [view.message for view in result.messages] for result in results
    } == {case.case_id: case.chat_messages for case in cases}
    for result in results:
        for view in result.messages:
            assert view.parsed.msg_id == view.message.msg_id
            if view.message.msg_id in parsed:
                assert view.parsed == parsed[view.message.msg_id]
            else:
                assert view.parsed.parse_failed
                assert view.parsed.extraction is None


def test_unmatched_open_case_is_returned_as_r0(rules: RulesConfig) -> None:
    case = make_case(deadline=None)

    [result] = evaluate_all([case], {}, NOW, rules)

    assert result.case == case
    assert result.base_rule == "R0"
    assert result.fired == ["R0"]
    assert result.priority == "gray"
