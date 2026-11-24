from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from summary.models import Case, ChatMessage, Extraction, Label, ParsedMessage

Priority = Literal["red", "yellow", "gray", "green"]
SEVERITY: dict[Priority, int] = {"green": 0, "gray": 1, "yellow": 2, "red": 3}
SORT_RANK: dict[Priority, int] = {"red": 0, "yellow": 1, "gray": 2, "green": 3}
BUMP: dict[Priority, Priority] = {
    "green": "yellow",
    "gray": "yellow",
    "yellow": "red",
    "red": "red",
}
AUTHOR_GENITIVE = {
    "employee": "сотрудника",
    "manager": "руководителя",
    "courier": "курьера",
    "recipient": "получателя",
}


class BaseRule(BaseModel):
    priority: Priority
    action: str


class RulesConfig(BaseModel):
    soon_hours: float
    base: dict[str, BaseRule]
    chat: dict[str, str]


def load_rules(path: str | Path) -> RulesConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = payload.get("base") or {}
    chat = payload.get("chat") or {}
    missing = ({"R0", "R1", "R2", "R3", "R4", "R5", "R6"} - base.keys()) | (
        {"C1", "C2", "C4", "C5"} - chat.keys()
    )
    if missing:
        raise ValueError(f"Missing rules: {', '.join(sorted(missing))}")

    return RulesConfig(soon_hours=payload["thresholds"]["soon_hours"], base=base, chat=chat)


class MessageView(BaseModel):
    message: ChatMessage
    parsed: ParsedMessage


class CaseResult(BaseModel):
    case: Case
    priority: Priority
    base_rule: str
    fired: list[str]
    overdue: bool
    action: str
    notes: list[str]
    messages: list[MessageView]


def _base_rule(case: Case, now: datetime, soon_hours: float) -> str:
    deadline = case.deadline
    if deadline is not None and case.type == "card_delivery" and case.crm_status == "in_transit":
        if now >= deadline:
            return "R1"
        if deadline - now < timedelta(hours=soon_hours):
            return "R2"
    if case.type == "card_delivery" and case.crm_status == "accepted":
        return "R3"
    if deadline is not None:
        days_remaining = (deadline.date() - now.date()).days
        if (
            case.type == "offboarding"
            and days_remaining == 0
            and case.confirmations.manager is None
        ):
            return "R4"
        if case.type == "onboarding" and case.confirmations.documents is None:
            return {1: "R5", 0: "R6"}.get(days_remaining, "R0")
    return "R0"


def _late_eta(extraction: Extraction, deadline: datetime | None) -> time | None:
    if deadline is None:
        return None
    estimates = [ref.time for ref in extraction.times if ref.role == "new_eta"]
    if not estimates:
        return None
    hour, minute = estimates[-1].split(":")
    eta = time(int(hour), int(minute))
    return eta if datetime.combine(deadline.date(), eta) > deadline else None


def _message_views(
    messages: list[ChatMessage], parsed: dict[str, ParsedMessage]
) -> list[MessageView]:
    views = []
    for message in messages:
        result = parsed.get(message.msg_id)
        if result is None:
            result = ParsedMessage(msg_id=message.msg_id, parse_failed=True)
        views.append(MessageView(message=message, parsed=result))
    return views


def _chat_rules(
    case: Case, messages: list[MessageView], chat: dict[str, str]
) -> tuple[list[str], list[str]]:
    deadline = case.deadline
    labels: set[Label] = set()
    discrepancies = []
    requests = []
    uncertain = []
    for view in messages:
        extraction = view.parsed.extraction
        if extraction is None or extraction.uncertain:
            uncertain.append(chat["C5"].format(text=view.message.text))
        if extraction is None:
            continue

        labels.update(extraction.labels)
        eta = _late_eta(extraction, deadline)
        if deadline is not None and eta is not None:
            discrepancies.append(
                chat["C1"].format(
                    crm=deadline.strftime("%H:%M"),
                    author=AUTHOR_GENITIVE[view.message.author_role],
                    chat=eta.strftime("%H:%M"),
                )
            )
        if "asks_action" in extraction.labels:
            requests.append(chat["C4"].format(summary=extraction.summary))

    confirmation = {
        "onboarding": case.confirmations.documents,
        "offboarding": case.confirmations.manager,
        "card_delivery": case.confirmations.acceptance,
    }[case.type]
    claims = [chat["C2"]] if "claims_done" in labels and confirmation is None else []

    fired = []
    for rule_id, matched in (
        ("C1", bool(discrepancies)),
        ("C2", bool(claims)),
        ("C3", "reports_problem" in labels),
        ("C4", bool(requests)),
        ("C5", bool(uncertain)),
    ):
        if matched:
            fired.append(rule_id)
    return fired, discrepancies + claims + requests + uncertain


def evaluate(
    case: Case, parsed: dict[str, ParsedMessage], now: datetime, rules: RulesConfig
) -> CaseResult:
    base_rule = _base_rule(case, now, rules.soon_hours)
    base = rules.base[base_rule]
    messages = _message_views(case.chat_messages, parsed)
    fired, notes = _chat_rules(case, messages, rules.chat)

    priority = BUMP[base.priority] if "C3" in fired else base.priority
    if "C1" in fired and SEVERITY[priority] < SEVERITY["yellow"]:
        priority = "yellow"

    deadline = case.deadline
    return CaseResult(
        case=case,
        priority=priority,
        base_rule=base_rule,
        fired=[base_rule, *fired],
        overdue=deadline is not None
        and now >= deadline
        and case.crm_status in ("pending", "in_transit"),
        action=base.action,
        notes=notes,
        messages=messages,
    )


def evaluate_all(
    cases: list[Case], parsed: dict[str, ParsedMessage], now: datetime, rules: RulesConfig
) -> list[CaseResult]:
    results = [evaluate(case, parsed, now, rules) for case in cases if case.is_open]
    return sorted(
        results,
        key=lambda result: (
            SORT_RANK[result.priority],
            not result.overdue,
            result.case.deadline or datetime.max,
            result.case.case_id,
        ),
    )
