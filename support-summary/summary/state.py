import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from summary.models import CrmStatus
from summary.render import EMOJI, ROLE_RU, STATUS_RU, render_item
from summary.rules import SEVERITY, CaseResult, Priority


class CaseSnapshot(BaseModel):
    priority: Priority
    crm_status: CrmStatus
    deadline: datetime | None
    deadline_history: list[str]
    msg_ids: list[str]


class State(BaseModel):
    last_run: datetime | None = None
    last_full_date: date | None = None
    cases: dict[str, CaseSnapshot] = Field(default_factory=dict)


ChangeKind = Literal[
    "new_case", "closed", "priority_up", "new_message", "deadline_changed", "status_changed"
]


class Change(BaseModel):
    case_id: str
    kind: ChangeKind
    detail: str


def load_state(path: str | Path) -> State:
    try:
        data = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return State()
    return State.model_validate_json(data)


def save_state(path: str | Path, state: State) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2)
    path.write_text(data, encoding="utf-8")


def snapshot(results: list[CaseResult], prev: State) -> dict[str, CaseSnapshot]:
    snapshots = {}
    for result in results:
        case = result.case
        previous = prev.cases.get(case.case_id)
        history = previous.deadline_history.copy() if previous is not None else []
        deadline = case.deadline
        if deadline is not None and (time := deadline.strftime("%H:%M")) not in history[-1:]:
            history.append(time)
        snapshots[case.case_id] = CaseSnapshot(
            priority=result.priority,
            crm_status=case.crm_status,
            deadline=deadline,
            deadline_history=history,
            msg_ids=[message.msg_id for message in case.chat_messages],
        )
    return snapshots


def diff(prev: State, results: list[CaseResult]) -> list[Change]:
    changes = []
    current_ids = set()
    for result in results:
        case = result.case
        case_id = case.case_id
        current_ids.add(case_id)
        previous = prev.cases.get(case_id)
        if previous is None:
            changes.append(Change(case_id=case_id, kind="new_case", detail="новая заявка"))
            continue

        if SEVERITY[result.priority] > SEVERITY[previous.priority]:
            detail = f"приоритет {EMOJI[previous.priority]} → {EMOJI[result.priority]}"
            changes.append(Change(case_id=case_id, kind="priority_up", detail=detail))
        if case.crm_status != previous.crm_status:
            detail = f"статус «{STATUS_RU[previous.crm_status]}» → «{STATUS_RU[case.crm_status]}»"
            changes.append(Change(case_id=case_id, kind="status_changed", detail=detail))
        if case.deadline != previous.deadline:
            changes.append(Change(case_id=case_id, kind="deadline_changed", detail=""))

        previous_ids = set(previous.msg_ids)
        for message in case.chat_messages:
            if message.msg_id not in previous_ids:
                detail = f"новое сообщение ({ROLE_RU[message.author_role]})"
                changes.append(Change(case_id=case_id, kind="new_message", detail=detail))

    for case_id in sorted(prev.cases.keys() - current_ids):
        changes.append(Change(case_id=case_id, kind="closed", detail="закрыта по CRM"))
    return changes


def is_full_run(prev: State, now: datetime, full_at: time) -> bool:
    return prev.last_full_date != now.date() and now.time() >= full_at


def render_changes(
    changes: list[Change],
    results: list[CaseResult],
    new_snapshots: dict[str, CaseSnapshot],
    now: datetime,
) -> str:
    if not changes:
        return ""

    details: dict[str, list[str]] = {}
    changed_deadlines = set()
    for change in changes:
        case_details = details.setdefault(change.case_id, [])
        if change.detail:
            case_details.append(change.detail)
        if change.kind == "deadline_changed":
            changed_deadlines.add(change.case_id)

    for case_id in changed_deadlines:
        history = new_snapshots[case_id].deadline_history
        details[case_id].append(f"срок {' → '.join(history)} ({len(history) - 1}-е изменение)")

    bullets = "\n".join(
        f"- **{case_id}**: {'; '.join(texts)}" for case_id, texts in details.items()
    )
    sections = [f"# Изменения · {now:%d.%m.%Y %H:%M}", bullets]
    attention = [
        result
        for result in results
        if result.case.case_id in details
        and result.case.is_open
        and result.priority in ("red", "yellow")
    ]
    if attention:
        sections.append("## Требуют внимания")
        sections.extend(render_item(n, result, now) for n, result in enumerate(attention, start=1))

    return "\n\n".join(sections) + "\n"
