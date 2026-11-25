from collections import Counter
from datetime import datetime

from summary.models import AuthorRole, Case, CaseType, CrmStatus
from summary.rules import CaseResult, MessageView, Priority

EMOJI: dict[Priority, str] = {"red": "🔴", "yellow": "🟡", "gray": "⚪", "green": "🟢"}
TYPE_RU: dict[CaseType, str] = {
    "onboarding": "Оформление сотрудника",
    "offboarding": "Увольнение сотрудника",
    "card_delivery": "Поставка карт",
}
STATUS_RU: dict[CrmStatus, str] = {
    "pending": "Ожидает",
    "in_transit": "В пути",
    "accepted": "Принято",
    "done": "Завершено",
}
ROLE_RU: dict[AuthorRole, str] = {
    "employee": "сотрудник",
    "manager": "руководитель",
    "courier": "курьер",
    "recipient": "получатель",
}
SHORT_LIMIT = 200


def day_word(dt: datetime, now: datetime) -> str:
    days = (dt.date() - now.date()).days
    return {-1: "вчера", 0: "сегодня", 1: "завтра"}.get(days, f"{dt:%d.%m}")


def fmt_when(dt: datetime, now: datetime) -> str:
    return f"{day_word(dt, now)} {dt:%H:%M}"


def crm_line(case: Case, now: datetime) -> str:
    deadline = case.deadline
    confirmations = case.confirmations

    if case.type == "card_delivery":
        status = f"статус «{STATUS_RU[case.crm_status]}»."
        if deadline is not None:
            line = f"доставка {day_word(deadline, now)} до {deadline:%H:%M}, {status}"
        elif case.crm_status == "accepted":
            line = status
        else:
            line = f"срок доставки не указан, {status}"
        if confirmations.acceptance is not None:
            line += f" Приёмка подтверждена {fmt_when(confirmations.acceptance, now)}."
        return line

    if case.type == "onboarding":
        first_day = (
            f"первый выход {day_word(deadline, now)}"
            if deadline is not None
            else "дата первого выхода не указана"
        )
        documents = (
            f"получены {fmt_when(confirmations.documents, now)}"
            if confirmations.documents is not None
            else "ожидаются"
        )
        return f"{first_day}, документы {documents}."

    last_day = (
        f"последний рабочий день {day_word(deadline, now)}"
        if deadline is not None
        else "последний рабочий день не указан"
    )
    manager = (
        f"получено {fmt_when(confirmations.manager, now)}"
        if confirmations.manager is not None
        else "отсутствует"
    )
    return f"{last_day}, подтверждение руководителя {manager}."


def _chat_line(view: MessageView) -> str:
    text = view.message.text.strip()
    prefix = f"**Чат (не подтверждено), {ROLE_RU[view.message.author_role]}"
    if not text:
        return f"{prefix}:** (пустое сообщение)"
    if len(text) > SHORT_LIMIT:
        extraction = view.parsed.extraction
        if extraction is not None:
            return f"{prefix}, кратко:** {extraction.summary}"
        text = f"{text[:SHORT_LIMIT]}…"
    return f"{prefix}:** «{text}»"


def render_item(n: int, r: CaseResult, now: datetime) -> str:
    case = r.case
    title = f"{n}. {case.case_id} · {TYPE_RU[case.type]}"
    if r.overdue:
        title += " · срок уже нарушен"
    elif case.crm_status == "accepted":
        title += " · принято по CRM"
    elif case.deadline is not None:
        when = fmt_when if case.type == "card_delivery" else day_word
        title += f" · срок {when(case.deadline, now)}"

    lines = [
        f"{EMOJI[r.priority]} **{title}**",
        f"**CRM:** {crm_line(case, now)}",
        *(_chat_line(view) for view in r.messages),
        *r.notes,
        f"→ {r.action.rstrip('.')}.",
    ]
    return "\n".join(lines)


def render_full(results: list[CaseResult], now: datetime) -> str:
    header = f"# Сводка support · {now:%d.%m.%Y %H:%M}"
    if not results:
        return f"{header}\n\nОткрытых заявок нет.\n"

    counts = Counter(result.priority for result in results)
    counts_line = " · ".join(f"{emoji} {counts[priority]}" for priority, emoji in EMOJI.items())
    items = (render_item(n, result, now) for n, result in enumerate(results, start=1))
    return "\n\n".join((header, counts_line, *items)) + "\n"
