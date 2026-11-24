from datetime import datetime

import pytest

from summary.models import (
    Case,
    CaseType,
    ChatMessage,
    Confirmations,
    CrmStatus,
    Extraction,
    ParsedMessage,
)
from summary.render import crm_line, day_word, fmt_when, render_full, render_item
from summary.rules import CaseResult, MessageView

NOW = datetime(2026, 11, 23, 10, 0)
TODAY_EVENING = datetime(2026, 11, 23, 18, 0)
TOMORROW_MORNING = datetime(2026, 11, 24, 9, 0)


def make_case(
    case_type: CaseType = "card_delivery",
    deadline: datetime | None = None,
    *,
    status: CrmStatus = "pending",
    confirmations: Confirmations | None = None,
    case_id: str = "П-01",
) -> Case:
    return Case(
        case_id=case_id,
        type=case_type,
        crm_status=status,
        deadline=deadline,
        crm_updated_at=NOW,
        confirmations=confirmations if confirmations is not None else Confirmations(),
    )


def make_result(
    case: Case,
    *,
    overdue: bool = False,
    message: MessageView | None = None,
) -> CaseResult:
    return CaseResult(
        case=case,
        priority="gray",
        base_rule="R0",
        fired=["R0"],
        overdue=overdue,
        action="Разобрать вручную.",
        notes=[],
        messages=[] if message is None else [message],
    )


def make_message(text: str, extraction: Extraction | None = None) -> MessageView:
    message = ChatMessage(msg_id="m1", author_role="manager", sent_at=NOW, text=text)
    return MessageView(
        message=message,
        parsed=ParsedMessage(
            msg_id=message.msg_id,
            extraction=extraction,
            parse_failed=extraction is None,
        ),
    )


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (TODAY_EVENING, "сегодня"),
        (TOMORROW_MORNING, "завтра"),
        (datetime(2026, 11, 22, 23, 59), "вчера"),
        (datetime(2026, 11, 25, 9, 0), "25.11"),
    ],
)
def test_day_word_uses_calendar_dates(when: datetime, expected: str) -> None:
    assert day_word(when, NOW) == expected


def test_fmt_when_includes_zero_padded_time() -> None:
    assert fmt_when(TOMORROW_MORNING, NOW) == "завтра 09:00"


def test_crm_line_card_delivery_without_deadline() -> None:
    assert crm_line(make_case(), NOW) == "срок доставки не указан, статус «Ожидает»."


def test_crm_line_onboarding_with_confirmed_documents() -> None:
    confirmations = Confirmations(documents=datetime(2026, 11, 23, 8, 15))
    case = make_case("onboarding", TOMORROW_MORNING, confirmations=confirmations)

    assert crm_line(case, NOW) == "первый выход завтра 09:00, документы получены сегодня 08:15."


def test_crm_line_onboarding_without_deadline() -> None:
    assert crm_line(make_case("onboarding"), NOW).startswith("дата первого выхода не указана")


def test_crm_line_offboarding_with_manager_confirmation() -> None:
    confirmations = Confirmations(manager=datetime(2026, 11, 23, 9, 0))
    case = make_case("offboarding", TODAY_EVENING, confirmations=confirmations)

    assert crm_line(case, NOW) == (
        "последний рабочий день сегодня 18:00, подтверждение руководителя получено сегодня 09:00."
    )


def test_crm_line_offboarding_without_deadline() -> None:
    assert crm_line(make_case("offboarding"), NOW).startswith("последний рабочий день не указан")


@pytest.mark.parametrize(
    ("status", "deadline", "overdue", "title"),
    [
        ("in_transit", NOW, True, "⚪ **1. П-01 · Поставка карт · срок уже нарушен**"),
        ("accepted", TODAY_EVENING, False, "⚪ **1. П-01 · Поставка карт · принято по CRM**"),
        ("pending", TODAY_EVENING, False, "⚪ **1. П-01 · Поставка карт · срок сегодня 18:00**"),
        ("pending", None, False, "⚪ **1. П-01 · Поставка карт**"),
    ],
)
def test_render_item_title_suffix(
    status: CrmStatus, deadline: datetime | None, overdue: bool, title: str
) -> None:
    result = make_result(make_case(deadline=deadline, status=status), overdue=overdue)

    assert render_item(1, result, NOW).split("\n", 1)[0] == title


def test_render_item_quotes_message_at_200_characters_after_stripping() -> None:
    text = "а" * 200
    extraction = Extraction(summary="Краткое содержание", quote=text)
    message = make_message(f" {text} ", extraction)
    result = make_result(make_case(), message=message)

    assert render_item(1, result, NOW).splitlines()[2] == (
        f"**Чат (не подтверждено), руководитель:** «{text}»"
    )


def test_render_item_summarizes_201_character_message_with_extraction() -> None:
    text = "а" * 200 + "б"
    extraction = Extraction(summary="Краткое содержание", quote=text)
    result = make_result(make_case(), message=make_message(text, extraction))

    rendered = render_item(1, result, NOW)

    assert rendered.splitlines()[2] == (
        "**Чат (не подтверждено), руководитель, кратко:** Краткое содержание"
    )
    assert text not in rendered


def test_render_item_truncates_201_character_message_when_parse_failed() -> None:
    first_200 = "а" * 200
    result = make_result(make_case(), message=make_message(first_200 + "б"))

    assert render_item(1, result, NOW).splitlines()[2] == (
        f"**Чат (не подтверждено), руководитель:** «{first_200}…»"
    )


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_render_item_labels_empty_messages(text: str) -> None:
    result = make_result(make_case(), message=make_message(text))

    assert render_item(1, result, NOW).splitlines()[2] == (
        "**Чат (не подтверждено), руководитель:** (пустое сообщение)"
    )


def test_render_item_preserves_single_terminal_period_without_newline() -> None:
    result = make_result(make_case())

    assert render_item(1, result, NOW).split("\n")[-1] == "→ Разобрать вручную."


def test_render_full_counts_gray_items_and_preserves_input_order_and_spacing() -> None:
    results = [make_result(make_case(case_id=case_id)) for case_id in ("П-02", "П-01")]

    assert render_full(results, NOW) == (
        "# Сводка support · 23.11.2026 10:00\n"
        "\n"
        "🔴 0 · 🟡 0 · ⚪ 2 · 🟢 0\n"
        "\n"
        "⚪ **1. П-02 · Поставка карт**\n"
        "**CRM:** срок доставки не указан, статус «Ожидает».\n"
        "→ Разобрать вручную.\n"
        "\n"
        "⚪ **2. П-01 · Поставка карт**\n"
        "**CRM:** срок доставки не указан, статус «Ожидает».\n"
        "→ Разобрать вручную.\n"
    )


def test_render_full_without_results() -> None:
    assert render_full([], NOW) == "# Сводка support · 23.11.2026 10:00\n\nОткрытых заявок нет.\n"
