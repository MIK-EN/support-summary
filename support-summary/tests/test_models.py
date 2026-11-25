import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from summary.models import Case, Extraction, ParsedMessage, TimeRef

# Keep the missing-model error visible during the tests-only phase.
# isort: split
from summary.adapters import load_cases

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def fixture_cases():
    return load_cases(DATA_DIR / "cases.json")


@pytest.fixture
def case_payload():
    return {
        "case_id": "case-1",
        "type": "onboarding",
        "crm_status": "pending",
        "crm_updated_at": "2026-11-23T08:00",
    }


@pytest.fixture
def message_payload():
    return {
        "msg_id": "message-1",
        "author_role": "employee",
        "sent_at": "2026-11-23T09:00",
        "text": "Отправил документы",
    }


def write_cases(tmp_path, payload):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_cases_fixture_in_file_order(fixture_cases):
    assert len(fixture_cases) == 4
    assert all(isinstance(case, Case) for case in fixture_cases)
    assert [case.case_id for case in fixture_cases] == ["С-17", "С-24", "П-08", "П-09"]
    assert [(case.type, case.crm_status) for case in fixture_cases] == [
        ("onboarding", "pending"),
        ("offboarding", "pending"),
        ("card_delivery", "in_transit"),
        ("card_delivery", "accepted"),
    ]
    assert [case.deadline for case in fixture_cases] == [
        datetime(2026, 11, 24),
        datetime(2026, 11, 23),
        datetime(2026, 11, 23, 10, 0),
        None,
    ]
    assert [case.crm_updated_at for case in fixture_cases] == [
        datetime(2026, 11, 20, 12, 0),
        datetime(2026, 11, 21, 15, 0),
        datetime(2026, 11, 23, 8, 0),
        datetime(2026, 11, 23, 9, 20),
    ]
    assert [case.confirmations.acceptance for case in fixture_cases] == [
        None,
        None,
        None,
        datetime(2026, 11, 23, 9, 20),
    ]
    for case in fixture_cases:
        assert case.confirmations.manager is None
        assert case.confirmations.documents is None
        assert len(case.chat_messages) == 1

    messages = [case.chat_messages[0] for case in fixture_cases]
    assert [
        (message.msg_id, message.author_role, message.sent_at, message.text) for message in messages
    ] == [
        (
            "c17-1",
            "employee",
            datetime(2026, 11, 23, 8, 40),
            "Вчера отправил фотографии документов руководителю в личку",
        ),
        (
            "c24-1",
            "employee",
            datetime(2026, 11, 23, 9, 10),
            "С руководителем всё обсудили, он в курсе",
        ),
        (
            "p08-1",
            "courier",
            datetime(2026, 11, 23, 9, 55),
            "Задерживаюсь, раньше 13:00 не приеду",
        ),
        (
            "p09-1",
            "recipient",
            datetime(2026, 11, 23, 9, 25),
            "Карты получили, количество проверили",
        ),
    ]


def test_fixture_cases_are_open(fixture_cases):
    assert all(case.is_open is True for case in fixture_cases)


def test_done_case_is_closed(case_payload):
    case = Case.model_validate({**case_payload, "crm_status": "done"})

    assert case.is_open is False


def test_load_cases_fills_omitted_optional_fields(tmp_path, case_payload):
    path = write_cases(tmp_path, [case_payload])

    [case] = load_cases(str(path))

    assert case.deadline is None
    assert case.confirmations.manager is None
    assert case.confirmations.documents is None
    assert case.confirmations.acceptance is None
    assert case.chat_messages == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("type", "unknown"), ("crm_status", "unknown"), ("extra_field", "extra"), ("case_id", "")],
)
def test_load_cases_rejects_invalid_case_fields(tmp_path, case_payload, field, value):
    path = write_cases(tmp_path, [{**case_payload, field: value}])

    with pytest.raises(ValidationError):
        load_cases(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("author_role", "unknown"), ("extra_field", "extra"), ("msg_id", "")],
)
def test_load_cases_rejects_invalid_message_fields(
    tmp_path, case_payload, message_payload, field, value
):
    message = {**message_payload, field: value}
    path = write_cases(tmp_path, [{**case_payload, "chat_messages": [message]}])

    with pytest.raises(ValidationError):
        load_cases(path)


def test_load_cases_accepts_empty_message_text(tmp_path, case_payload, message_payload):
    message = {**message_payload, "text": ""}
    path = write_cases(tmp_path, [{**case_payload, "chat_messages": [message]}])

    [case] = load_cases(path)
    [loaded_message] = case.chat_messages

    assert loaded_message.text == ""


def test_load_cases_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "missing.json")


def test_load_cases_rejects_top_level_object(tmp_path, case_payload):
    path = write_cases(tmp_path, case_payload)

    with pytest.raises(ValueError) as error:
        load_cases(path)

    assert not isinstance(error.value, ValidationError)


def test_load_cases_rejects_duplicate_case_id(tmp_path, case_payload):
    case = {**case_payload, "case_id": "duplicate-case"}
    path = write_cases(tmp_path, [case, case])

    with pytest.raises(ValueError, match="duplicate-case"):
        load_cases(path)


def test_load_cases_rejects_duplicate_message_id_across_cases(
    tmp_path, case_payload, message_payload
):
    message = {**message_payload, "msg_id": "duplicate-message"}
    first_case = {**case_payload, "chat_messages": [message]}
    second_case = {**first_case, "case_id": "case-2"}
    path = write_cases(tmp_path, [first_case, second_case])

    with pytest.raises(ValueError, match="duplicate-message"):
        load_cases(path)


@pytest.mark.parametrize("time", ["9:20", "13:00", "0:05", "23:59"])
def test_time_ref_accepts_valid_times(time):
    assert TimeRef(time=time, role="reference").time == time


@pytest.mark.parametrize("time", ["24:00", "9.20", "13:60", "", "к обеду"])
def test_time_ref_rejects_invalid_times(time):
    with pytest.raises(ValidationError):
        TimeRef(time=time, role="reference")


def test_time_ref_rejects_unknown_role():
    with pytest.raises(ValidationError):
        TimeRef(time="9:20", role="unknown")


def test_extraction_rejects_unknown_label():
    with pytest.raises(ValidationError):
        Extraction(summary="Summary", quote="Quote", labels=["unknown"])


def test_parsed_message_accepts_failed_parse_without_extraction():
    message = ParsedMessage(msg_id="x", parse_failed=True)

    assert message.extraction is None
    assert message.parse_failed is True


def test_parsed_message_rejects_success_without_extraction():
    with pytest.raises(ValidationError):
        ParsedMessage(msg_id="x")


def test_parsed_message_rejects_failed_parse_with_extraction():
    extraction = Extraction(summary="Summary", quote="Quote")

    with pytest.raises(ValidationError):
        ParsedMessage(msg_id="x", parse_failed=True, extraction=extraction)


def test_llm_cache_matches_fixture_messages(fixture_cases):
    cache = TypeAdapter(dict[str, ParsedMessage]).validate_json(
        (DATA_DIR / "llm_cache.json").read_text(encoding="utf-8")
    )
    messages = {
        message.msg_id: message.text for case in fixture_cases for message in case.chat_messages
    }
    summaries = {
        "c17-1": (
            "Сотрудник говорит, что вчера отправил фото документов руководителю в личные сообщения"
        ),
        "c24-1": "Сотрудник говорит, что всё обсудил с руководителем",
        "p08-1": "Курьер задерживается, приедет не раньше 13:00",
        "p09-1": "Получатель сообщает, что карты получены и количество проверено",
    }

    assert cache.keys() == messages.keys() == summaries.keys()
    for msg_id, parsed in cache.items():
        assert parsed.msg_id == msg_id
        assert parsed.parse_failed is False
        extraction = parsed.extraction
        assert extraction is not None
        assert extraction.summary == summaries[msg_id]
        assert extraction.quote == messages[msg_id]
        assert extraction.uncertain is False
        if msg_id == "p08-1":
            assert [(time.time, time.role) for time in extraction.times] == [("13:00", "new_eta")]
            assert extraction.labels == ["reports_problem"]
        else:
            assert extraction.times == []
            assert extraction.labels == ["claims_done"]
