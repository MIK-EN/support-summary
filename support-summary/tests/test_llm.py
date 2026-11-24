import json
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest

from summary.models import ChatMessage, Extraction, ParsedMessage, TimeRef

# isort: split
from summary.llm import (
    PROMPT_PATH,
    OpenRouterClient,
    build_prompt,
    load_cache,
    parse_message,
    save_cache,
    time_in_text,
    validate,
)

DATA_DIR = Path(__file__).parent.parent / "data"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class FakeClient:
    model = "test-model"

    def __init__(self, *answers: str | RuntimeError) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        index = len(self.calls)
        self.calls.append((system, user))
        answer = self.answers[index]
        if isinstance(answer, RuntimeError):
            raise answer
        return answer


@pytest.fixture
def message() -> ChatMessage:
    return ChatMessage(
        msg_id="сообщение-1",
        author_role="courier",
        sent_at=datetime(2026, 11, 23, 10),
        text=(
            "Здрасте, я по заказу на 9:20, я в пробке, в 10:15 приеду, "
            "можете перенести клиента на 11"
        ),
    )


@pytest.fixture
def extraction(message: ChatMessage) -> Extraction:
    return Extraction(
        summary="Курьер в пробке, приедет в 10:15 и просит перенести клиента на 11:00",
        quote=message.text,
        times=[
            TimeRef(time="9:20", role="reference"),
            TimeRef(time="10:15", role="new_eta"),
            TimeRef(time="11:00", role="request"),
        ],
        labels=["reports_problem", "asks_action"],
        uncertain=False,
    )


@pytest.mark.parametrize(
    ("time", "text", "expected"),
    [
        ("13:00", "раньше 13:00 не приеду", True),
        ("11:00", "перенесите на 11", True),
        ("11:00", "перенесите на 11.", True),
        ("9:20", "на 09:20", True),
        ("9:20", "на 9.20", True),
        ("10:00", "в 10:15 приеду", False),
        ("1:00", "в 11 буду", False),
        ("12:30", "к обеду", False),
        ("13:00", "", False),
    ],
)
def test_time_in_text(time: str, text: str, expected: bool) -> None:
    assert time_in_text(time, text) is expected


def test_validate_keeps_valid_extraction(message: ChatMessage, extraction: Extraction) -> None:
    assert validate(extraction.model_dump_json(), message) == extraction


def test_validate_restores_verbatim_quote(message: ChatMessage, extraction: Extraction) -> None:
    message = message.model_copy(update={"text": f"  {message.text}\n"})
    result = validate(extraction.model_dump_json(), message)

    assert result == extraction.model_copy(update={"quote": message.text})


def test_validate_drops_time_absent_from_text(message: ChatMessage, extraction: Extraction) -> None:
    valid_time = TimeRef(time="10:15", role="new_eta")
    absent_time = TimeRef(time="14:00", role="new_eta")
    raw = extraction.model_copy(update={"times": [absent_time, valid_time]}).model_dump_json()
    result = validate(raw, message)

    assert result == extraction.model_copy(update={"times": [valid_time], "uncertain": True})


def test_validate_marks_two_remaining_etas_uncertain(message: ChatMessage) -> None:
    message = message.model_copy(update={"text": "Буду в 12, хотя нет, скорее в 13"})
    extraction = Extraction(
        summary="Курьер называет два времени прибытия",
        quote=message.text,
        times=[TimeRef(time="12:00", role="new_eta"), TimeRef(time="13:00", role="new_eta")],
        uncertain=False,
    )

    assert validate(extraction.model_dump_json(), message) == extraction.model_copy(
        update={"uncertain": True}
    )


@pytest.mark.parametrize("raw", ["not json", "[]", '{"summary": 1}'])
def test_validate_rejects_invalid_json_or_object(raw: str, message: ChatMessage) -> None:
    assert validate(raw, message) is None


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"labels": ["unknown"]}, id="unknown-label"),
        pytest.param({"times": [{"time": "к обеду", "role": "new_eta"}]}, id="bad-time"),
        pytest.param({"extra": True}, id="extra-field"),
    ],
)
def test_validate_rejects_invalid_fields(
    override: dict[str, object], message: ChatMessage, extraction: Extraction
) -> None:
    payload = extraction.model_dump(mode="json")
    payload.update(override)

    assert validate(json.dumps(payload), message) is None


def test_validate_rejects_missing_quote(message: ChatMessage, extraction: Extraction) -> None:
    payload = extraction.model_dump(mode="json", exclude={"quote"})

    assert validate(json.dumps(payload), message) is None


def test_parse_message_returns_cached_object(message: ChatMessage, extraction: Extraction) -> None:
    cached = ParsedMessage(msg_id=message.msg_id, extraction=extraction)
    cache = {message.msg_id: cached}
    client = FakeClient()

    assert parse_message(message, "card_delivery", client, cache) is cached
    assert client.calls == []


def test_parse_message_without_client_leaves_cache_unchanged(
    message: ChatMessage, extraction: Extraction
) -> None:
    existing = ParsedMessage(msg_id="existing", extraction=extraction)
    cache = {"existing": existing}

    result = parse_message(message, "card_delivery", None, cache)

    assert result == ParsedMessage(msg_id=message.msg_id, parse_failed=True)
    assert cache == {"existing": existing}


def test_parse_message_retries_invalid_answer_and_logs_attempts(
    message: ChatMessage, extraction: Extraction
) -> None:
    raw = extraction.model_dump_json()
    client = FakeClient("not json", raw)
    cache: dict[str, ParsedMessage] = {}
    log: list[dict] = []

    result = parse_message(message, "card_delivery", client, cache, log)

    assert result == ParsedMessage(msg_id=message.msg_id, extraction=extraction)
    assert cache == {message.msg_id: result}
    assert len(client.calls) == 2
    assert log == [
        {
            "msg_id": message.msg_id,
            "attempt": 1,
            "model": client.model,
            "raw": "not json",
            "error": None,
            "validated": None,
        },
        {
            "msg_id": message.msg_id,
            "attempt": 2,
            "model": client.model,
            "raw": raw,
            "error": None,
            "validated": extraction.model_dump(mode="json"),
        },
    ]


def test_parse_message_does_not_cache_two_failed_attempts(message: ChatMessage) -> None:
    client = FakeClient("not json", "[]")
    cache: dict[str, ParsedMessage] = {}

    result = parse_message(message, "card_delivery", client, cache)

    assert result == ParsedMessage(msg_id=message.msg_id, parse_failed=True)
    assert len(client.calls) == 2
    assert message.msg_id not in cache


def test_parse_message_retries_and_logs_client_exception(
    message: ChatMessage, extraction: Extraction
) -> None:
    client = FakeClient(RuntimeError("gateway unavailable"), extraction.model_dump_json())
    cache: dict[str, ParsedMessage] = {}
    log: list[dict] = []

    result = parse_message(message, "card_delivery", client, cache, log)

    assert result == ParsedMessage(msg_id=message.msg_id, extraction=extraction)
    assert len(client.calls) == len(log) == 2
    assert "gateway unavailable" in log[0]["error"]
    assert log[0]["raw"] is None
    assert log[0]["validated"] is None


def test_build_prompt_uses_file_and_only_chat_fields(message: ChatMessage) -> None:
    system, user = build_prompt(message, "card_delivery")

    assert system == PROMPT_PATH.read_text(encoding="utf-8")
    assert json.loads(user) == {
        "case_type": "card_delivery",
        "author_role": message.author_role,
        "text": message.text,
    }
    assert message.text in user


def test_prompt_file_contains_extraction_terms() -> None:
    assert PROMPT_PATH.is_file()
    system = PROMPT_PATH.read_text(encoding="utf-8")

    for term in ("new_eta", "claims_done", "uncertain", "9:20"):
        assert term in system
    assert "CRM" not in system.upper()
    assert len(system.splitlines()) <= 80


def test_openrouter_complete_sends_json_request(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[urllib.request.Request] = []
    content = '{"summary": "Курьер задерживается"}'
    response = {"choices": [{"message": {"content": content}}]}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> BytesIO:
        requests.append(request)
        assert timeout == 12.0
        return BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="k", model="test-model", timeout=12.0)

    assert client.complete("system prompt", "user prompt") == content

    [request] = requests
    assert request.full_url == OpenRouterClient.URL == OPENROUTER_URL
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer k"
    assert request.get_header("Content-type") == "application/json"
    assert request.data is not None
    assert json.loads(request.data) == {
        "model": "test-model",
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ],
    }


def test_openrouter_complete_raises_for_bad_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> BytesIO:
        requests.append(request)
        return BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="k", model="test-model")

    # The client contract allows any exception type for an invalid response shape.
    with pytest.raises(Exception):  # noqa: B017
        client.complete("system prompt", "user prompt")

    assert len(requests) == 1


def test_cache_round_trip_sorts_keys_and_preserves_cyrillic(
    tmp_path: Path, extraction: Extraction
) -> None:
    path = tmp_path / "nested" / "llm_cache.json"
    cache = {
        "я-2": ParsedMessage(msg_id="я-2", parse_failed=True),
        "а-1": ParsedMessage(msg_id="а-1", extraction=extraction),
    }

    save_cache(path, cache)

    serialized = path.read_text(encoding="utf-8")
    assert list(json.loads(serialized)) == sorted(cache)
    assert '  "а-1": {' in serialized
    assert extraction.summary in serialized
    assert load_cache(path) == cache


def test_load_cache_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    assert load_cache(tmp_path / "missing.json") == {}


def test_load_cache_reads_all_fixture_entries() -> None:
    cache = load_cache(DATA_DIR / "llm_cache.json")

    assert set(cache) == {"c17-1", "c24-1", "p08-1", "p09-1"}
    for msg_id, entry in cache.items():
        assert isinstance(entry, ParsedMessage)
        assert entry.msg_id == msg_id
        assert entry.parse_failed is False
        assert isinstance(entry.extraction, Extraction)
