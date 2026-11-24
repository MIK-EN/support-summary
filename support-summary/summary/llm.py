import json
import re
import urllib.request
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from summary.models import CaseType, ChatMessage, Extraction, ParsedMessage

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extract.md"


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class OpenRouterClient:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            content = json.load(response)["choices"][0]["message"]["content"]
        return TypeAdapter(str).validate_python(content, strict=True)


def build_prompt(message: ChatMessage, case_type: CaseType) -> tuple[str, str]:
    system = PROMPT_PATH.read_text(encoding="utf-8")
    user = json.dumps(
        {"case_type": case_type, "author_role": message.author_role, "text": message.text},
        ensure_ascii=False,
    )
    return system, user


def time_in_text(time: str, text: str) -> bool:
    hour, minutes = time.split(":")
    hour_pattern = rf"(?<!\d)0?{int(hour)}"
    suffix = rf"[:.]{minutes}(?!\d)"
    if minutes == "00":
        suffix = rf"(?:{suffix}|(?!\d|[:.]\d))"
    return re.search(hour_pattern + suffix, text) is not None


def validate(raw: str, message: ChatMessage) -> Extraction | None:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        extraction = Extraction.model_validate(payload)
    except (ValueError, RecursionError):
        return None

    extraction.quote = message.text
    times = [ref for ref in extraction.times if time_in_text(ref.time, message.text)]
    if len(times) != len(extraction.times):
        extraction.uncertain = True
    extraction.times = times
    if sum(ref.role == "new_eta" for ref in times) >= 2:
        extraction.uncertain = True
    return extraction


def parse_message(
    message: ChatMessage,
    case_type: CaseType,
    client: LLMClient | None,
    cache: dict[str, ParsedMessage],
    log: list[dict] | None = None,
) -> ParsedMessage:
    msg_id = message.msg_id
    if msg_id in cache:
        return cache[msg_id]
    if client is None:
        return ParsedMessage(msg_id=msg_id, parse_failed=True)

    for attempt in range(1, 3):
        system, user = build_prompt(message, case_type)
        try:
            raw = client.complete(system, user)
        except Exception as exc:
            raw = None
            error = str(exc) or None
            extraction = None
        else:
            error = None
            extraction = validate(raw, message)

        if log is not None:
            log.append(
                {
                    "msg_id": msg_id,
                    "attempt": attempt,
                    "model": client.model,
                    "raw": raw or None,
                    "error": error,
                    "validated": extraction.model_dump(mode="json")
                    if extraction is not None
                    else None,
                }
            )
        if extraction is not None:
            parsed = ParsedMessage(msg_id=msg_id, extraction=extraction)
            cache[msg_id] = parsed
            return parsed

    return ParsedMessage(msg_id=msg_id, parse_failed=True)


def load_cache(path: str | Path) -> dict[str, ParsedMessage]:
    try:
        data = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    return TypeAdapter(dict[str, ParsedMessage]).validate_json(data)


def save_cache(path: str | Path, cache: dict[str, ParsedMessage]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {msg_id: parsed.model_dump(mode="json") for msg_id, parsed in cache.items()}
    path.write_text(
        json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8"
    )
