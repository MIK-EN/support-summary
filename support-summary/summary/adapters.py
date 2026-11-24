import json
from pathlib import Path

from summary.models import Case


def load_cases(path: str | Path) -> list[Case]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array of CRM cases")

    cases = []
    case_ids: set[str] = set()
    message_ids: set[str] = set()
    for item in payload:
        case = Case.model_validate(item)
        if case.case_id in case_ids:
            raise ValueError(f"Duplicate case_id: {case.case_id}")
        case_ids.add(case.case_id)

        for message in case.chat_messages:
            if message.msg_id in message_ids:
                raise ValueError(f"Duplicate msg_id: {message.msg_id}")
            message_ids.add(message.msg_id)

        cases.append(case)
    return cases
