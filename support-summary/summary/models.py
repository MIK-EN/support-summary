from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CaseType = Literal["onboarding", "offboarding", "card_delivery"]
CrmStatus = Literal["pending", "in_transit", "accepted", "done"]
AuthorRole = Literal["employee", "manager", "courier", "recipient"]
TimeRole = Literal["new_eta", "reference", "request", "other"]
Label = Literal["claims_done", "reports_problem", "asks_action", "info"]


class Confirmations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manager: datetime | None = None
    documents: datetime | None = None
    acceptance: datetime | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    msg_id: str = Field(min_length=1)
    author_role: AuthorRole
    sent_at: datetime
    text: str


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    type: CaseType
    crm_status: CrmStatus
    deadline: datetime | None = None
    confirmations: Confirmations = Confirmations()
    crm_updated_at: datetime
    chat_messages: list[ChatMessage] = []

    @property
    def is_open(self) -> bool:
        return self.crm_status != "done"


class TimeRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: str = Field(pattern=r"^([01]?\d|2[0-3]):[0-5]\d$")
    role: TimeRole


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    quote: str
    times: list[TimeRef] = []
    labels: list[Label] = []
    uncertain: bool = False


class ParsedMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    msg_id: str
    extraction: Extraction | None = None
    parse_failed: bool = False

    @model_validator(mode="after")
    def validate_parse_result(self) -> Self:
        if self.parse_failed != (self.extraction is None):
            raise ValueError("extraction must be absent exactly when parse_failed is true")
        return self
