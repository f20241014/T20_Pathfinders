"""Message schema and counter tracking for negotiation and allocation overhead evaluation."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CoordinationMessage:
    msg_type: str
    sender_id: str
    recipient_id: str
    payload: Dict[str, Any]


@dataclass
class MessageCounter:
    requests: int = 0
    responses: int = 0
    results: int = 0
    allocations: int = 0

    def log_message(self, msg_type: str) -> None:
        if msg_type == "negotiation_request":
            self.requests += 1
        elif msg_type == "negotiation_response":
            self.responses += 1
        elif msg_type == "negotiation_result":
            self.results += 1
        elif msg_type == "allocation":
            self.allocations += 1

    @property
    def total_negotiation_messages(self) -> int:
        return self.requests + self.responses + self.results

    @property
    def total_messages(self) -> int:
        return self.requests + self.responses + self.results + self.allocations
