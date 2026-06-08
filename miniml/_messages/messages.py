# messages.py

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


# ==========================================================
# Message Types
# ==========================================================

MessageRole = Literal[
    "system",
    "user",
    "agent",
    "assistant",
    "inference",
    "validator",
]


# ==========================================================
# Base Message
# ==========================================================

@dataclass(slots=True)
class BaseMessage:
    role: MessageRole
    content: str


# ==========================================================
# Specialized Messages
# ==========================================================

@dataclass(slots=True)
class InferenceMessage(BaseMessage):
    pass


@dataclass(slots=True)
class ValidatorMessage(BaseMessage):
    pass


@dataclass(slots=True)
class AgentMessage(BaseMessage):
    pass


# ==========================================================
# Message Union
# ==========================================================

MessageType = Union[
    InferenceMessage,
    ValidatorMessage,
    AgentMessage,
]


# ==========================================================
# Message History
# ==========================================================

@dataclass(slots=True)
class MessageHistory:

    history: List[MessageType] = field(default_factory=list)  # type: ignore

    # ======================================================
    # Add Message
    # ======================================================

    def add_message(
        self,
        message: MessageType,
    ) -> None:

        self.history.append(message)

    # ======================================================
    # Get Full History
    # ======================================================

    def get_history(self) -> List[MessageType]:

        return self.history
    

    # =====================================================
    # Get Strigified History
    # ======================================================

    def get_history_string(self) -> str:

        return "\n".join([f"[{message.role}] {message.content}" for message in self.history])
    

    # =====================================================
    # Get last history
    # =====================================================

    def get_last_message_string(self) -> Optional[str]:

        if not self.history:
            return None
        
        _last_msg = self.history[-1]
        return f"[{_last_msg.role}] {_last_msg.content}"

    # ======================================================
    # Get First Message
    # ======================================================

    def get_first(self) -> Optional[MessageType]:

        if not self.history:
            return None

        return self.history[0]

    # ======================================================
    # Get Last Message
    # ======================================================

    def get_last(self) -> Optional[MessageType]:

        if not self.history:
            return None

        return self.history[-1]

    # ======================================================
    # Get First N Messages
    # ======================================================

    def get_first_n(
        self,
        n: int,
    ) -> List[MessageType]:

        return self.history[:n]

    # ======================================================
    # Get Last N Messages
    # ======================================================

    def get_last_n(
        self,
        n: int,
    ) -> List[MessageType]:

        return self.history[-n:]

    # ======================================================
    # Get Messages By Role
    # ======================================================

    def get_by_role(
        self,
        role: MessageRole,
    ) -> List[MessageType]:

        return [
            message
            for message in self.history
            if message.role == role
        ]

    # ======================================================
    # Get All Inference Messages
    # ======================================================

    def get_all_inference(
        self,
    ) -> List[InferenceMessage]:

        return [
            message
            for message in self.history
            if isinstance(message, InferenceMessage)
        ]

    # ======================================================
    # Get All Validator Messages
    # ======================================================

    def get_all_validator(
        self,
    ) -> List[ValidatorMessage]:

        return [
            message
            for message in self.history
            if isinstance(message, ValidatorMessage)
        ]

    # ======================================================
    # Clear History
    # ======================================================

    def clear(self) -> None:

        self.history.clear()

    # ======================================================
    # Length
    # ======================================================

    def __len__(self) -> int:

        return len(self.history)

    # ======================================================
    # Iteration
    # ======================================================

    def __iter__(self):

        return iter(self.history)

    # ======================================================
    # String Representation
    # ======================================================

    def __str__(self) -> str:

        return "\n".join(
            [
                f"[{message.role}] {message.content}"
                for message in self.history
            ]
        )