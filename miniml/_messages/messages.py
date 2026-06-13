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
# Message Stage / Task Tags
# ==========================================================

MessageStage = Literal[
    "generation",
    "tool_execution",
    "validation",
    "proc_end_validation",
    "tool_exec_validation",
    "fallback",
]

MessageTask = Literal[
    "tool_call",
    "tool_exec_validation_failure",
    "proc_end_validation_failure",
    "tool_execution",
    "error",
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
# NEW: Tagged Message
#
# Carries an explicit stage + task label so history can be
# filtered precisely without scanning content strings.
# All fields beyond role/content default to None so existing
# code that constructs AgentMessage / ValidatorMessage is
# unaffected — only new code opts into tagging.
# ==========================================================

@dataclass(slots=True)
class TaggedMessage(BaseMessage):
    """
    A message that carries an explicit stage and task label.

    Use this instead of AgentMessage / ValidatorMessage when you
    need to retrieve a specific slice of history later, e.g.:

        history.get_by_stage("tool_exec_validation")
        history.get_by_task("proc_end_validation_failure")
        history.get_by_stage_and_task("validation", "tool_exec_validation_failure")

    Parameters
    ----------
    role    : inherited from BaseMessage
    content : inherited from BaseMessage
    stage   : broad phase the message belongs to (e.g. "tool_execution")
    task    : fine-grained action within that phase (e.g. "tool_call")
    """
    stage: Optional[MessageStage] = None
    task:  Optional[MessageTask]  = None


# ==========================================================
# Message Union
# ==========================================================

MessageType = Union[
    InferenceMessage,
    ValidatorMessage,
    AgentMessage,
    TaggedMessage,       # NEW
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
    # Get Stringified History
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

    # ======================================================
    # NEW: Get All Tagged Messages
    # ======================================================

    def get_all_tagged(self) -> List[TaggedMessage]:
        """Return every TaggedMessage in history, in insertion order."""
        return [
            message
            for message in self.history
            if isinstance(message, TaggedMessage)
        ]

    # ======================================================
    # NEW: Get Tagged Messages By Stage
    # ======================================================

    def get_by_stage(
        self,
        stage: MessageStage,
    ) -> List[TaggedMessage]:
        """
        Return all TaggedMessages whose stage matches.

        Typical use — feed only execution records to the
        proc-end validator:

            exec_records = history.get_by_stage("tool_execution")
        """
        return [
            message
            for message in self.history
            if isinstance(message, TaggedMessage) and message.stage == stage
        ]

    # ======================================================
    # NEW: Get Tagged Messages By Task
    # ======================================================

    def get_by_task(
        self,
        task: MessageTask,
    ) -> List[TaggedMessage]:
        """
        Return all TaggedMessages whose task matches.

        Typical use — feed only validation failures to the
        tool-exec retry validator:

            failures = history.get_by_task("tool_exec_validation_failure")
        """
        return [
            message
            for message in self.history
            if isinstance(message, TaggedMessage) and message.task == task
        ]

    # ======================================================
    # NEW: Get Tagged Messages By Stage AND Task
    # ======================================================

    def get_by_stage_and_task(
        self,
        stage: MessageStage,
        task: MessageTask,
    ) -> List[TaggedMessage]:
        """
        Return all TaggedMessages matching both stage and task.

        Typical use — precise slice for a specific sub-phase:

            records = history.get_by_stage_and_task(
                "validation", "proc_end_validation_failure"
            )
        """
        return [
            message
            for message in self.history
            if isinstance(message, TaggedMessage)
            and message.stage == stage
            and message.task == task
        ]

    # ======================================================
    # NEW: Get Stringified Slice (Tagged, filtered)
    # ======================================================

    def get_tagged_string(
        self,
        unique: bool = False,
        stage: Optional[MessageStage] = None,
        task: Optional[MessageTask] = None,
    ) -> str:
        """
        Return a formatted string of TaggedMessages filtered by
        stage and/or task. If neither is given, returns all tagged
        messages as a string.

        Designed as a drop-in replacement for get_history_string()
        when you need a clean, focused context block.
        """
        messages: List[TaggedMessage] = [
            message
            for message in self.history
            if isinstance(message, TaggedMessage)
            and (stage is None or message.stage == stage)
            and (task is None or message.task == task)
        ]

        if unique:
            seen = set()
            unique_messages = []

            for msg in messages:
                key = (
                    msg.role,
                    msg.content,
                    msg.stage,
                    msg.task,
                )

                if key not in seen:
                    seen.add(key)
                    unique_messages.append(msg)

            messages = unique_messages

        return "\n".join(
            f"[{m.role}] {m.content}"
            for m in messages
        )