"""High-level API for message annotation"""

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Union

from llm_delusions_annotations.annotation_prompts import build_prompt
from llm_delusions_annotations.chat.chat_io import Chat, load_chats_for_file
from llm_delusions_annotations.chat.chat_utils import iter_chat_messages
from llm_delusions_annotations.classify_messages import (
    ClassifyResult,
    build_completion_messages,
    make_classify_requests,
    to_litellm_messages,
)
from llm_delusions_annotations.configs import load_annotation_configs

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_WORKERS = 32


@dataclass(frozen=True)
class AnnotatableMessage:
    """Message for annotation, which could be part of a longer conversation"""

    content: str
    """Text content of the message"""

    role: str
    """Role of the message creator"""

    preceding_messages: Optional[Dict] = None
    """Preceding messages as dicts with `role` and `content` keys"""


@dataclass(frozen=True)
class _AnnotatorTask:
    """Annotation task for the annotator, consisting of a single message and annotations"""

    message_index: int
    """Index of the message if it was part of a conversation"""

    message: AnnotatableMessage
    """Message to annotate"""

    annotation_id: str
    """ID of the annotation to be run"""


@dataclass(frozen=True)
class ChatWithAnnotations:
    chat: Chat
    annotations: Sequence[Dict[str, ClassifyResult]]


def _to_annotatable_message(
    message: Union[AnnotatableMessage, Dict]
) -> AnnotatableMessage:
    """Given a dict or AnnotatableMessage, return a normalized AnnotatableMessage.

    Convenience function to allow users to specify inputs as dicts."""
    if isinstance(message, dict):
        return AnnotatableMessage(**message)
    return message


def build_annotation_request(
    message: Union[AnnotatableMessage, Dict],
    annotation_id: str,
    *,
    cot_enabled: bool = False,
):
    """Return the request to be sent to the model for annotating a single message
    with a single annotation."""
    annotation_configs = load_annotation_configs([annotation_id])
    assert len(annotation_configs) == 1
    annotation_config = annotation_configs[0]
    message = _to_annotatable_message(message)
    if message.role not in annotation_config.allowed_roles:
        raise ValueError(
            f"Message role '{message.role}' is not an allowed role for the annotation '{annotation_config.spec['id']}'; allowed roles are {annotation_config.allowed_roles}"
        )
    prompt_text = build_prompt(
        annotation=annotation_config.spec,
        message_text=message.content,
        role=message.role,
        context_messages=(
            message.preceding_messages
            if message.preceding_messages is not None
            else None
        ),
        include_cot_addendum=cot_enabled,
    )
    return to_litellm_messages(build_completion_messages(prompt_text))


def chat_message_iterator(
    chat_messages: List[Dict], *, preceding_count: int = 0
) -> Iterator[AnnotatableMessage]:
    """Iterate over annotatable message within a chat

    The number of annotatable messages may be fewer than the number
    of chat messages, because some messages may be filtered out for having
    a non-user and non-assistant role.

    If `preceding_count` is set, each output will contain the requested
    numer of preceding message."""
    for message_index, message_context_data in enumerate(
        iter_chat_messages(
            chat_messages,
            allowed_roles=None,
            reverse_conversations=False,
            preceding_count=preceding_count,
        )
    ):
        assert message_index == message_context_data.message_index
        yield AnnotatableMessage(
            content=message_context_data.content,
            role=message_context_data.role,
            preceding_messages=message_context_data.preceding,
        )


class Annotator:
    """High-level interface for annotating chats"""

    def __init__(
        self, *, timeout: int = DEFAULT_TIMEOUT, max_workers: int = DEFAULT_MAX_WORKERS
    ):
        self.timeout = timeout
        self.max_workers = max_workers

    def annotate_chat(
        self,
        chat_messages: Sequence[AnnotatableMessage],
        model: str,
        annotation_ids: List[str] = None,
        *,
        preceding_count: int = 0,
        cot_enabled: bool = False,
    ) -> Sequence[Dict[str, ClassifyResult]]:
        """Return annotations for the chat."""
        messages = list(
            chat_message_iterator(chat_messages, preceding_count=preceding_count)
        )
        return self.annotate_messages(
            messages,
            model=model,
            annotation_ids=annotation_ids,
            cot_enabled=cot_enabled,
        )

    def annotate_messages(
        self,
        messages: Sequence[AnnotatableMessage],
        model: str,
        annotation_ids: List[str] = None,
        *,
        cot_enabled: bool = False,
    ) -> Sequence[Dict[str, ClassifyResult]]:
        """Return annotations for messages.

        If the messages are part of a single chat, and the annotations of each message
        should use its the preceding context, use `annotate_chat()` instead."""
        annotation_configs = load_annotation_configs(annotation_ids)
        tasks = [
            _AnnotatorTask(message_index, message, annotation_config.spec["id"])
            for message_index, message in enumerate(messages)
            for annotation_config in annotation_configs
            if message.role in annotation_config.allowed_roles
        ]
        requests = [
            build_annotation_request(
                task.message, task.annotation_id, cot_enabled=cot_enabled
            )
            for task in tasks
        ]
        classify_results = make_classify_requests(
            requests, model=model, timeout=self.timeout, max_workers=self.max_workers
        )

        assert len(classify_results) == len(tasks)
        results: List[Dict] = [{} for _ in messages]
        for task, classify_result in zip(tasks, classify_results):
            results[task.message_index][task.annotation_id] = classify_result
        return results

    def annotate_message(
        self,
        message: Union[AnnotatableMessage, Dict],
        model: str,
        annotation_ids: List[str] = None,
        *,
        cot_enabled: bool = False,
    ) -> Dict[str, ClassifyResult]:
        """Return annotations for a single message."""
        message = _to_annotatable_message(message)
        results = self.annotate_messages(
            [message],
            model=model,
            annotation_ids=annotation_ids,
            cot_enabled=cot_enabled,
        )
        assert len(results) == 1
        return results[0]

    def annotate_chats_in_file(
        self,
        path: str,
        model: str,
        annotation_ids: List[str] = None,
        *,
        preceding_count: int = 0,
        cot_enabled: bool = False,
    ) -> Iterator[ChatWithAnnotations]:
        chats = load_chats_for_file(pathlib.Path(path))
        for chat in chats:
            annotations = self.annotate_chat(
                chat.messages,
                model,
                annotation_ids,
                preceding_count=preceding_count,
                cot_enabled=cot_enabled,
            )
            yield ChatWithAnnotations(chat, annotations)
