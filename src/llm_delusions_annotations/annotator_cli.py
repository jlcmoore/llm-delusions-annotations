"""Command-line interface for running the annotator over transcript files."""

import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from llm_delusions_annotations.annotation_prompts import (
    ANNOTATIONS,
    add_preceding_context_argument,
)
from llm_delusions_annotations.annotator import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_TIMEOUT,
    Annotator,
    ChatWithAnnotations,
)
from llm_delusions_annotations.classify_messages import ClassifyResult


@dataclass(frozen=True)
class AnnotatorRunConfig:
    """Configuration for a single annotator run."""

    model: str
    annotation_ids: Optional[List[str]]
    preceding_count: int
    timeout: int
    max_workers: int


def _message_annotations_to_dict(annotations: Dict[str, ClassifyResult]) -> Dict:
    """Convert annotation results into a JSON-serializable dictionary."""

    return {
        annotation_id: dataclasses.asdict(classify_result)
        for annotation_id, classify_result in annotations.items()
    }


def _annotated_chat_to_dict(chat_with_annotations: ChatWithAnnotations):
    """Convert annotated chat data into a JSON-serializable dictionary."""

    result = dataclasses.asdict(chat_with_annotations)
    result["messages"] = [
        {
            "role": message["role"],
            "content": message["content"],
            "annotations": _message_annotations_to_dict(message_annotations),
        }
        for message, message_annotations in zip(
            chat_with_annotations.chat.messages, chat_with_annotations.annotations
        )
    ]
    return result


def annotate_chats_in_file_and_write(
    input_path: str,
    output_path: str,
    config: AnnotatorRunConfig,
):
    """Annotate chats in a file and write the results to disk."""

    annotator = Annotator(timeout=config.timeout, max_workers=config.max_workers)
    output_dicts: List[Dict] = []
    for chat_with_annotation in annotator.annotate_chats_in_file(
        path=input_path,
        model=config.model,
        annotation_ids=config.annotation_ids,
        preceding_count=config.preceding_count,
    ):
        output_dicts.append(_annotated_chat_to_dict(chat_with_annotation))
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(output_dicts, output_file, indent=2)


DEFAULT_MODEL = "openai/gpt-5.1"


def main():
    """Run the annotator CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=("Model name understood by LiteLLM " f"(default: {DEFAULT_MODEL})."),
    )
    annotation_ids = [spec["id"] for spec in ANNOTATIONS]
    parser.add_argument(
        "--annotation-ids",
        "-a",
        action="append",
        choices=annotation_ids,
        help=(
            "Annotation ID to use for classification (repeatable). "
            "Defaults to all annotations except those with category 'test' when "
            "omitted."
        ),
    )
    add_preceding_context_argument(parser)
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Request timeout for the LiteLLM API (seconds).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "Maximum worker threads used by the LiteLLM batch_completion "
            f"API (default: {DEFAULT_MAX_WORKERS}). Lower this to reduce concurrency "
            "when hitting provider rate limits."
        ),
    )
    args = parser.parse_args()
    config = AnnotatorRunConfig(
        model=args.model,
        annotation_ids=args.annotation_ids,
        preceding_count=args.preceding_context,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )
    annotate_chats_in_file_and_write(
        input_path=args.input,
        output_path=args.output,
        config=config,
    )


if __name__ == "__main__":
    main()
