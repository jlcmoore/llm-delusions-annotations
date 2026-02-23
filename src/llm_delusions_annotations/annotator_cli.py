import argparse
import dataclasses
import json
from typing import Dict, List

from llm_delusions_annotations.annotation_prompts import (
    ANNOTATIONS,
    THINK_CLOSE_TAG,
    THINK_OPEN_TAG,
    add_preceding_context_argument,
)
from llm_delusions_annotations.annotator import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_TIMEOUT,
    Annotator,
    ChatWithAnnotations,
)
from llm_delusions_annotations.classify_messages import ClassifyResult


def _message_annotations_to_dict(annotations: Dict[str, ClassifyResult]) -> Dict:
    return {
        annotation_id: dataclasses.asdict(classify_result)
        for annotation_id, classify_result in annotations.items()
    }


def _annotated_chat_to_dict(chat_with_annotations: ChatWithAnnotations):
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
    model: str,
    annotation_ids: List[str] = None,
    *,
    preceding_count: int = 0,
    cot_enabled: bool = False,
    timeout: int = 30,
    max_workers: int = 32,
):
    annotator = Annotator(timeout=timeout, max_workers=max_workers)
    output_dicts: List[Dict] = []
    for chat_with_annotation in annotator.annotate_chats_in_file(
        path=input_path,
        model=model,
        annotation_ids=annotation_ids,
        preceding_count=preceding_count,
        cot_enabled=cot_enabled,
    ):
        output_dicts.append(_annotated_chat_to_dict(chat_with_annotation))
    with open(output_path, "w") as output_file:
        json.dump(output_dicts, output_file, indent=2)


DEFAULT_MODEL = "openai/gpt-5.1"


def main():
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
        "--annotation",
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
        "--cot",
        action="store_true",
        help=(
            "Allow the model to use a chain-of-thought scratchpad wrapped in "
            f"'{THINK_OPEN_TAG}' and '{THINK_CLOSE_TAG}' tags before the final "
            "JSON answer."
        ),
    )
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
    annotate_chats_in_file_and_write(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        annotation_ids=args.annotation,
        preceding_count=args.preceding_context,
        cot_enabled=args.cot,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
