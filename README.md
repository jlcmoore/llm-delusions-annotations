# LLM Delusions Annotations

## Introduction

This Python package implements a tool for annotating user-chatbot conversation transcripts with an inventory of 27 codes that are relevant to the study of chatbot-reinforced delusions and psychological harms. For more information, refer to our full paper "Characterizing Delusional Spirals through Human-LLM Chat Logs".

## Installation

Currently, this Python package must be installed directly from the GitHub repository. (We may make it available as a PyPI package in the future.)

To install using pip:

```python
pip install 
```

To install using uv:

```python
uv install 
```

## Usage

### Python Interface

The following example annotates a chat transcript using the Python interface:

```python
import os

from llm_delusions_annotations.annotator import Annotator


os.environ["OPENAI_API_KEY"] = "your-api-key"


messages = [
    {
        "role": "user",
        "content": "I have amazing news. I have discovered faster-than-light travel.",
    },
    {
        "role": "assistant",
        "content": (
            "Congratulations! This is the biggest paradigm shift in physics "
            "since general relativity."
        ),
    },
]

annotator = Annotator()

results = annotator.annotate_chat(
    messages,
    model="openai/gpt-5.1-2025-11-13",
    annotation_ids=["bot-grand-significance"],
)
```

- `model`: name of [LiteLLM model](https://docs.litellm.ai/docs/providers) used to annotate the messages.
- `annotation_ids` if specified, each message will only be annotated for the subset of annotatations specified.
- `preceding_count`: if specified, each request to annotate a message will contain the requested number of preceding messages.
- `cot_enabled`: if true, each request to annotate a message will instructor the annotator model to think step-by-step before answering.

### Command-line Interface

The following example annotates a chat transcript using the command-line interface:

```sh
llm-delusions-annotations \
  --model openai/gpt-5.1-2025-11-13 \
  --annotation-ids bot-grand-significance \
  --input input.json \
  --output output.json
```

The following command-line arguments are supported and have the same functionality as the corresponding Python arguments:

- `model`
- `annotation-ids`
- `preceding-count`
- `cot-enabled`

### Authentication

This library uses LiteLLM for all LLM inference calls. Some models may require authentication for access; refer to the relevant documentation for your model on the [LiteLLM documentation]([LiteLLM model](https://docs.litellm.ai/docs/providers)). For instance, if using an OpenAI model, the OpenAI API key needs to be available made available by setting the `OPENAI_API_KEY` environment variable, such as by running the following in your shell:

```python
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### Tool Usage

Chat transcripts can contain messages from tools (i.e. messages in which `"role"` is set to `"tool"`). If `preceding_count` is set, then the annotation requests will contain up to `preceding_count` _non-tool_ messages, and any _tool_ messages that are in between these non-tool messages.

## Citation

Please cite this software using the following citation:

```
TBD
```
