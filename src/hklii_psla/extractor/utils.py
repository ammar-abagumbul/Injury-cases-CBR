from typing import Any
from hklii_psla.extractor.base import TokenCount

from pydantic import BaseModel

def _extract_raw_text(raw_msg: Any) -> str:
    """Extract textual content from a LangChain message."""
    if raw_msg is None:
        return ""

    content = getattr(raw_msg, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    texts.append(str(text))
            else:
                texts.append(str(block))

        return "\n".join(texts)

    return str(content) if content is not None else ""


def extract_token_counts(result: dict) -> TokenCount:
    """Extract token counts from a ``with_structured_output(..., include_raw=True)`` result.

    Parameters
    ----------
    result : dict
        The dict returned by ``structured_model.invoke()`` when using
        ``include_raw=True``. Must contain a ``"raw"`` key with an AIMessage.

    Returns
    -------
    TokenCount
        Input, output, and reasoning token counts.
    """
    raw_msg = result["raw"]
    usage = raw_msg.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Reasoning tokens (DeepSeek and similar providers)
    output_details = usage.get("output_token_details", {})
    reasoning_tokens = output_details.get("reasoning", 0)

    return TokenCount(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


# ---------------------------------------------------------------------------
# Error handling helpers
# ---------------------------------------------------------------------------


def _truncate(value: str, limit: int = 4000) -> str:
    """Truncate a long string for inclusion in an error message."""
    if len(value) <= limit:
        return value
    return value[:limit] + "\n\n[... TRUNCATED ...]"


def _format_invalid_tool_calls(raw: Any) -> str | None:
    """
    Extract and format malformed tool calls from an AIMessage.

    This is the important failure mode when using:
        with_structured_output(..., include_raw=True)

    LangChain may put malformed tool calls in `invalid_tool_calls`
    instead of raising an exception from the invoke() call.
    """
    invalid_tool_calls = getattr(raw, "invalid_tool_calls", None)

    if not invalid_tool_calls:
        return None

    parts = []

    for i, call in enumerate(invalid_tool_calls):
        name = call.get("name", "<unknown>")
        call_id = call.get("id", "<unknown>")
        args = call.get("args", "")
        error = call.get("error")

        parts.append(
            f"Invalid tool call [{i}]\n"
            f"  name: {name}\n"
            f"  id: {call_id}\n"
            f"  error: {error or 'No error supplied'}\n"
            # f"  arguments:\n{_truncate(str(args))}"
        )

    return "\n\n".join(parts)


def build_structured_output_error(result: dict[str, Any], verbose=False) -> str:
    """
    Build a useful error message when `include_raw=True` returned
    parsed=None.

    There are two particularly important cases:

    1. The model generated an invalid tool call / invalid JSON.
    2. The model returned something that could not be parsed into Case.
    """
    raw = result.get("raw")
    parsing_error = result.get("parsing_error")

    parts = ["Structured extraction failed."]

    # Case 1: malformed tool call
    invalid_tool_calls = _format_invalid_tool_calls(raw)

    if invalid_tool_calls:
        parts.append(
            "\n"
            "Failure type: INVALID_STRUCTURED_OUTPUT\n"
            "The LLM attempted to call the structured-output tool, but "
            "the generated arguments were not valid JSON.\n\n"
            f"{invalid_tool_calls if verbose else ''}"
        )

    # Case 2: LangChain reported a parsing error
    if parsing_error:
        parts.append(
            "\n"
            "Parsing error:\n"
            f"{parsing_error if verbose else ''}"
        )

    # Raw message metadata
    if raw is not None:
        response_metadata = getattr(raw, "response_metadata", None)

        if response_metadata:
            model = response_metadata.get("model_name")
            finish_reason = response_metadata.get("finish_reason")

            metadata_parts = []

            if model:
                metadata_parts.append(f"model={model}")

            if finish_reason:
                metadata_parts.append(f"finish_reason={finish_reason}")

            if metadata_parts:
                parts.append(
                    "\n"
                    f"LLM response: {', '.join(metadata_parts)}"
                )

    # Raw textual content
    raw_text = _extract_raw_text(raw)

    if raw_text and raw_text.strip() and verbose:
        parts.append(
            "\n"
            "Raw LLM content:\n"
            f"{_truncate(raw_text)}"
        )

    return "\n".join(parts)


def get_sub_schema(field_name: str, full_model: BaseModel | type[BaseModel]) -> type[BaseModel]:
    model = full_model if isinstance(full_model, type) else type(full_model)
    sub_model = model.model_fields[field_name].annotation
    if isinstance(sub_model, type) and issubclass(sub_model, BaseModel):
        return sub_model

    raise KeyError(f"Field '{field_name}' not found in schema")
