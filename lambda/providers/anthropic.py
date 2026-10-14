"""Anthropic menu transcription: Haiku Messages requests and structured response parsing."""

import asyncio
import json

import anthropic as anthropic_sdk
from anthropic.types import ContentBlockParam

from .shared import (
    ApiError,
    MenuImage,
    MENU_TRANSCRIPTION_PROMPT,
    MENU_TRANSCRIPTION_SCHEMA,
    get_api_key,
    get_request_timeout,
    run_with_deadline,
    validate_menu,
)


DEFAULT_MODEL = "claude-haiku-5-5"
TIMEOUT_SECONDS = 20.0
SYSTEM_PROMPT = MENU_TRANSCRIPTION_PROMPT
# SYSTEM_PROMPT = (
#     MENU_TRANSCRIPTION_PROMPT + "\nPreserve reading order, prices, descriptions and dietary notes. "
#     "Use a descriptive restaurant placeholder if its name is absent. "
#     "Mark uncertain text [illegible]. If no menu is readable, report no_menu or illegible_text."
# )


def transcribe_menu(images: list[MenuImage], context):
    """Transcribe images using the configured Anthropic model."""
    api_key = get_api_key("ANTHROPIC_API_KEY")
    timeout = get_request_timeout(context, provider_name="Anthropic", max_seconds=TIMEOUT_SECONDS)
    content: list[ContentBlockParam] = [{"type": "text", "text": "Transcribe these menu images."}]
    content.extend({"type": "image", "source": {
        "type": "base64", "media_type": image.media_type, "data": image.base64_data,
    }} for image in images)
    try:
        message = asyncio.run(run_with_deadline(_request_message(api_key, content, timeout), timeout))
        print(f"Anthropic response: {message}")
    except (anthropic_sdk.APITimeoutError, asyncio.TimeoutError):
        raise ApiError(504, "anthropic_timeout", "Menu transcription timed out. Try fewer or smaller images.") from None
    except anthropic_sdk.APIStatusError as error:
        raise map_api_error(error) from None
    except anthropic_sdk.APIConnectionError:
        raise ApiError(502, "anthropic_unavailable", "Could not connect to Anthropic. Try again later.") from None
    except anthropic_sdk.AnthropicError:
        raise ApiError(502, "anthropic_failure", "Anthropic could not complete menu transcription.") from None
    except (ValueError, TypeError):
        raise ApiError(502, "invalid_anthropic_response", "Anthropic returned an invalid response.") from None
    return parse_response(message)


async def _request_message(api_key, content, timeout):
    async with anthropic_sdk.AsyncAnthropic(api_key=api_key, base_url="https://api.anthropic.com",
                                          timeout=timeout, max_retries=0) as client:
        return await client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "low","format": {"type": "json_schema", "schema": MENU_TRANSCRIPTION_SCHEMA}},
            thinking={"type": "disabled"},
        )


def parse_response(message):
    """Read menu text only, skipping any thinking metadata returned by Haiku."""
    invalid = ApiError(502, "invalid_anthropic_response", "Anthropic returned an invalid menu. Try again.")
    if not isinstance(message, anthropic_sdk.types.Message):
        raise invalid
    if message.stop_reason == "refusal":
        raise ApiError(422, "transcription_refused", "Anthropic could not transcribe these images. Try different menu photos.")
    if message.stop_reason in ("max_tokens", "model_context_window_exceeded"):
        raise ApiError(502, "incomplete_transcription", "The menu is too long to transcribe in one response. Try fewer images.")
    if message.stop_reason != "end_turn" or not isinstance(message.content, list) or not message.content:
        raise invalid
    texts = []
    for block in message.content:
        if isinstance(block, (anthropic_sdk.types.ThinkingBlock, anthropic_sdk.types.RedactedThinkingBlock)):
            continue
        if not isinstance(block, anthropic_sdk.types.TextBlock) or not isinstance(block.text, str):
            raise invalid
        texts.append(block.text)
    if not texts:
        raise invalid
    try:
        data = json.loads("".join(texts))
    except (ValueError, TypeError, RecursionError):
        raise invalid from None
    return validate_menu(data, provider_name="Anthropic")


def map_api_error(error):
    """Translate upstream errors without exposing response bodies or credentials."""
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error", body)
    details = details if isinstance(details, dict) else {}
    error_type = details.get("type")
    # Exhausted credits are currently reported as invalid_request_error,
    # with billing information in the message rather than a dedicated code.
    message = details.get("message", "")
    message = message.lower() if isinstance(message, str) else ""
    if error_type == "billing_error" or (error.status_code == 400 and any(
        phrase in message for phrase in ("credit balance is too low", "insufficient credits", "billing")
    )):
        return ApiError(503, "anthropic_not_funded", "Anthropic credits or billing quota are exhausted. The operator must check billing.")
    if error.status_code in (401, 403):
        return ApiError(503, "anthropic_authentication_failed", "Anthropic credentials or model permissions need attention.")
    if error.status_code == 429:
        return ApiError(503, "anthropic_rate_limited", "Anthropic is rate limiting requests. Try again later.")
    if error.status_code in (408, 504):
        return ApiError(504, "anthropic_timeout", "Menu transcription timed out. Try fewer or smaller images.")
    if error.status_code == 413 or (error.status_code == 400 and "image" in message):
        return ApiError(400, "invalid_image", "Anthropic could not read an image. Re-export it as JPEG, PNG, or WebP.")
    if error.status_code == 404:
        return ApiError(503, "anthropic_model_unavailable", "The configured Anthropic model is unavailable. The operator must check model access.")
    if error.status_code in (503, 529):
        return ApiError(503, "anthropic_unavailable", "Anthropic is temporarily unavailable. Try again later.")
    return ApiError(502, "anthropic_failure", "Anthropic could not complete menu transcription.")