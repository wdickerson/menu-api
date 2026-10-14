"""OpenAI menu transcription with code-owned settings, Chat Completions, and response parsing."""

import asyncio
import json

import openai as openai_sdk
from openai.types.chat import ChatCompletionContentPartParam
from openai.types.shared import ReasoningEffort

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


DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT: ReasoningEffort = "none"
TIMEOUT_SECONDS = 20.0


def transcribe_menu(images: list[MenuImage], context):
    """Send all images in one request; never retry or fall back to another provider."""
    api_key = get_api_key("OPENAI_API_KEY")
    timeout = get_request_timeout(context, provider_name="OpenAI", max_seconds=TIMEOUT_SECONDS)
    content: list[ChatCompletionContentPartParam] = [{"type": "text", "text": "Transcribe these menu images."}]
    content.extend({"type": "image_url", "image_url": {
        "url": f"data:{image.media_type};base64,{image.base64_data}", "detail": "high",
    }} for image in images)
    try:
        completion = asyncio.run(run_with_deadline(
            _request_completion(api_key, content, timeout), timeout))
    except (openai_sdk.APITimeoutError, asyncio.TimeoutError):
        raise ApiError(504, "openai_timeout", "Menu transcription timed out. Try fewer or smaller images.") from None
    except openai_sdk.APIStatusError as error:
        raise map_api_error(error) from None
    except openai_sdk.APIConnectionError:
        raise ApiError(502, "openai_unavailable", "Could not connect to OpenAI. Try again later.") from None
    except openai_sdk.OpenAIError:
        raise ApiError(502, "openai_failure", "OpenAI could not complete menu transcription.") from None
    except (ValueError, TypeError):
        raise ApiError(502, "invalid_openai_response", "OpenAI returned an invalid response.") from None
    return parse_response(completion)


async def _request_completion(api_key, content, timeout):
    async with openai_sdk.AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                                     timeout=timeout, max_retries=0) as client:
        return await client.chat.completions.create(
            model=DEFAULT_MODEL,
            # Mini does not support the reasoning parameter; other models use the code setting.
            reasoning_effort=openai_sdk.omit if DEFAULT_MODEL == "gpt-4.1-mini" else DEFAULT_REASONING_EFFORT,
            messages=[{"role": "system", "content": MENU_TRANSCRIPTION_PROMPT},
                      {"role": "user", "content": content}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "menu_transcription", "strict": True, "schema": MENU_TRANSCRIPTION_SCHEMA,
            }},
            max_completion_tokens=8000,
            store=False,
        )


def parse_response(completion):
    """Require a complete, non-refused Chat Completion before validating its menu."""
    if not isinstance(completion, openai_sdk.types.chat.ChatCompletion) or not completion.choices:
        raise ApiError(502, "invalid_openai_response", "OpenAI returned no menu.")
    choice = completion.choices[0]
    if not isinstance(choice, openai_sdk.types.chat.chat_completion.Choice) or not isinstance(
        choice.message, openai_sdk.types.chat.ChatCompletionMessage
    ):
        raise ApiError(502, "invalid_openai_response", "OpenAI returned an invalid response.")
    if choice.message.refusal or choice.finish_reason == "content_filter":
        raise ApiError(422, "transcription_refused", "OpenAI could not transcribe these images. Try different menu photos.")
    if choice.finish_reason == "length":
        raise ApiError(502, "incomplete_transcription", "The menu is too long to transcribe in one response. Try fewer images.")
    if choice.finish_reason != "stop":
        raise ApiError(502, "invalid_openai_response", "OpenAI did not finish the menu transcription.")
    try:
        data = json.loads(choice.message.content or "")
    except (ValueError, TypeError, RecursionError):
        raise ApiError(502, "invalid_openai_response", "OpenAI returned an invalid menu.") from None
    return validate_menu(data, provider_name="OpenAI")


def map_api_error(error):
    """Translate upstream errors without exposing response bodies or credentials."""
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error", body)
    details = details if isinstance(details, dict) else {}
    codes = (details.get("code"), details.get("type"))
    if any(code in ("insufficient_quota", "billing_hard_limit_reached", "billing_not_active") for code in codes):
        return ApiError(503, "openai_not_funded", "OpenAI credits or billing quota are exhausted. The operator must check billing.")
    if error.status_code in (401, 403):
        return ApiError(503, "openai_authentication_failed", "OpenAI credentials or model permissions need attention.")
    if error.status_code == 429:
        return ApiError(503, "openai_rate_limited", "OpenAI is rate limiting requests. Try again later.")
    if error.status_code in (408, 504):
        return ApiError(504, "openai_timeout", "Menu transcription timed out. Try fewer or smaller images.")
    if any(code in ("invalid_image", "invalid_image_format", "invalid_base64_image", "image_parse_error", "image_too_large", "unsupported_image") for code in codes):
        return ApiError(400, "invalid_image", "OpenAI could not read an image. Re-export it as JPEG, PNG, or WebP.")
    return ApiError(502, "openai_failure", "OpenAI could not complete menu transcription.")