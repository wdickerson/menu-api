"""OpenAI image transcription and online menu search with code-owned settings."""

import asyncio
import json

import openai as openai_sdk
from openai.types.responses import (
    Response,
    ResponseError,
    ResponseInputContentParam,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.shared import ReasoningEffort

from .shared import (
    ApiError,
    MenuImage,
    MENU_TRANSCRIPTION_PROMPT,
    MENU_TRANSCRIPTION_SCHEMA,
    MENU_SEARCH_PROMPT,
    MENU_SEARCH_SCHEMA,
    get_api_key,
    get_request_timeout,
    run_with_deadline,
    run_search_with_deadline,
    validate_menu,
    validate_search_results,
    search_error,
)


DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_REASONING_EFFORT: ReasoningEffort = "none"
SEARCH_MODEL = "gpt-6-luna"
SEARCH_REASONING_EFFORT: ReasoningEffort = "medium"
TIMEOUT_SECONDS = 20.0


def transcribe_menu(images: list[MenuImage], context):
    """Send all images in one request; never retry or fall back to another provider."""
    api_key = get_api_key("OPENAI_API_KEY")
    timeout = get_request_timeout(context, provider_name="OpenAI", max_seconds=TIMEOUT_SECONDS)
    content: list[ResponseInputContentParam] = [{"type": "input_text", "text": "Transcribe these menu images."}]
    content.extend({
        "type": "input_image", "image_url": f"data:{image.media_type};base64,{image.base64_data}",
        "detail": "high",
    } for image in images)
    try:
        response = asyncio.run(run_with_deadline(
            _request_response(api_key, content, timeout), timeout))
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
    return parse_response(response)


async def _request_response(api_key, content, timeout):
    async with openai_sdk.AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                                     timeout=timeout, max_retries=0) as client:
        return await client.responses.create(
            model=DEFAULT_MODEL,
            service_tier="ultrafast",
            reasoning={"effort": "low"},
            instructions=MENU_TRANSCRIPTION_PROMPT,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema",
                "name": "menu_transcription", "strict": True, "schema": MENU_TRANSCRIPTION_SCHEMA,
            }},
            max_output_tokens=8000,
            store=False,
        )


def parse_response(response):
    return validate_menu(_parse_response_data(response), provider_name="OpenAI")


def _parse_response_data(response):
    """Require a complete, non-refused Response before validating its menu text."""
    invalid = ApiError(502, "invalid_openai_response", "OpenAI returned an invalid response.")
    if not isinstance(response, Response):
        raise invalid
    if response.error is not None:
        if not isinstance(response.error, ResponseError) or not isinstance(getattr(response.error, "code", None), str):
            raise invalid
        raise _map_error(None, (response.error.code,))
    output = getattr(response, "output", None)
    # Preserve refusal priority even when partial or malformed text accompanies it.
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, ResponseOutputMessage) or getattr(item, "type", None) != "message":
                continue
            content = getattr(item, "content", None)
            if isinstance(content, list) and any(
                isinstance(part, ResponseOutputRefusal) and getattr(part, "type", None) == "refusal"
                for part in content
            ):
                raise ApiError(422, "transcription_refused", "OpenAI could not transcribe these images. Try different menu photos.")
    details = response.incomplete_details
    if details is not None and not isinstance(details, IncompleteDetails):
        raise invalid
    reason = details.reason if details is not None else None
    if reason == "content_filter":
        raise ApiError(422, "transcription_refused", "OpenAI could not transcribe these images. Try different menu photos.")
    if reason == "max_output_tokens":
        raise ApiError(502, "incomplete_transcription", "The menu is too long to transcribe in one response. Try fewer images.")
    if response.status != "completed" or details is not None:
        raise ApiError(502, "invalid_openai_response", "OpenAI did not finish the menu transcription.")
    # The SDK accepts partially malformed envelopes; check types before using output_text.
    if not isinstance(output, list) or not output:
        raise ApiError(502, "invalid_openai_response", "OpenAI returned no menu.")
    for item in output:
        if isinstance(item, ResponseReasoningItem) and item.type == "reasoning":
            if item.status not in (None, "completed"):
                raise invalid
            continue
        if (not isinstance(item, ResponseOutputMessage) or getattr(item, "type", None) != "message"
                or getattr(item, "role", None) != "assistant" or getattr(item, "status", None) != "completed"
                or not isinstance(getattr(item, "content", None), list) or not item.content):
            raise invalid
        for part in item.content:
            if (not isinstance(part, ResponseOutputText) or getattr(part, "type", None) != "output_text"
                    or not isinstance(getattr(part, "text", None), str)):
                raise invalid
    try:
        data = json.loads(response.output_text)
    except (ValueError, TypeError, RecursionError):
        raise ApiError(502, "invalid_openai_response", "OpenAI returned an invalid menu.") from None
    return data


def search_menus(query: str, context):
    api_key = get_api_key("OPENAI_API_KEY")
    timeout = get_request_timeout(context, provider_name="OpenAI", max_seconds=TIMEOUT_SECONDS)
    try:
        response = run_search_with_deadline(_request_search(api_key, query, timeout), timeout)
        # Temporary debug output; remove before deploying to production.
        return validate_search_results(json.loads(response.output_text), provider_name="OpenAI")
    except (openai_sdk.APITimeoutError, asyncio.TimeoutError):
        raise ApiError(504, "openai_timeout", "Menu search timed out. Please try again.") from None
    except openai_sdk.APIStatusError as error:
        raise search_error(map_api_error(error)) from None
    except openai_sdk.APIConnectionError:
        raise ApiError(502, "openai_unavailable", "Could not connect to OpenAI. Try again later.") from None
    except openai_sdk.OpenAIError:
        raise ApiError(502, "openai_failure", "OpenAI could not complete menu search.") from None
    except ApiError as error:
        raise search_error(error) from None
    except (ValueError, TypeError, AttributeError, RecursionError):
        raise ApiError(502, "invalid_openai_response", "OpenAI returned invalid menu search results.") from None


async def _request_search(api_key, query, timeout):
    async with openai_sdk.AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                                     timeout=timeout, max_retries=0) as client:
        return await client.responses.create(
            model=SEARCH_MODEL,
            reasoning={"effort": SEARCH_REASONING_EFFORT},
            instructions=MENU_SEARCH_PROMPT,
            input=[{"role": "user", "content": [{"type": "input_text", "text": query}]}],
            tools=[{"type": "web_search"}], tool_choice="required", max_tool_calls=3,
            text={"format": {"type": "json_schema", "name": "menu_online_search",
                             "strict": True, "schema": MENU_SEARCH_SCHEMA}},
            max_output_tokens=4000, store=False,
        )


def map_api_error(error):
    """Translate upstream errors without exposing response bodies or credentials."""
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error", body)
    details = details if isinstance(details, dict) else {}
    codes = (details.get("code"), details.get("type"))
    return _map_error(error.status_code, codes)


def _map_error(status_code, codes):
    """Share safe mappings between HTTP errors and failed Responses."""
    if any(code in ("insufficient_quota", "billing_hard_limit_reached", "billing_not_active") for code in codes):
        return ApiError(503, "openai_not_funded", "OpenAI credits or billing quota are exhausted. The operator must check billing.")
    if status_code in (401, 403):
        return ApiError(503, "openai_authentication_failed", "OpenAI credentials or model permissions need attention.")
    if status_code == 429 or (status_code is None and "rate_limit_exceeded" in codes):
        return ApiError(503, "openai_rate_limited", "OpenAI is rate limiting requests. Try again later.")
    if status_code in (408, 504):
        return ApiError(504, "openai_timeout", "Menu transcription timed out. Try fewer or smaller images.")
    if any(code in ("invalid_image", "invalid_image_format", "invalid_base64_image", "image_parse_error", "image_too_large", "unsupported_image") for code in codes):
        return ApiError(400, "invalid_image", "OpenAI could not read an image. Re-export it as JPEG, PNG, or WebP.")
    return ApiError(502, "openai_failure", "OpenAI could not complete menu transcription.")