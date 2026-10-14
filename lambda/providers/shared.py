"""Provider-independent image data, credentials, deadlines, and menu validation."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
import json
import os
import time
from typing import Literal, TypeVar

import boto3
from botocore.config import Config


ImageMediaType = Literal["image/jpeg", "image/png", "image/webp"]
SECRET_CACHE_SECONDS = 300
LAMBDA_RESPONSE_BUFFER_SECONDS = 3
_api_key_cache: dict[tuple[str, str], tuple[str, float]] = {}
ResponseT = TypeVar("ResponseT")

MENU_TRANSCRIPTION_PROMPT = """
Quickly transcribe the menu in these images.
Prefer transcriptions that are easily read aloud by screen readers.
Omit standard footers (ie, about fees, allergies, warnings, etc).
Include item descriptions if they are legible.
If item descriptions include symbols, include their meaning.
If you can't find a restaurant name, use a descriptive placeholder.
"""
# MENU_TRANSCRIPTION_PROMPT = """
# Find the menu in these images and transcribe it quickly.
# Include item descriptions if they are legible.
# Include all sections, even if they are non-food sections.
# Include symbols only if their meaning is apparent.
# Use lowercase letters and minimize whitespace.
# Your output will be read by a screen reader for blind users.
# If you can't find a restaurant name, use a descriptive placeholder.
# """

MENU_TRANSCRIPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "restaurantName", "sections"],
    "properties": {
        "status": {"type": "string", "enum": ["success", "no_menu", "illegible_text"]},
        "restaurantName": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "items"],
                "properties": {
                    "name": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


class ApiError(Exception):
    """Only these explicitly safe messages may be sent to callers."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class MenuImage:
    """Validated image bytes encoded once, without a provider-specific wrapper."""

    media_type: ImageMediaType
    base64_data: str


def get_api_key(key_field: str) -> str:
    """Read only the chosen key field; cache separately by secret ID and provider."""
    key = os.environ.get(key_field, "").strip()
    if not key:
        secret_id = os.environ.get("MENU_API_SECRET_ID")
        if not secret_id:
            raise ApiError(503, "api_not_configured", "Menu transcription credentials are not configured.")
        cache_key = (secret_id, key_field)
        cached = _api_key_cache.get(cache_key)
        if cached and time.monotonic() < cached[1]:
            return cached[0]
        try:
            client = boto3.client("secretsmanager", config=Config(
                connect_timeout=2, read_timeout=2, retries={"total_max_attempts": 1},
            ))
            secret = json.loads(client.get_secret_value(SecretId=secret_id)["SecretString"])
            key = secret[key_field].strip()
            if not key or not key.isascii() or any(char.isspace() for char in key):
                raise ValueError("Invalid key")
        except Exception:
            # SDK exceptions and secret contents must never reach responses/logs.
            raise ApiError(503, "api_not_configured", "Menu transcription credentials are unavailable.") from None
        _api_key_cache[cache_key] = (key, time.monotonic() + SECRET_CACHE_SECONDS)
    if not key.isascii() or any(char.isspace() for char in key):
        raise ApiError(503, "api_not_configured", "Menu transcription credentials are invalid.")
    return key


def get_request_timeout(context, *, provider_name: str, max_seconds: float) -> float:
    """Leave time for Lambda to serialize and return its HTTP response."""
    timeout = max_seconds
    if context is not None:
        timeout = min(timeout, context.get_remaining_time_in_millis() / 1000 - LAMBDA_RESPONSE_BUFFER_SECONDS)
    if timeout <= 0:
        raise ApiError(504, f"{provider_name.lower()}_timeout", "Not enough time remains to transcribe the menu.")
    return timeout


async def run_with_deadline(request: Awaitable[ResponseT], timeout: float) -> ResponseT:
    # SDK timeouts bound individual network operations, not total elapsed time.
    return await asyncio.wait_for(request, timeout=timeout)


def validate_menu(data, *, provider_name: str):
    """Validate structured output and normalize it to the frontend menu contract."""
    invalid = ApiError(502, f"invalid_{provider_name.lower()}_response",
                       f"{provider_name} returned an invalid menu. Try again.")
    if not isinstance(data, dict) or set(data) != {"status", "restaurantName", "sections"}:
        raise invalid
    if not isinstance(data["restaurantName"], str) or not isinstance(data["sections"], list):
        raise invalid
    if data["status"] == "no_menu":
        raise ApiError(422, "no_menu_found", "No food or drink menu was found in the images.")
    if data["status"] == "illegible_text":
        raise ApiError(422, "illegible_text", "The menu text is not legible. Try a closer, sharper photo.")
    if data["status"] != "success" or not data["sections"]:
        raise invalid
    sections = []
    for section in data["sections"]:
        if (not isinstance(section, dict) or set(section) != {"name", "items"}
                or not isinstance(section["name"], str) or not section["name"].strip()
                or not isinstance(section["items"], list) or not section["items"]
                or any(not isinstance(item, str) or not item.strip() for item in section["items"])):
            raise invalid
        sections.append({"name": section["name"].strip(), "items": [item.strip() for item in section["items"]]})
    return {
        "isDummy": False,
        "restaurantName": data["restaurantName"].strip() or "Restaurant",
        "sections": sections,
    }