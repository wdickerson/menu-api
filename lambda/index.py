"""Version 2 menu transcription, exposed as an API Gateway REST proxy."""

import base64
import binascii
import json
import logging
import os
from pathlib import Path
import re
from typing import cast

from providers import anthropic as anthropic_provider
from providers import openai as openai_provider
from providers.shared import ApiError, ImageMediaType, MenuImage
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header


# Lambda's synchronous event limit is 6 MiB (6,291,456 bytes). A 4,600,000-byte
# binary body becomes 6,133,336 base64 bytes, leaving 158,120 bytes for metadata.
# Oversized events may still be rejected by AWS before this handler is invoked.
MAX_BODY_BYTES = 4_600_000  # Complete multipart body, including boundaries/headers.
MAX_IMAGE_BYTES = 4_500_000  # Combined original file bytes, not per file.
MAX_JSON_BODY_BYTES = 1024  # JSON is retained only for {"dummy": true}.

TRANSCRIPTION_PROVIDER = "ANTHROPIC"  # Set to "OPENAI" or "ANTHROPIC" in code only.

TEXT_FIELD_LIMITS = {b"dummy": 5}
logger = logging.getLogger(__name__)


def parse_payload(event):
    values = [value for key, value in (event.get("headers") or {}).items()
              if key.lower() == "content-type"]
    for key, entries in (event.get("multiValueHeaders") or {}).items():
        if key.lower() == "content-type":
            values.extend(entries)
    if not values:
        raise ApiError(415, "unsupported_media_type", "Send multipart/form-data with image files.")
    if len(set(values)) != 1 or any(char in values[0] for char in "\r\n"):
        raise ApiError(400, "invalid_request", "Provide one valid Content-Type header.")
    try:
        media_type, options = parse_options_header(values[0])
    except (ValueError, UnicodeError):
        raise ApiError(400, "invalid_request", "Invalid Content-Type header.") from None
    media_type = media_type.lower()
    if media_type == b"application/json":
        return parse_dummy_json(decode_body(event, MAX_JSON_BODY_BYTES))
    if media_type != b"multipart/form-data":
        raise ApiError(415, "unsupported_media_type", "Send multipart/form-data with image files.")
    boundary = options.get(b"boundary", b"")
    if not re.fullmatch(rb"[0-9A-Za-z'()+_,./:=? -]{1,70}", boundary) or boundary.endswith(b" "):
        raise ApiError(400, "invalid_multipart", "A valid multipart boundary is required. Let your client set Content-Type.")
    if event.get("isBase64Encoded") is not True:
        # Never attempt a lossy UTF-8/Latin-1 reconstruction of binary images.
        raise ApiError(400, "invalid_request", "Multipart data was not delivered as binary. Check API Gateway binary media configuration.")
    return parse_multipart(decode_body(event, MAX_BODY_BYTES), boundary)


def decode_body(event, max_bytes):
    body = event.get("body") or ""
    if not isinstance(body, str):
        raise ApiError(400, "invalid_body", "Request body must be a string in the gateway event.")
    encoded = event.get("isBase64Encoded", False)
    limit = 4 * ((max_bytes + 2) // 3) if encoded else max_bytes
    if len(body) > limit:
        raise ApiError(413, "request_too_large", f"Request body exceeds {max_bytes:,} bytes.")
    try:
        raw = base64.b64decode(body, validate=True) if encoded else body.encode("utf-8")
    except (ValueError, binascii.Error):
        raise ApiError(400, "invalid_body", "Invalid gateway body encoding.") from None
    if len(raw) > max_bytes:
        raise ApiError(413, "request_too_large", f"Request body exceeds {max_bytes:,} bytes.")
    return raw


def parse_dummy_json(body):
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, TypeError, binascii.Error, RecursionError):
        raise ApiError(400, "invalid_json", "Body must be a valid JSON object.") from None
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_json", "Body must be a JSON object.")
    if set(payload) != {"dummy"} or payload["dummy"] is not True:
        raise ApiError(400, "invalid_request", 'JSON is supported only for {"dummy": true}. Send image files using multipart/form-data.')
    return payload


def parse_multipart(body, boundary):
    """Bounded, in-memory parsing; filenames are never used as filesystem paths."""
    images = []
    dummy = None
    seen_text_fields = set()
    total_bytes = 0
    part_count = 0
    ended = False
    headers = {}
    header_name = bytearray()
    header_value = bytearray()
    part_data = bytearray()
    field_name = b""
    media_type = b""

    def on_part_begin():
        nonlocal part_count
        part_count += 1
        if part_count > 11:  # Ten image files plus the optional dummy flag.
            raise ApiError(400, "invalid_request", "Too many multipart fields.")
        headers.clear()
        part_data.clear()

    def on_header_begin():
        header_name.clear()
        header_value.clear()

    def on_header_field(data, start, end):
        header_name.extend(data[start:end])

    def on_header_value(data, start, end):
        header_value.extend(data[start:end])

    def on_header_end():
        name = bytes(header_name).lower()
        if name in headers:
            raise ApiError(400, "invalid_multipart", "Duplicate multipart headers are not supported.")
        headers[name] = bytes(header_value)

    def on_headers_finished():
        nonlocal field_name, media_type
        disposition, options = parse_options_header(headers.get(b"content-disposition"))
        field_name = options.get(b"name", b"")
        if disposition.lower() != b"form-data" or field_name not in (b"images", *TEXT_FIELD_LIMITS):
            raise ApiError(400, "invalid_request", "Use images file fields and an optional dummy text field.")
        if b"content-transfer-encoding" in headers:
            raise ApiError(400, "invalid_multipart", "Send original file bytes, not transfer-encoded multipart parts.")
        if field_name == b"images":
            if not options.get(b"filename"):
                raise ApiError(400, "invalid_image", "Each images field must be a file with a nonempty filename.")
            if len(images) >= 10:
                raise ApiError(400, "invalid_images", "Provide between 1 and 10 image files.")
            media_type = parse_options_header(headers.get(b"content-type"))[0].lower()
        else:
            if b"filename" in options or field_name in seen_text_fields:
                raise ApiError(400, "invalid_request", "Provide dummy once, as a text field.")
            seen_text_fields.add(field_name)

    def on_part_data(data, start, end):
        nonlocal total_bytes
        if field_name == b"images":
            total_bytes += end - start
            if total_bytes > MAX_IMAGE_BYTES:
                raise ApiError(413, "images_too_large", f"Combined image files exceed {MAX_IMAGE_BYTES:,} bytes. Resize or compress them.")
        elif len(part_data) + end - start > TEXT_FIELD_LIMITS[field_name]:
            raise ApiError(400, "invalid_request", "The dummy text field exceeds its length limit.")
        part_data.extend(data[start:end])

    def on_part_end():
        nonlocal dummy
        if field_name == b"images":
            images.append((media_type, bytes(part_data)))
        elif field_name == b"dummy":
            if part_data not in (b"true", b"false"):
                raise ApiError(400, "invalid_request", "dummy must be the text true or false.")
            dummy = part_data == b"true"

    def on_end():
        nonlocal ended
        ended = True

    parser = MultipartParser(boundary, callbacks={
        "on_part_begin": on_part_begin, "on_header_begin": on_header_begin,
        "on_header_field": on_header_field, "on_header_value": on_header_value,
        "on_header_end": on_header_end, "on_headers_finished": on_headers_finished,
        "on_part_data": on_part_data, "on_part_end": on_part_end, "on_end": on_end,
    }, max_size=MAX_BODY_BYTES, max_header_count=8, max_header_size=4096)
    try:
        parser.write(body)
        parser.finalize()
    except (MultipartParseError, ValueError):
        raise ApiError(400, "invalid_multipart", "Malformed multipart data or oversized part headers.") from None
    # finalize() alone does not reject a truncated body in python-multipart.
    if not ended:
        raise ApiError(400, "invalid_multipart", "Incomplete multipart body or mismatched boundary.")
    return {"dummy": dummy is True, "images": images}


def validate_images(images) -> list[MenuImage]:
    if not isinstance(images, list) or not 1 <= len(images) <= 10:
        raise ApiError(400, "invalid_images", "Provide between 1 and 10 images file fields.")
    validated_images = []
    for declared_type, raw in images:
        if declared_type not in (b"", b"application/octet-stream", b"image/jpeg", b"image/png", b"image/webp"):
            raise ApiError(400, "unsupported_image_type", "Use JPEG, PNG, or WebP image files.")
        # Sniff only signatures; the provider performs full image decoding. Missing or
        # generic MIME types are inferred from bytes, never from file extensions.
        if raw.startswith(b"\xff\xd8\xff"):
            actual_type = b"image/jpeg"
        elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
            actual_type = b"image/png"
        elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            actual_type = b"image/webp"
        else:
            raise ApiError(400, "invalid_image", "File is empty or does not have a supported image signature.")
        if declared_type not in (b"", b"application/octet-stream", actual_type):
            raise ApiError(400, "invalid_image", "Image bytes do not match the declared image type.")
        validated_images.append(MenuImage(
            media_type=cast(ImageMediaType, actual_type.decode("ascii")),
            base64_data=base64.b64encode(raw).decode("ascii"),
        ))
    return validated_images


def lambda_handler(event, context):
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Cache-Control": "no-store",
    }

    def response(status_code, payload):
        return {"statusCode": status_code, "headers": headers, "body": json.dumps(payload)}

    try:
        if event.get("httpMethod") != "POST":
            headers["Allow"] = "POST,OPTIONS"
            raise ApiError(405, "method_not_allowed", "Method not allowed. Use POST.")
        payload = parse_payload(event)

        if payload.get("dummy", False):
            # No credentials, Secrets Manager calls, or provider requests in dummy mode.
            menu = json.loads(Path(__file__).with_name("dummy-menu.json").read_text(encoding="utf-8"))
            return response(200, {"isDummy": True, **menu})
        images = validate_images(payload.get("images"))
        if os.environ.get("ENABLE_LIVE_TRANSCRIPTION", "false").lower() != "true":
            raise ApiError(503, "transcription_disabled", "Live menu transcription is disabled. Use dummy mode or contact the operator.")
        if TRANSCRIPTION_PROVIDER == "ANTHROPIC":
            return response(200, anthropic_provider.transcribe_menu(images, context))
        if TRANSCRIPTION_PROVIDER != "OPENAI":
            raise ApiError(503, "api_not_configured", "The menu transcription provider is not configured correctly.")
        return response(200, openai_provider.transcribe_menu(images, context))
    except ApiError as error:
        if error.status >= 500:
            logger.warning("Menu API failure: %s", error.code)
        return response(error.status, {"error": {"code": error.code, "message": error.message}})
    except Exception:
        # Do not log exceptions: upstream messages can contain keys or image data.
        logger.error("Menu API failure: internal_error")
        return response(500, {"error": {"code": "internal_error", "message": "An unexpected error occurred."}})