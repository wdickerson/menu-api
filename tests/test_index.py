"""Offline contract tests using the real OpenAI SDK with a mocked HTTP transport."""

import asyncio
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import AsyncOpenAI


HANDLER_PATH = Path(__file__).resolve().parents[1] / "lambda" / "index.py"
# Source is not yet copied into the deployment package when CodeBuild runs tests.
sys.path.insert(0, str(HANDLER_PATH.parent))
spec = importlib.util.spec_from_file_location("menu_handler", HANDLER_PATH)
assert spec is not None and spec.loader is not None
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)

openai_provider = handler.openai_provider
shared = importlib.import_module("providers.shared")

# Signature-only fixtures; full image decoding belongs to OpenAI.
PNG_BYTES = b"\x89PNG\r\n\x1a\ntest-image\x00\xff\x80\r\n"
PNG_FILE = ("images", ("menu.png", PNG_BYTES, "image/png"))
PNG = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
MENU = {"status": "success", "restaurantName": "Coffee Shop",
        "sections": [{"name": "Drinks", "items": ["Café latte. $5.", "Tea. $3."]}]}


def upload_event(files=None, fields=None):
    """Use a real HTTP client encoder, then emulate Gateway's binary wrapping."""
    parts = [(key, (None, value)) for key, value in (fields or [])]
    parts += files if files is not None else [PNG_FILE]
    request = httpx.Request("POST", "https://menu.example.com/menu_image_transcribe",
                            files=parts or [("dummy", (None, "false"))])
    return {
        "httpMethod": "POST", "path": "/menu_image_transcribe",
        "body": base64.b64encode(request.read()).decode("ascii"),
        "isBase64Encoded": True, "headers": {"Content-Type": request.headers["content-type"]},
    }


class MenuHandlerTests(unittest.TestCase):
    def setUp(self):
        self.start_patch(patch.dict(os.environ, {
            "ALLOWED_ORIGIN": "https://menu.example.com", "ENABLE_LIVE_TRANSCRIPTION": "true",
            "OPENAI_API_KEY": "test-key-not-a-real-secret",
        }, clear=True))
        self.start_patch(patch.object(handler, "TRANSCRIPTION_PROVIDER", "OPENAI"))
        self.start_patch(patch.object(shared, "_api_key_cache", {}))
        self.aws = self.start_patch(patch.object(shared.boto3, "client"))
        self.aws.side_effect = AssertionError("Unexpected AWS call")
        self.requests = []
        self.http_status = 200
        self.http_body = self.response(MENU)
        self.network_error = None
        self.openai = self.start_patch(patch.object(openai_provider.openai_sdk, "AsyncOpenAI", side_effect=self.make_client))
        self.log = self.start_patch(patch.object(handler, "logger"))

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def make_client(self, **kwargs):
        return AsyncOpenAI(http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.transport)), **kwargs)

    def transport(self, request):
        self.requests.append(request)
        if self.network_error:
            raise self.network_error
        if isinstance(self.http_body, bytes):
            return httpx.Response(self.http_status, content=self.http_body)
        return httpx.Response(self.http_status, json=self.http_body)

    @staticmethod
    def response(data, status="completed", incomplete_reason=None, refusal=None):
        content = [{"type": "output_text", "text": json.dumps(data), "annotations": []}]
        if refusal is not None:
            content = [{"type": "refusal", "refusal": refusal}]
        return {
            "id": "resp-test", "object": "response", "created_at": 0, "model": "gpt-6.1-sol",
            "status": status, "error": None,
            "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
            "output": [{"id": "msg-test", "type": "message", "status": "completed",
                        "role": "assistant", "content": content}],
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        }

    def invoke(self, payload=None, *, body=None, method="POST", encoded=None, headers=None,
               context=None, files=None, fields=None):
        if payload is not None or body is not None:
            event = {"body": json.dumps(payload) if body is None else body,
                     "isBase64Encoded": False, "headers": {"Content-Type": "application/json"}}
        else:
            event = upload_event(files, fields)
        event["path"] = "/menu_image_transcribe"
        event["httpMethod"] = method
        if headers is not None:
            event["headers"] = headers
        if encoded is not None:
            event["isBase64Encoded"] = encoded
        self.last_event = event
        result = handler.lambda_handler(event, context)
        for key, expected in {
            "Content-Type": "application/json", "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST,OPTIONS", "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-store",
        }.items():
            self.assertEqual(result["headers"][key], expected)
        return result, json.loads(result["body"])

    def assert_error(self, status, code, **kwargs):
        result, data = self.invoke(**kwargs)
        self.assertEqual(result["statusCode"], status, data)
        self.assertEqual(data["error"]["code"], code, data)
        self.assertTrue(data["error"]["message"])
        return result, data

    def assert_provider_settings(self, sent):
        """Check the configured settings in the JSON sent to the provider SDK."""
        self.assertEqual(sent["model"], "gpt-6.1-sol")
        self.assertEqual(sent["service_tier"], "ultrafast")
        self.assertEqual(sent["reasoning"], {"effort": "low"})
        for legacy_field in ("messages", "response_format", "reasoning_effort", "max_completion_tokens"):
            self.assertNotIn(legacy_field, sent)

    def test_real_transcription_matches_frontend_contract(self):
        result, menu = self.invoke()
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(menu, {"isDummy": False, "restaurantName": MENU["restaurantName"], "sections": MENU["sections"]})
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
        self.assertEqual(request.headers["authorization"], "Bearer test-key-not-a-real-secret")
        sent = json.loads(request.content)
        self.assert_provider_settings(sent)
        self.assertEqual(sent["instructions"], shared.MENU_TRANSCRIPTION_PROMPT)
        self.assertFalse(sent["store"])
        self.assertEqual(sent["max_output_tokens"], 8000)
        self.assertEqual(sent["text"]["format"], {
            "type": "json_schema", "name": "menu_transcription", "strict": True,
            "schema": shared.MENU_TRANSCRIPTION_SCHEMA,
        })
        self.assertEqual(sent["input"], [{"role": "user", "content": [
            {"type": "input_text", "text": "Transcribe these menu images."},
            {"type": "input_image", "image_url": PNG, "detail": "high"},
        ]}])
        self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
        self.aws.assert_not_called()

    def test_one_through_five_images_use_one_request(self):
        for count in range(1, 6):
            before = len(self.requests)
            result, _ = self.invoke(files=[PNG_FILE] * count, fields=[("dummy", "false")])
            self.assertEqual(result["statusCode"], 200)
            self.assertEqual(len(self.requests), before + 1)
            self.assertEqual(len(json.loads(self.requests[-1].content)["input"][0]["content"]), count + 1)

    def test_distinct_images_preserve_upload_order(self):
        images = [("image/jpeg", b"\xff\xd8\xfffirst"), ("image/png", PNG_BYTES),
                  ("image/webp", b"RIFF1234WEBPlast")]
        files = [("images", (f"page-{index}", raw, mime)) for index, (mime, raw) in enumerate(images)]
        result, _ = self.invoke(files=files)
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(len(self.requests), 1)
        sent = json.loads(self.requests[0].content)
        self.assertEqual(sent["input"][0]["content"][1:], [
            {"type": "input_image", "detail": "high",
             "image_url": f"data:{mime};base64," + base64.b64encode(raw).decode()}
            for mime, raw in images
        ])

    def test_supported_image_signatures(self):
        for mime, raw in (("jpeg", b"\xff\xd8\xfftest"), ("png", b"\x89PNG\r\n\x1a\ntest"),
                          ("webp", b"RIFF1234WEBPtest")):
            result, _ = self.invoke(files=[("images", (f"menu.{mime}", raw, f"image/{mime}"))])
            self.assertEqual(result["statusCode"], 200)
            image = json.loads(self.requests[-1].content)["input"][0]["content"][1]["image_url"]
            self.assertEqual(image, f"data:image/{mime};base64," + base64.b64encode(raw).decode())

    def test_missing_or_generic_mime_is_sniffed(self):
        for mime in (None, "application/octet-stream"):
            result, _ = self.invoke(files=[("images", ("menu", PNG_BYTES, mime))])
            self.assertEqual(result["statusCode"], 200)
            self.assertEqual(json.loads(self.requests[-1].content)["input"][0]["content"][1]["image_url"], PNG)

    def test_dummy_works_without_credentials_or_live_mode(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "ENABLE_LIVE_TRANSCRIPTION": "false"}):
            result, menu = self.invoke({"dummy": True})
        self.assertEqual(result["statusCode"], 200)
        stored = json.loads(HANDLER_PATH.with_name("dummy-menu.json").read_text())
        self.assertEqual(menu, {"isDummy": True, **stored})
        self.assertEqual(menu["restaurantName"], "Rough & Tumble")
        self.assertEqual(len(menu["sections"]), 11)
        self.assertEqual(sum(len(section["items"]) for section in menu["sections"]), 86)
        self.assertEqual(set(menu), {"isDummy", "restaurantName", "sections"})
        self.openai.assert_not_called()
        self.aws.assert_not_called()

    def test_dummy_ignores_images_and_is_deterministic(self):
        _, first = self.invoke(files=[("images", ("invalid.png", b"not-an-image", "image/png"))],
                               fields=[("dummy", "true")])
        _, second = self.invoke({"dummy": True})
        self.assertEqual(first, second)
        self.assertNotIn("not-an-image", json.dumps(first))
        self.openai.assert_not_called()

    def test_multipart_dummy_without_files_credentials_or_live_mode(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "ENABLE_LIVE_TRANSCRIPTION": "false"}):
            result, menu = self.invoke(files=[], fields=[("dummy", "true")])
        self.assertEqual(result["statusCode"], 200)
        self.assertTrue(menu["isDummy"])
        self.aws.assert_not_called()
        self.openai.assert_not_called()

    def test_missing_or_invalid_image_counts(self):
        self.assert_error(400, "invalid_images", files=[])
        self.assert_error(400, "invalid_images", files=[PNG_FILE] * 6)
        self.openai.assert_not_called()

    def test_json_is_only_for_dummy(self):
        for value in ("true", "false", 1, 0, None, [], {}):
            self.assert_error(400, "invalid_request", payload={"dummy": value})
        for payload in ({}, {"dummy": False}, {"images": [PNG]}, {"dummy": True, "images": []}):
            self.assert_error(400, "invalid_request", payload=payload)
        self.openai.assert_not_called()

    def test_dummy_text_is_strict_and_not_duplicated(self):
        for value in ("", "True", "1", "0", "true ", "invalid", "true" * 1000):
            self.assert_error(400, "invalid_request", fields=[("dummy", value)])
        self.assert_error(400, "invalid_request", fields=[("dummy", "true"), ("dummy", "false")])
        self.assert_error(400, "invalid_request", files=[("dummy", ("dummy.txt", b"true", "text/plain"))])

    def test_dummy_after_files_is_supported(self):
        result, menu = self.invoke(files=[PNG_FILE, ("dummy", (None, "true"))])
        self.assertEqual(result["statusCode"], 200)
        self.assertTrue(menu["isDummy"])
        self.openai.assert_not_called()

    def test_unknown_fields_rejected(self):
        for field in ("apiKey", "base_url", "imageUrl", "prompt", "unexpected_field"):
            self.assert_error(400, "invalid_request", fields=[(field, "private-value")])
        self.assert_error(400, "invalid_request", files=[("images[]", ("menu.png", PNG_BYTES, "image/png"))])

    def test_invalid_json(self):
        for body in ("", "not json", "[]", "null", '"hello"', "123", "true"):
            self.assert_error(400, "invalid_json", body=body)

    def test_base64_json_dummy_is_still_supported(self):
        body = base64.b64encode(b'{"dummy":true}').decode()
        result, _ = self.invoke(body=body, encoded=True)
        self.assertEqual(result["statusCode"], 200)

    def test_invalid_gateway_body(self):
        self.assert_error(400, "invalid_body", body="%%%", encoded=True)
        self.assert_error(400, "invalid_body", body=123)
        self.assert_error(400, "invalid_json", body=base64.b64encode(b"\xff").decode(), encoded=True)
        self.assert_error(400, "invalid_request", encoded=False)

    def test_content_type(self):
        self.assert_error(415, "unsupported_media_type", headers={})
        self.assert_error(415, "unsupported_media_type", headers={"Content-Type": "image/png"})
        self.assert_error(400, "invalid_multipart", headers={"content-type": "multipart/form-data"})
        result, _ = self.invoke({"dummy": True}, headers={"content-type": "application/json; charset=utf-8"})
        self.assertEqual(result["statusCode"], 200)

    def test_multivalue_and_quoted_boundary_headers(self):
        event = upload_event()
        content_type = event["headers"].pop("Content-Type")
        event["multiValueHeaders"] = {"content-type": [content_type]}
        self.assertEqual(handler.lambda_handler(event, None)["statusCode"], 200)
        mime, boundary = content_type.split("boundary=")
        event["multiValueHeaders"] = None
        event["headers"] = {"content-type": mime + 'boundary="' + boundary + '"'}
        self.assertEqual(handler.lambda_handler(event, None)["statusCode"], 200)
        event["multiValueHeaders"] = {"Content-Type": ["application/json"]}
        self.assertEqual(handler.lambda_handler(event, None)["statusCode"], 400)

    def test_invalid_images(self):
        for raw in (b"", b"not-an-image", PNG.encode(), b"GIF89atest"):
            self.assert_error(400, "invalid_image", files=[("images", ("menu.png", raw, "image/png"))])
        self.assert_error(400, "invalid_image", files=[("images", ("menu.jpg", PNG_BYTES, "image/jpeg"))])
        self.assert_error(400, "invalid_image", files=[("images", ("", PNG_BYTES, "image/png"))])
        self.assert_error(400, "invalid_image", fields=[("images", "https://example.com/menu.jpg")], files=[])
        self.openai.assert_not_called()

    def test_unsupported_image_types_rejected(self):
        for mime in ("image/gif", "image/heic", "application/pdf", "multipart/mixed"):
            self.assert_error(400, "unsupported_image_type", files=[("images", ("menu", PNG_BYTES, mime))])

    def test_body_size_limits(self):
        body = b"x" * (handler.MAX_BODY_BYTES + 1)
        self.assert_error(413, "request_too_large", body=base64.b64encode(body).decode(), encoded=True,
                          headers={"Content-Type": "multipart/form-data; boundary=boundary"})
        self.assert_error(413, "request_too_large", body=" " * (handler.MAX_JSON_BODY_BYTES + 1))
        self.openai.assert_not_called()

    def test_combined_image_size_limit(self):
        raw = PNG_BYTES + b"x" * (handler.MAX_IMAGE_BYTES - len(PNG_BYTES))
        self.assert_error(413, "images_too_large", files=[("images", ("menu.png", raw, "image/png")), PNG_FILE])
        self.assert_error(413, "images_too_large", files=[("images", ("menu.png", raw + b"x", "image/png"))])
        self.assert_error(413, "images_too_large", files=[("images", ("menu.png", raw + b"x", "image/png"))],
                          fields=[("dummy", "true")])
        self.openai.assert_not_called()

    def test_maximum_file_size_and_lambda_envelope_budget(self):
        raw = PNG_BYTES + b"x" * (handler.MAX_IMAGE_BYTES - len(PNG_BYTES))
        result, _ = self.invoke(files=[("images", ("menu.png", raw, "image/png"))])
        self.assertEqual(result["statusCode"], 200)
        uploaded = json.loads(self.requests[-1].content)["input"][0]["content"][1]["image_url"]
        self.assertEqual(base64.b64decode(uploaded.split(",", 1)[1]), raw)
        # Reserve 128 KiB beyond the maximum body for normal proxy metadata.
        self.assertLess(4 * ((handler.MAX_BODY_BYTES + 2) // 3) + 128 * 1024, 6 * 1024 * 1024)
        self.last_event["requestContext"] = {"padding": "x" * (128 * 1024)}
        self.assertLess(len(json.dumps(self.last_event).encode()), 6 * 1024 * 1024)

    def test_truncated_or_mismatched_multipart_is_rejected(self):
        event = upload_event()
        raw = base64.b64decode(event["body"])
        for broken in (b"", raw[:-10], b"invalid-body"):
            self.assert_error(400, "invalid_multipart", body=base64.b64encode(broken).decode(),
                              encoded=True, headers=event["headers"])
        self.assert_error(400, "invalid_multipart", headers={"Content-Type": "multipart/form-data; boundary=wrong"})
        self.assert_error(400, "invalid_multipart", headers={"Content-Type": "multipart/form-data; boundary=" + "x" * 71})

    def test_part_headers_and_transfer_encoding_are_bounded(self):
        event = upload_event()
        original = base64.b64decode(event["body"])
        for insertion in (b"Content-Transfer-Encoding: base64\r\n", b"Content-Type: image/png\r\n",
                          b"X-Long: " + b"x" * 5000 + b"\r\n",
                          b"".join(f"X-{i}: value\r\n".encode() for i in range(10))):
            raw = original.replace(b"Content-Type: image/png\r\n", insertion + b"Content-Type: image/png\r\n", 1)
            self.assert_error(400, "invalid_multipart", body=base64.b64encode(raw).decode(), encoded=True,
                              headers=event["headers"])

    def test_filenames_are_not_echoed_or_used_as_paths(self):
        result, menu = self.invoke(files=[("images", ("../../private-menu.png", PNG_BYTES, "image/png"))])
        self.assertEqual(result["statusCode"], 200)
        self.assertNotIn("private-menu", json.dumps(menu))
        self.assertNotIn("private-menu", self.requests[-1].content.decode())

    def test_other_methods_rejected(self):
        for method in ("GET", "PUT", "DELETE", "OPTIONS", None):
            result, _ = self.assert_error(405, "method_not_allowed", method=method)
            self.assertEqual(result["headers"]["Allow"], "POST,OPTIONS")

    def test_permissive_origin_and_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            result = handler.lambda_handler(upload_event(), None)
        self.assertEqual(result["headers"]["Access-Control-Allow-Origin"], "*")
        self.assertEqual(json.loads(result["body"])["error"]["code"], "transcription_disabled")
        self.openai.assert_not_called()
        self.aws.assert_not_called()

    def test_any_origin_allowed_despite_configured_origin(self):
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173", "https://other.example.com", "null"):
            headers = {"Content-Type": "application/json", "Origin": origin}
            with self.subTest(origin=origin):
                result, _ = self.invoke({"dummy": True}, headers=headers)
                self.assertEqual(result["statusCode"], 200)
                self.assert_error(400, "invalid_request", payload={"dummy": False}, headers=headers)
        self.openai.assert_not_called()
        self.aws.assert_not_called()

    def test_defaults_ignore_environment_overrides(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4.1", "OPENAI_REASONING_EFFORT": "high",
                                     "OPENAI_SERVICE_TIER": "default"}):
            self.invoke()
        sent = json.loads(self.requests[-1].content)
        self.assert_provider_settings(sent)

    def test_configured_settings_are_used_on_successive_requests(self):
        for count in (1, 2, 5):
            with self.subTest(image_count=count):
                before = len(self.requests)
                result, _ = self.invoke(files=[PNG_FILE] * count)
                self.assertEqual(result["statusCode"], 200)
                self.assertEqual(len(self.requests), before + 1)
                sent = json.loads(self.requests[-1].content)
                self.assert_provider_settings(sent)
                self.assertTrue(sent["text"]["format"]["strict"])
                self.assertEqual(len(sent["input"][0]["content"]), count + 1)

    def test_six_parts_with_dummy_after_files_and_seven_parts_rejected(self):
        files = [PNG_FILE] * 5 + [("dummy", (None, "false"))]
        result, _ = self.invoke(files=files)
        self.assertEqual(result["statusCode"], 200)
        sent = json.loads(self.requests[-1].content)
        self.assert_provider_settings(sent)
        self.assertEqual(len(sent["input"][0]["content"]), 6)
        self.assertEqual(len(self.requests), 1)
        with patch.object(openai_provider, "transcribe_menu") as transcribe:
            self.assert_error(400, "invalid_request", files=files + [PNG_FILE])
        transcribe.assert_not_called()
        self.openai.assert_called_once()
        self.assertEqual(len(self.requests), 1)
        self.aws.assert_not_called()

    def test_menu_detection_errors(self):
        for status, code in (("no_menu", "no_menu_found"), ("illegible_text", "illegible_text")):
            self.http_body = self.response({"status": status, "restaurantName": "", "sections": []})
            self.assert_error(422, code)

    def test_restaurant_name_fallback_and_trimming(self):
        for name, expected in (("Indian Restaurant", "Indian Restaurant"), ("  Café  ", "Café"), (" ", "Restaurant")):
            self.http_body = self.response({**MENU, "restaurantName": name})
            _, data = self.invoke()
            self.assertEqual(data["restaurantName"], expected)

    def test_invalid_menu_output(self):
        cases = [None, [], {}, {**MENU, "status": "unknown"}, {**MENU, "restaurantName": 4},
                 {**MENU, "sections": []}, {**MENU, "sections": "wrong"}, {**MENU, "extra": True}]
        for section in (None, {}, {"name": "", "items": ["Coffee"]}, {"name": "Drinks", "items": []},
                        {"name": "Drinks", "items": [""]}, {"name": "Drinks", "items": [1]},
                        {"name": "Drinks", "items": [{"name": "Coffee"}]}):
            cases.append({**MENU, "sections": [section]})
        for data in cases:
            self.http_body = self.response(data)
            self.assert_error(502, "invalid_openai_response")

    def test_refusal_truncation_and_unexpected_status(self):
        for response_status, reason, refusal, status, code in (
            ("completed", None, "Refusal", 422, "transcription_refused"),
            ("incomplete", "content_filter", None, 422, "transcription_refused"),
            ("incomplete", "max_output_tokens", None, 502, "incomplete_transcription"),
            ("incomplete", None, None, 502, "invalid_openai_response"),
            ("failed", None, None, 502, "invalid_openai_response"),
            ("queued", None, None, 502, "invalid_openai_response"),
            ("in_progress", None, None, 502, "invalid_openai_response"),
            ("cancelled", None, None, 502, "invalid_openai_response"),
            (None, None, None, 502, "invalid_openai_response"),
        ):
            with self.subTest(response_status=response_status, reason=reason):
                # Even valid menu JSON must not turn an incomplete response into success.
                self.http_body = self.response(MENU, response_status, reason, refusal)
                self.assert_error(status, code)

    def test_missing_output_and_invalid_content(self):
        for output in ([], None, [None]):
            self.http_body = {**self.response(MENU), "output": output}
            self.assert_error(502, "invalid_openai_response")
        for content in (None, "not-json", "```json\n{}\n```", "[" * 2000):
            self.http_body = self.response(MENU)
            self.http_body["output"][0]["content"][0]["text"] = content
            self.assert_error(502, "invalid_openai_response")

    def test_malformed_success_envelopes(self):
        for data in (b"not-json", [], {}, {"output": [None]}, {"output": [{"content": None}]}):
            self.http_body = data
            self.assert_error(502, "invalid_openai_response")

    def test_reasoning_metadata_before_menu_text_is_ignored(self):
        self.http_body["output"].insert(0, {
            "id": "rs-test", "type": "reasoning", "summary": [],
            "content": [{"type": "reasoning_text", "text": "Not menu JSON"}],
            "encrypted_content": "opaque-reasoning",
        })
        result, menu = self.invoke(files=[PNG_FILE] * 2)
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(menu["sections"], MENU["sections"])
        self.assertEqual(len(self.requests), 1)

    def test_text_blocks_and_messages_are_combined_in_order(self):
        text = json.dumps(MENU)
        for separate_messages in (False, True):
            with self.subTest(separate_messages=separate_messages):
                self.http_body = self.response(MENU)
                message = self.http_body["output"][0]
                message["content"][0]["text"] = text[:20]
                remaining = {"type": "output_text", "text": text[20:], "annotations": []}
                if separate_messages:
                    self.http_body["output"].append({**message, "id": "msg-second", "content": [remaining]})
                else:
                    message["content"].append(remaining)
                result, menu = self.invoke()
                self.assertEqual(result["statusCode"], 200)
                self.assertEqual(menu["sections"], MENU["sections"])

    def test_refusal_alongside_valid_menu_is_not_ignored(self):
        self.http_body["output"][0]["content"].append({"type": "refusal", "refusal": "PRIVATE-REFUSAL"})
        result, _ = self.assert_error(422, "transcription_refused")
        self.assertNotIn("PRIVATE-REFUSAL", result["body"])
        self.assertNotIn("PRIVATE-REFUSAL", str(self.log.mock_calls))

    def test_refusal_takes_priority_over_truncation_and_malformed_text(self):
        for status, reason in (("incomplete", "max_output_tokens"), ("completed", None)):
            with self.subTest(status=status):
                self.http_body = self.response(MENU, status, reason)
                self.http_body["output"][0]["content"][0]["text"] = None
                self.http_body["output"][0]["content"].append({"type": "refusal", "refusal": "Refusal"})
                self.assert_error(422, "transcription_refused")

    def test_incomplete_response_without_text_preserves_error_mapping(self):
        for reason, status, code in (("content_filter", 422, "transcription_refused"),
                                     ("max_output_tokens", 502, "incomplete_transcription")):
            self.http_body = {**self.response(MENU, "incomplete", reason), "output": []}
            self.assert_error(status, code)

    def test_response_errors_are_sanitized_without_retries(self):
        for code, status, expected in (
            ("server_error", 502, "openai_failure"),
            ("rate_limit_exceeded", 503, "openai_rate_limited"),
            ("invalid_image", 400, "invalid_image"),
            ("insufficient_quota", 503, "openai_not_funded"),
        ):
            with self.subTest(code=code):
                self.http_body = self.response(MENU, "failed")
                self.http_body["error"] = {"code": code, "message": "SECRET-UPSTREAM-MESSAGE"}
                before = len(self.requests)
                result, _ = self.assert_error(status, expected)
                self.assertEqual(len(self.requests), before + 1)
                self.assertNotIn("SECRET-UPSTREAM-MESSAGE", result["body"])
                self.assertNotIn("SECRET-UPSTREAM-MESSAGE", str(self.log.mock_calls))

    def test_unexpected_output_items_are_rejected(self):
        for item in (None, {}, {"type": "unknown"},
                     {"id": "fc-test", "type": "function_call", "name": "unexpected",
                      "call_id": "call-test", "arguments": "{}"}):
            with self.subTest(item=item):
                self.http_body = self.response(MENU)
                self.http_body["output"].append(item)
                self.assert_error(502, "invalid_openai_response")

    def test_malformed_output_messages_are_rejected(self):
        for changes in ({"status": "in_progress"}, {"status": "incomplete"}, {"status": None},
                        {"role": "user"}, {"content": None}, {"content": []},
                        {"content": [None]}, {"content": [{}]},
                        {"content": [{"type": "unknown", "text": json.dumps(MENU)}]},
                        {"content": [{"type": "output_text", "text": 123}]}):
            with self.subTest(changes=changes):
                self.http_body = self.response(MENU)
                self.http_body["output"][0].update(changes)
                self.assert_error(502, "invalid_openai_response")

    def test_reasoning_without_menu_text_is_rejected(self):
        self.http_body["output"] = [{"id": "rs-test", "type": "reasoning", "summary": []}]
        self.assert_error(502, "invalid_openai_response")

    def test_openai_http_errors_are_sanitized_without_retries(self):
        cases = [
            (429, "insufficient_quota", 503, "openai_not_funded"),
            (400, "billing_hard_limit_reached", 503, "openai_not_funded"),
            (429, "billing_not_active", 503, "openai_not_funded"),
            (429, "rate_limit_exceeded", 503, "openai_rate_limited"),
            (401, "invalid_api_key", 503, "openai_authentication_failed"),
            (403, None, 503, "openai_authentication_failed"),
            (400, "invalid_image", 400, "invalid_image"),
            (400, "invalid_image_format", 400, "invalid_image"),
            (400, "bad_schema", 502, "openai_failure"),
            (404, None, 502, "openai_failure"),
            (500, None, 502, "openai_failure"), (503, None, 502, "openai_failure"),
            (504, None, 504, "openai_timeout"),
        ]
        for upstream, code, status, expected in cases:
            with self.subTest(upstream=upstream, code=code):
                self.http_status = upstream
                self.http_body = {"error": {"code": code, "message": "SECRET-UPSTREAM-MESSAGE"}}
                before = len(self.requests)
                result, _ = self.assert_error(status, expected)
                self.assertEqual(len(self.requests), before + 1)
                self.assertNotIn("SECRET-UPSTREAM-MESSAGE", result["body"])
                self.assertNotIn("SECRET-UPSTREAM-MESSAGE", str(self.log.mock_calls))

    def test_billing_error_type_and_non_json_error(self):
        self.http_status = 429
        self.http_body = {"error": {"type": "insufficient_quota", "code": None}}
        self.assert_error(503, "openai_not_funded")
        self.http_status = 502
        self.http_body = b"private upstream error"
        self.assert_error(502, "openai_failure")

    def test_timeouts_and_connection_errors(self):
        for error, status, code in ((httpx.ReadTimeout("private"), 504, "openai_timeout"),
                                    (httpx.ConnectError("private"), 502, "openai_unavailable")):
            self.network_error = error
            self.assert_error(status, code)

    def test_remaining_lambda_time_caps_timeout(self):
        context = Mock()
        context.get_remaining_time_in_millis.return_value = 9000
        self.invoke(context=context)
        self.assertEqual(self.openai.call_args.kwargs["timeout"], min(openai_provider.TIMEOUT_SECONDS, 6))
        context.get_remaining_time_in_millis.return_value = 2000
        before = len(self.requests)
        self.assert_error(504, "openai_timeout", context=context)
        self.assertEqual(len(self.requests), before)

    def test_elapsed_deadline_cancels_pending_request(self):
        cancelled = []

        async def slow_request(*_):
            try:
                await asyncio.Future()
            finally:
                cancelled.append(True)

        with patch.object(openai_provider, "_request_response", side_effect=slow_request), \
            patch.object(openai_provider, "TIMEOUT_SECONDS", 0.01):
            self.assert_error(504, "openai_timeout")
        self.assertEqual(cancelled, [True])

    def test_missing_or_malformed_environment_key(self):
        for key in ("", "bad\nkey", "é"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
                self.assert_error(503, "api_not_configured")
        self.openai.assert_not_called()

    def configure_secret(self, value):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["MENU_API_SECRET_ID"] = "menu_api_secrets"
        self.aws.side_effect = None
        self.aws.return_value.get_secret_value.return_value = {"SecretString": value}

    def test_secret_fetch_cache_expiry_and_rotation(self):
        self.configure_secret('{"OPENAI_API_KEY": "first-test-key"}')
        with patch.object(shared.time, "monotonic", return_value=100):
            self.invoke()
            self.invoke()
        self.aws.assert_called_once()
        self.assertEqual(self.aws.call_args.kwargs["config"].retries["total_max_attempts"], 1)
        self.aws.return_value.get_secret_value.assert_called_once_with(SecretId="menu_api_secrets")
        self.aws.return_value.get_secret_value.return_value = {"SecretString": '{"OPENAI_API_KEY": "rotated-test-key"}'}
        with patch.object(shared.time, "monotonic", return_value=101 + shared.SECRET_CACHE_SECONDS):
            self.invoke()
        self.assertEqual(self.requests[-1].headers["authorization"], "Bearer rotated-test-key")
        self.assertEqual(self.aws.call_count, 2)

    def test_changing_secret_id_invalidates_cache(self):
        self.configure_secret('{"OPENAI_API_KEY": "first-test-key"}')
        self.invoke()
        os.environ["MENU_API_SECRET_ID"] = "other_secret"
        self.invoke()
        self.assertEqual(self.aws.call_count, 2)

    def test_shared_secret_uses_only_openai_key(self):
        self.configure_secret(json.dumps({
            "OPENAI_API_KEY": "openai-test-key",
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "GEMINI_API_KEY": "gemini-test-key",
            "GOOGLE_PLACES_API_KEY": "places-test-key",
            "OTHER_SECRET": {"unused": True},
        }))
        result, _ = self.invoke()
        self.assertEqual(result["statusCode"], 200)
        self.aws.return_value.get_secret_value.assert_called_once_with(SecretId="menu_api_secrets")
        self.assertEqual(self.requests[-1].headers["authorization"], "Bearer openai-test-key")
        self.assertEqual(len(self.requests), 1)

    def test_shared_secret_id_is_required_and_legacy_variables_are_ignored(self):
        os.environ.pop("OPENAI_API_KEY")
        os.environ["AI_SECRET_ID"] = "ignored-secret"
        os.environ["OPENAI_SECRET_ID"] = "ignored-secret"
        for secret_id in (None, ""):
            with self.subTest(secret_id=secret_id):
                if secret_id is None:
                    os.environ.pop("MENU_API_SECRET_ID", None)
                else:
                    os.environ["MENU_API_SECRET_ID"] = secret_id
                self.assert_error(503, "api_not_configured")
        self.aws.assert_not_called()
        self.openai.assert_not_called()

    def test_shared_secret_requires_openai_key(self):
        self.configure_secret('{"ANTHROPIC_API_KEY": "unused-test-key"}')
        self.assert_error(503, "api_not_configured")
        self.aws.return_value.get_secret_value.assert_called_once_with(SecretId="menu_api_secrets")
        self.openai.assert_not_called()

    def test_secret_failures_and_bad_values_are_safe(self):
        for value in ("not json", "{}", "[]", '{"OPENAI_API_KEY": null}', '{"OPENAI_API_KEY": ""}',
                      '{"OPENAI_API_KEY": "bad key"}', '{"ANTHROPIC_API_KEY": "unused-test-key"}'):
            self.configure_secret(value)
            self.assert_error(503, "api_not_configured")
        self.aws.return_value.get_secret_value.side_effect = RuntimeError("private-secret")
        result, _ = self.assert_error(503, "api_not_configured")
        self.assertNotIn("private-secret", result["body"])
        self.assertNotIn("private-secret", str(self.log.mock_calls))
        self.openai.assert_not_called()

    def test_environment_key_takes_precedence(self):
        os.environ["MENU_API_SECRET_ID"] = "not-read"
        self.invoke()
        self.aws.assert_not_called()

    def test_unexpected_errors_do_not_leak_details(self):
        with patch.object(openai_provider, "transcribe_menu", side_effect=RuntimeError("private-secret")):
            result, _ = self.assert_error(500, "internal_error")
        self.assertNotIn("private-secret", result["body"])
        self.assertNotIn("private-secret", str(self.log.mock_calls))


if __name__ == "__main__":
    unittest.main()