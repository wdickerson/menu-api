"""Offline contract tests using the real OpenAI SDK with a mocked HTTP transport."""

import asyncio
import base64
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import AsyncOpenAI


HANDLER_PATH = Path(__file__).resolve().parents[1] / "lambda" / "index.py"
spec = importlib.util.spec_from_file_location("menu_handler", HANDLER_PATH)
assert spec is not None and spec.loader is not None
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)

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
    request = httpx.Request("POST", "https://menu.example.com/menus",
                            files=parts or [("dummy", (None, "false"))])
    return {
        "httpMethod": "POST", "body": base64.b64encode(request.read()).decode("ascii"),
        "isBase64Encoded": True, "headers": {"Content-Type": request.headers["content-type"]},
    }


class MenuHandlerTests(unittest.TestCase):
    def setUp(self):
        self.start_patch(patch.dict(os.environ, {
            "ALLOWED_ORIGIN": "https://menu.example.com", "ENABLE_LIVE_TRANSCRIPTION": "true",
            "OPENAI_API_KEY": "test-key-not-a-real-secret",
        }, clear=True))
        self.start_patch(patch.object(handler, "_secret_cache", None))
        self.aws = self.start_patch(patch.object(handler.boto3, "client"))
        self.aws.side_effect = AssertionError("Unexpected AWS call")
        self.requests = []
        self.http_status = 200
        self.http_body = self.completion(MENU)
        self.network_error = None
        self.openai = self.start_patch(patch.object(handler.openai, "AsyncOpenAI", side_effect=self.make_client))
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
    def completion(data, finish_reason="stop", refusal=None):
        return {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "gpt-6-luna",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
                "role": "assistant", "content": json.dumps(data), "refusal": refusal,
            }}],
        }

    def invoke(self, payload=None, *, body=None, method="POST", encoded=None, headers=None,
               context=None, files=None, fields=None):
        if payload is not None or body is not None:
            event = {"body": json.dumps(payload) if body is None else body,
                     "isBase64Encoded": False, "headers": {"Content-Type": "application/json"}}
        else:
            event = upload_event(files, fields)
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

    def test_real_transcription_matches_frontend_contract(self):
        result, menu = self.invoke()
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(menu, {"isDummy": False, "restaurantName": MENU["restaurantName"], "sections": MENU["sections"]})
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer test-key-not-a-real-secret")
        sent = json.loads(request.content)
        self.assertEqual(sent["model"], "gpt-6-luna")
        self.assertEqual(sent["reasoning_effort"], "none")
        self.assertFalse(sent["store"])
        self.assertEqual(sent["max_completion_tokens"], 8000)
        self.assertTrue(sent["response_format"]["json_schema"]["strict"])
        self.assertEqual(sent["response_format"]["json_schema"]["schema"], handler.TRANSCRIPTION_SCHEMA)
        self.assertEqual(sent["messages"][1]["content"][1]["image_url"], {"url": PNG, "detail": "high"})
        self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
        self.aws.assert_not_called()

    def test_one_through_ten_images_use_one_request(self):
        for count in (1, 2, 10):
            before = len(self.requests)
            result, _ = self.invoke(files=[PNG_FILE] * count, fields=[("dummy", "false")])
            self.assertEqual(result["statusCode"], 200)
            self.assertEqual(len(self.requests), before + 1)
            self.assertEqual(len(json.loads(self.requests[-1].content)["messages"][1]["content"]), count + 1)

    def test_supported_image_signatures(self):
        for mime, raw in (("jpeg", b"\xff\xd8\xfftest"), ("png", b"\x89PNG\r\n\x1a\ntest"),
                          ("webp", b"RIFF1234WEBPtest")):
            result, _ = self.invoke(files=[("images", (f"menu.{mime}", raw, f"image/{mime}"))])
            self.assertEqual(result["statusCode"], 200)
            image = json.loads(self.requests[-1].content)["messages"][1]["content"][1]["image_url"]["url"]
            self.assertEqual(image, f"data:image/{mime};base64," + base64.b64encode(raw).decode())

    def test_missing_or_generic_mime_is_sniffed(self):
        for mime in (None, "application/octet-stream"):
            result, _ = self.invoke(files=[("images", ("menu", PNG_BYTES, mime))])
            self.assertEqual(result["statusCode"], 200)
            self.assertEqual(json.loads(self.requests[-1].content)["messages"][1]["content"][1]["image_url"]["url"], PNG)

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
        self.assert_error(400, "invalid_images", files=[PNG_FILE] * 11)
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
        for field in ("apiKey", "base_url", "imageUrl", "prompt", "reasoningEffort"):
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
        uploaded = json.loads(self.requests[-1].content)["messages"][1]["content"][1]["image_url"]["url"]
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
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4.1", "OPENAI_REASONING_EFFORT": "high"}):
            self.invoke()
        sent = json.loads(self.requests[-1].content)
        self.assertEqual(sent["model"], "gpt-6-luna")
        self.assertEqual(sent["reasoning_effort"], "none")

    def test_request_model_and_effort_overrides(self):
        result, _ = self.invoke(fields=[("model", "gpt-6.1-sol"), ("reasoning_effort", "low")])
        self.assertEqual(result["statusCode"], 200)
        sent = json.loads(self.requests[-1].content)
        self.assertEqual(sent["model"], "gpt-6.1-sol")
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertEqual(str(self.requests[-1].url), "https://api.openai.com/v1/chat/completions")
        self.assertEqual(len(self.requests), 1)
        # Settings never leak into a later request.
        self.invoke()
        sent = json.loads(self.requests[-1].content)
        self.assertEqual(sent["model"], "gpt-6-luna")
        self.assertEqual(sent["reasoning_effort"], "none")

    def test_options_default_independently(self):
        for fields, model, effort in (
            ([("model", "gpt-6.1-sol")], "gpt-6.1-sol", "none"),
            ([("reasoning_effort", "low")], "gpt-6-luna", "low"),
        ):
            result, _ = self.invoke(fields=fields)
            self.assertEqual(result["statusCode"], 200)
            sent = json.loads(self.requests[-1].content)
            self.assertEqual(sent["model"], model)
            self.assertEqual(sent["reasoning_effort"], effort)

    def test_gpt_41_mini_requires_none_and_omits_upstream_parameter(self):
        result, _ = self.invoke(fields=[("model", "gpt-4.1-mini"), ("reasoning_effort", "none")])
        self.assertEqual(result["statusCode"], 200)
        sent = json.loads(self.requests[-1].content)
        self.assertEqual(sent["model"], "gpt-4.1-mini")
        self.assertNotIn("reasoning_effort", sent)
        self.assertEqual(sent["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(sent["messages"][1]["content"][1]["image_url"]["url"], PNG)
        # Omitting it for Mini must not alter subsequent reasoning-model requests.
        self.invoke()
        sent = json.loads(self.requests[-1].content)
        self.assertEqual(sent["model"], "gpt-6-luna")
        self.assertEqual(sent["reasoning_effort"], "none")

    def test_gpt_41_mini_rejects_missing_or_non_none_effort(self):
        self.assert_error(400, "invalid_request", fields=[("model", "gpt-4.1-mini")])
        for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
            self.assert_error(400, "invalid_request", fields=[("model", "gpt-4.1-mini"),
                                                            ("reasoning_effort", effort)])
        self.openai.assert_not_called()
        self.aws.assert_not_called()

    def test_each_recognized_effort_is_forwarded(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            result, _ = self.invoke(fields=[("reasoning_effort", effort)])
            self.assertEqual(result["statusCode"], 200)
            self.assertEqual(json.loads(self.requests[-1].content)["reasoning_effort"], effort)

    def test_invalid_model_and_effort_are_rejected_before_paid_calls(self):
        for model in ("", " ", "gpt model", "gpt-6-luna\n", "https://example.com/model", "é", "a" * 129):
            self.assert_error(400, "invalid_request", fields=[("model", model)])
        for effort in ("", "None", "auto", "low ", "invalid", "é", "x" * 128):
            self.assert_error(400, "invalid_request", fields=[("reasoning_effort", effort)])
        self.openai.assert_not_called()
        self.aws.assert_not_called()

    def test_option_fields_cannot_be_duplicated_or_files(self):
        for key, value in (("model", "gpt-6-luna"), ("reasoning_effort", "none")):
            self.assert_error(400, "invalid_request", fields=[(key, value), (key, value)])
            self.assert_error(400, "invalid_request", files=[(key, ("option.txt", value.encode(), "text/plain"))])
        self.openai.assert_not_called()

    def test_all_thirteen_parts_and_options_after_files(self):
        files = [PNG_FILE] * 10 + [("dummy", (None, "false")),
                                   ("model", (None, "gpt-6.1-sol")),
                                   ("reasoning_effort", (None, "low"))]
        result, _ = self.invoke(files=files)
        self.assertEqual(result["statusCode"], 200)
        sent = json.loads(self.requests[-1].content)
        self.assertEqual(sent["model"], "gpt-6.1-sol")
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertEqual(len(sent["messages"][1]["content"]), 11)

    def test_dummy_accepts_valid_options_without_openai(self):
        result, menu = self.invoke(files=[], fields=[("dummy", "true"), ("model", "gpt-6.1-sol"),
                                                     ("reasoning_effort", "low")])
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(menu["restaurantName"], "Rough & Tumble")
        self.assert_error(400, "invalid_request", files=[], fields=[("dummy", "true"), ("reasoning_effort", "invalid")])
        self.openai.assert_not_called()
        self.aws.assert_not_called()

    def test_upstream_rejects_incompatible_or_unavailable_model_options(self):
        for param in ("model", "reasoning_effort"):
            self.http_status = 400
            self.http_body = {"error": {"code": "unsupported_value", "param": param,
                                        "message": "private-upstream-detail"}}
            result, _ = self.assert_error(400, "unsupported_model_options",
                                          fields=[("model", "gpt-6.1-sol"), ("reasoning_effort", "none")])
            self.assertNotIn("private-upstream-detail", result["body"])

    def test_menu_detection_errors(self):
        for status, code in (("no_menu", "no_menu_found"), ("illegible_text", "illegible_text")):
            self.http_body = self.completion({"status": status, "restaurantName": "", "sections": []})
            self.assert_error(422, code)

    def test_restaurant_name_fallback_and_trimming(self):
        for name, expected in (("Indian Restaurant", "Indian Restaurant"), ("  Café  ", "Café"), (" ", "Restaurant")):
            self.http_body = self.completion({**MENU, "restaurantName": name})
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
            self.http_body = self.completion(data)
            self.assert_error(502, "invalid_openai_response")

    def test_refusal_truncation_and_unexpected_finish(self):
        for reason, refusal, status, code in (
            ("stop", "Refusal", 422, "transcription_refused"),
            ("content_filter", None, 422, "transcription_refused"),
            ("length", None, 502, "incomplete_transcription"),
            ("tool_calls", None, 502, "invalid_openai_response"),
        ):
            self.http_body = self.completion(MENU, reason, refusal)
            self.assert_error(status, code)

    def test_missing_choices_and_invalid_content(self):
        self.http_body = {**self.completion(MENU), "choices": []}
        self.assert_error(502, "invalid_openai_response")
        for content in (None, "not-json", "```json\n{}\n```", "[" * 2000):
            self.http_body = self.completion(MENU)
            self.http_body["choices"][0]["message"]["content"] = content
            self.assert_error(502, "invalid_openai_response")

    def test_malformed_success_envelopes(self):
        for data in (b"not-json", [], {}, {"choices": [None]}, {"choices": [{"message": None}]}):
            self.http_body = data
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
            (404, "model_not_found", 400, "unsupported_model_options"),
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
        self.assertEqual(self.openai.call_args.kwargs["timeout"], 6)
        context.get_remaining_time_in_millis.return_value = 2000
        before = len(self.requests)
        self.assert_error(504, "openai_timeout", context=context)
        self.assertEqual(len(self.requests), before)

    def test_elapsed_deadline_cancels_pending_request(self):
        cancelled = []

        async def slow_request(*args):
            try:
                await asyncio.Future()
            finally:
                cancelled.append(True)

        with patch.object(handler, "request_completion", side_effect=slow_request), \
                patch.object(handler, "OPENAI_TIMEOUT_SECONDS", 0.01):
            self.assert_error(504, "openai_timeout")
        self.assertEqual(cancelled, [True])

    def test_missing_or_malformed_environment_key(self):
        for key in ("", "bad\nkey", "é"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
                self.assert_error(503, "api_not_configured")
        self.openai.assert_not_called()

    def configure_secret(self, value):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["OPENAI_SECRET_ID"] = "openai_menu"
        self.aws.side_effect = None
        self.aws.return_value.get_secret_value.return_value = {"SecretString": value}

    def test_secret_fetch_cache_expiry_and_rotation(self):
        self.configure_secret('{"OPENAI_API_KEY": "first-test-key"}')
        with patch.object(handler.time, "monotonic", return_value=100):
            self.invoke()
            self.invoke()
        self.aws.assert_called_once()
        self.assertEqual(self.aws.call_args.kwargs["config"].retries["total_max_attempts"], 1)
        self.aws.return_value.get_secret_value.assert_called_once_with(SecretId="openai_menu")
        self.aws.return_value.get_secret_value.return_value = {"SecretString": '{"OPENAI_API_KEY": "rotated-test-key"}'}
        with patch.object(handler.time, "monotonic", return_value=401):
            self.invoke()
        self.assertEqual(self.requests[-1].headers["authorization"], "Bearer rotated-test-key")
        self.assertEqual(self.aws.call_count, 2)

    def test_changing_secret_id_invalidates_cache(self):
        self.configure_secret('{"OPENAI_API_KEY": "first-test-key"}')
        self.invoke()
        os.environ["OPENAI_SECRET_ID"] = "other_secret"
        self.invoke()
        self.assertEqual(self.aws.call_count, 2)

    def test_secret_failures_and_bad_values_are_safe(self):
        for value in ("not json", "{}", "[]", '{"OPENAI_API_KEY": null}', '{"OPENAI_API_KEY": ""}',
                      '{"OPENAI_API_KEY": "bad key"}'):
            self.configure_secret(value)
            self.assert_error(503, "api_not_configured")
        self.aws.return_value.get_secret_value.side_effect = RuntimeError("private-secret")
        result, _ = self.assert_error(503, "api_not_configured")
        self.assertNotIn("private-secret", result["body"])
        self.assertNotIn("private-secret", str(self.log.mock_calls))
        self.openai.assert_not_called()

    def test_environment_key_takes_precedence(self):
        os.environ["OPENAI_SECRET_ID"] = "not-read"
        self.invoke()
        self.aws.assert_not_called()

    def test_unexpected_errors_do_not_leak_details(self):
        with patch.object(handler, "transcribe", side_effect=RuntimeError("private-secret")):
            result, _ = self.assert_error(500, "internal_error")
        self.assertNotIn("private-secret", result["body"])
        self.assertNotIn("private-secret", str(self.log.mock_calls))


if __name__ == "__main__":
    unittest.main()