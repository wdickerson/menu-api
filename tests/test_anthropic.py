"""Offline Anthropic contract tests using the real SDK and mocked HTTP transport."""

import asyncio
import base64
import json
import os
import unittest
from unittest.mock import Mock, patch

from anthropic import AsyncAnthropic
import httpx2

from test_index import MENU, PNG_BYTES, PNG_FILE, handler, openai_provider, shared, upload_event


anthropic_provider = handler.anthropic_provider


class AnthropicHandlerTests(unittest.TestCase):
    def setUp(self):
        self.start_patch(patch.dict(os.environ, {
            "ENABLE_LIVE_TRANSCRIPTION": "true", "ANTHROPIC_API_KEY": "anthropic-test-key",
            "OPENAI_API_KEY": "unused-openai-test-key",
        }, clear=True))
        self.start_patch(patch.object(handler, "TRANSCRIPTION_PROVIDER", "ANTHROPIC"))
        self.start_patch(patch.object(shared, "_api_key_cache", {}))
        self.aws = self.start_patch(patch.object(shared.boto3, "client"))
        self.aws.side_effect = AssertionError("Unexpected AWS call")
        self.openai = self.start_patch(patch.object(openai_provider.openai_sdk, "AsyncOpenAI"))
        self.openai.side_effect = AssertionError("Unexpected OpenAI call")
        self.requests = []
        self.http_status = 200
        self.http_body = self.completion(MENU)
        self.network_error = None
        self.anthropic = self.start_patch(patch.object(
            anthropic_provider.anthropic_sdk, "AsyncAnthropic", side_effect=self.make_client))
        self.log = self.start_patch(patch.object(handler, "logger"))

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def make_client(self, **kwargs):
        return AsyncAnthropic(http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(self.transport)), **kwargs)

    def transport(self, request):
        self.requests.append(request)
        if self.network_error:
            raise self.network_error
        if isinstance(self.http_body, bytes):
            return httpx2.Response(self.http_status, content=self.http_body)
        return httpx2.Response(self.http_status, json=self.http_body)

    @staticmethod
    def completion(data, stop_reason="end_turn"):
        return {
            "id": "msg-test", "type": "message", "role": "assistant",
            "model": "claude-haiku-5-5", "stop_reason": stop_reason,
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 20},
            "content": [{"type": "text", "text": json.dumps(data)}],
        }

    def invoke(self, *, files=None, fields=None, context=None, dummy_json=False):
        event = upload_event(files, fields)
        if dummy_json:
            event = {"httpMethod": "POST", "path": "/menu_image_transcribe",
                     "headers": {"Content-Type": "application/json"},
                     "body": '{"dummy":true}', "isBase64Encoded": False}
        response = handler.lambda_handler(event, context)
        self.assertEqual(response["headers"]["Cache-Control"], "no-store")
        self.assertEqual(response["headers"]["Access-Control-Allow-Origin"], "*")
        self.openai.assert_not_called()
        return response, json.loads(response["body"])

    def assert_error(self, status, code, **kwargs):
        response, data = self.invoke(**kwargs)
        self.assertEqual(response["statusCode"], status, data)
        self.assertEqual(data["error"]["code"], code)
        self.assertTrue(data["error"]["message"])
        return response

    def configure_secret(self, value):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["MENU_API_SECRET_ID"] = "menu_api_secrets"
        self.aws.side_effect = None
        self.aws.return_value.get_secret_value.return_value = {"SecretString": value}

    def test_real_sdk_request_and_frontend_response_contract(self):
        response, menu = self.invoke()
        self.assertEqual(response["statusCode"], 200, menu)
        self.assertEqual(menu, {"isDummy": False, "restaurantName": MENU["restaurantName"],
                                "sections": MENU["sections"]})
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages")
        self.assertEqual(request.headers["x-api-key"], "anthropic-test-key")
        self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
        sent = json.loads(request.content)
        self.assertEqual(sent["model"], "claude-haiku-5-5")
        self.assertEqual(sent["max_tokens"], 8000)
        self.assertEqual(anthropic_provider.SYSTEM_PROMPT, sent["system"])
        self.assertEqual(sent["output_config"]["format"], {
            "type": "json_schema", "schema": shared.MENU_TRANSCRIPTION_SCHEMA})
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertEqual(sent["messages"][0]["content"][1], {
            "type": "image", "source": {"type": "base64", "media_type": "image/png",
            "data": base64.b64encode(PNG_BYTES).decode("ascii")}})
        for option in ("reasoning_effort", "store", "response_format"):
            self.assertNotIn(option, sent)
        self.assertEqual(self.anthropic.call_args.kwargs["max_retries"], 0)
        self.aws.assert_not_called()

    def test_one_through_five_images_in_one_call(self):
        for count in range(1, 6):
            before = len(self.requests)
            response, menu = self.invoke(files=[PNG_FILE] * count)
            self.assertEqual(response["statusCode"], 200, menu)
            self.assertEqual(len(self.requests), before + 1)
            self.assertEqual(len(json.loads(self.requests[-1].content)["messages"][0]["content"]), count + 1)

    def test_two_images_with_thinking_and_menu_text(self):
        for metadata in (
            {"type": "thinking", "thinking": "", "signature": "test-signature"},
            {"type": "thinking", "thinking": "private-thinking", "signature": "test-signature"},
            {"type": "redacted_thinking", "data": "private-redacted-data"},
        ):
            with self.subTest(block_type=metadata["type"], metadata=metadata):
                self.http_body = self.completion(MENU)
                self.http_body["content"].insert(0, metadata)
                response, menu = self.invoke(files=[PNG_FILE] * 2)
                self.assertEqual(response["statusCode"], 200, menu)
                self.assertEqual(menu, {"isDummy": False, "restaurantName": MENU["restaurantName"],
                                       "sections": MENU["sections"]})
                self.assertNotIn("private-thinking", response["body"])
                self.assertNotIn("private-redacted-data", response["body"])

    def test_thinking_without_menu_text_is_invalid(self):
        self.http_body = self.completion(MENU)
        self.http_body["content"] = [
            {"type": "thinking", "thinking": "", "signature": "test-signature"},
            {"type": "redacted_thinking", "data": "test-data"},
        ]
        self.assert_error(502, "invalid_anthropic_response", files=[PNG_FILE] * 2)

    def test_unexpected_content_block_with_menu_is_invalid(self):
        self.http_body = self.completion(MENU)
        self.http_body["content"].insert(0, {
            "type": "tool_use", "id": "tool-test", "name": "unexpected", "input": {},
        })
        self.assert_error(502, "invalid_anthropic_response")

    def test_image_types_and_bytes_preserved(self):
        for mime, raw in (("jpeg", b"\xff\xd8\xfftest"), ("png", PNG_BYTES),
                          ("webp", b"RIFF1234WEBPtest")):
            response, _ = self.invoke(files=[("images", ("menu", raw, f"image/{mime}"))])
            self.assertEqual(response["statusCode"], 200)
            source = json.loads(self.requests[-1].content)["messages"][0]["content"][1]["source"]
            self.assertEqual(source["media_type"], f"image/{mime}")
            self.assertEqual(base64.b64decode(source["data"]), raw)

    def test_dummy_requires_neither_provider_nor_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            for kwargs in ({"dummy_json": True}, {"files": [], "fields": [("dummy", "true")]}):
                response, menu = self.invoke(**kwargs)
                self.assertEqual(response["statusCode"], 200)
                self.assertTrue(menu["isDummy"])
        self.aws.assert_not_called()
        self.anthropic.assert_not_called()

    def test_disabled_live_mode_and_bad_images_skip_paid_calls(self):
        with patch.dict(os.environ, {"ENABLE_LIVE_TRANSCRIPTION": "false"}):
            self.assert_error(503, "transcription_disabled")
        self.assert_error(400, "invalid_images", files=[])
        self.assert_error(400, "invalid_images", files=[PNG_FILE] * 6)
        self.assert_error(400, "invalid_image", files=[("images", ("bad", b"bad", "image/png"))])
        self.aws.assert_not_called()
        self.anthropic.assert_not_called()

    def test_provider_and_model_are_code_owned(self):
        with patch.dict(os.environ, {"TRANSCRIPTION_PROVIDER": "OPENAI", "ANTHROPIC_MODEL": "ignored"}):
            response, _ = self.invoke()
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(self.requests[-1].content)["model"], anthropic_provider.DEFAULT_MODEL)
        self.assert_error(400, "invalid_request", fields=[("provider", "OPENAI")])

    def test_changed_code_model_is_used_on_successive_requests_and_restored(self):
        original_model = anthropic_provider.DEFAULT_MODEL
        with patch.object(anthropic_provider, "DEFAULT_MODEL", "claude-sonnet-4-6"):
            for _ in range(2):
                before = len(self.requests)
                response, _ = self.invoke()
                self.assertEqual(response["statusCode"], 200)
                self.assertEqual(len(self.requests), before + 1)
                self.assertEqual(json.loads(self.requests[-1].content)["model"], "claude-sonnet-4-6")
        response, _ = self.invoke()
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(self.requests[-1].content)["model"], original_model)

    def test_invalid_code_provider_fails_safely(self):
        with patch.object(handler, "TRANSCRIPTION_PROVIDER", "OTHER"):
            self.assert_error(503, "api_not_configured")
        self.aws.assert_not_called()
        self.anthropic.assert_not_called()

    def test_secret_reads_selected_field_and_caches(self):
        self.configure_secret(json.dumps({"ANTHROPIC_API_KEY": "secret-anthropic-key",
                                          "OPENAI_API_KEY": "secret-openai-key"}))
        self.invoke()
        self.invoke()
        self.aws.return_value.get_secret_value.assert_called_once_with(SecretId="menu_api_secrets")
        self.assertEqual(self.requests[-1].headers["x-api-key"], "secret-anthropic-key")

    def test_cached_openai_key_is_not_used_for_anthropic(self):
        self.configure_secret(json.dumps({"ANTHROPIC_API_KEY": "secret-anthropic-key",
                                          "OPENAI_API_KEY": "secret-openai-key"}))
        os.environ.pop("OPENAI_API_KEY")
        self.assertEqual(shared.get_api_key("OPENAI_API_KEY"), "secret-openai-key")
        self.invoke()
        self.assertEqual(self.requests[-1].headers["x-api-key"], "secret-anthropic-key")
        self.assertEqual(self.aws.call_count, 2)
        # Switching back to OpenAI reuses only its own cached key.
        self.assertEqual(shared.get_api_key("OPENAI_API_KEY"), "secret-openai-key")
        self.assertEqual(self.aws.call_count, 2)

    def test_cache_expiry_rotation_and_secret_id(self):
        self.configure_secret('{"ANTHROPIC_API_KEY":"first-key"}')
        with patch.object(shared.time, "monotonic", return_value=100):
            self.invoke()
            self.invoke()
        self.aws.return_value.get_secret_value.return_value = {"SecretString": '{"ANTHROPIC_API_KEY":"rotated-key"}'}
        with patch.object(shared.time, "monotonic", return_value=401):
            self.invoke()
        self.assertEqual(self.requests[-1].headers["x-api-key"], "rotated-key")
        os.environ["MENU_API_SECRET_ID"] = "other-secret"
        self.invoke()
        self.assertEqual(self.aws.call_count, 3)

    def test_missing_malformed_and_unavailable_secrets(self):
        for value in ("not json", "[]", "{}", '{"OPENAI_API_KEY":"wrong-provider"}',
                      '{"ANTHROPIC_API_KEY":null}', '{"ANTHROPIC_API_KEY":""}',
                      '{"ANTHROPIC_API_KEY":"bad key"}'):
            self.configure_secret(value)
            self.assert_error(503, "api_not_configured")
        self.aws.return_value.get_secret_value.side_effect = RuntimeError("private-secret")
        response = self.assert_error(503, "api_not_configured")
        self.assertNotIn("private-secret", response["body"])
        self.assertNotIn("private-secret", str(self.log.mock_calls))
        self.anthropic.assert_not_called()

    def test_local_key_precedence_and_no_openai_fallback(self):
        os.environ["MENU_API_SECRET_ID"] = "not-read"
        self.invoke()
        self.aws.assert_not_called()
        os.environ.pop("MENU_API_SECRET_ID")
        for key in ("", "bad\nkey", "é"):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": key}):
                self.assert_error(503, "api_not_configured")
        self.assertEqual(len(self.requests), 1)

    def test_detection_errors_and_placeholder(self):
        for status, code in (("no_menu", "no_menu_found"), ("illegible_text", "illegible_text")):
            self.http_body = self.completion({"status": status, "restaurantName": "", "sections": []})
            self.assert_error(422, code)
        self.http_body = self.completion({**MENU, "restaurantName": " "})
        _, menu = self.invoke()
        self.assertEqual(menu["restaurantName"], "Restaurant")

    def test_invalid_menu_outputs(self):
        for data in (None, [], {}, {**MENU, "sections": []},
                     {**MENU, "sections": [{"name": "Drinks", "items": [1]}]}):
            self.http_body = self.completion(data)
            self.assert_error(502, "invalid_anthropic_response")

    def test_empty_sections_are_returned_in_order(self):
        empty = {"name": "Monday-Friday 3-6", "items": []}
        for sections in ([empty], [empty, *MENU["sections"], empty]):
            with self.subTest(sections=sections):
                self.http_body = self.completion({**MENU, "sections": sections})
                response, menu = self.invoke()
                self.assertEqual(response["statusCode"], 200, menu)
                self.assertEqual(menu["sections"], sections)

    def test_invalid_envelopes_and_content(self):
        for body in (b"not-json", [], {}, {**self.completion(MENU), "content": None},
                     {**self.completion(MENU), "content": []},
                     {**self.completion(MENU), "content": [None]},
                     {**self.completion(MENU), "content": [{"type": "text", "text": None}]}):
            self.http_body = body
            self.assert_error(502, "invalid_anthropic_response")
        for text in ("not json", "```json\n{}\n```", "[" * 2000):
            self.http_body = self.completion(MENU)
            self.http_body["content"][0]["text"] = text
            self.assert_error(502, "invalid_anthropic_response")

    def test_refusal_truncation_and_unexpected_stop(self):
        for reason, status, code in (("refusal", 422, "transcription_refused"),
                                    ("max_tokens", 502, "incomplete_transcription"),
                                    ("model_context_window_exceeded", 502, "incomplete_transcription"),
                                    ("tool_use", 502, "invalid_anthropic_response"),
                                    (None, 502, "invalid_anthropic_response")):
            self.http_body = self.completion(MENU, reason)
            self.assert_error(status, code)

    def test_http_errors_are_sanitized_without_retries(self):
        for upstream, error_type, message, status, code in (
            (400, "invalid_request_error", "Your credit balance is too low", 503, "anthropic_not_funded"),
            (400, "billing_error", "private", 503, "anthropic_not_funded"),
            (401, "authentication_error", "private", 503, "anthropic_authentication_failed"),
            (403, "permission_error", "private", 503, "anthropic_authentication_failed"),
            (429, "rate_limit_error", "private", 503, "anthropic_rate_limited"),
            (400, "invalid_request_error", "invalid image", 400, "invalid_image"),
            (413, "request_too_large", "private", 400, "invalid_image"),
            (404, "not_found_error", "private", 503, "anthropic_model_unavailable"),
            (400, "invalid_request_error", "private", 502, "anthropic_failure"),
            (500, "api_error", "private", 502, "anthropic_failure"),
            (529, "overloaded_error", "private", 503, "anthropic_unavailable"),
            (504, "api_error", "private", 504, "anthropic_timeout"),
        ):
            self.http_status = upstream
            self.http_body = {"type": "error", "error": {"type": error_type,
                              "message": message + " PRIVATE-UPSTREAM"}}
            before = len(self.requests)
            response = self.assert_error(status, code)
            self.assertEqual(len(self.requests), before + 1)
            self.assertNotIn("PRIVATE-UPSTREAM", response["body"])
            self.assertNotIn("PRIVATE-UPSTREAM", str(self.log.mock_calls))

    def test_network_errors_and_non_json_error(self):
        for error, status, code in ((httpx2.ReadTimeout("private"), 504, "anthropic_timeout"),
                                    (httpx2.ConnectError("private"), 502, "anthropic_unavailable")):
            self.network_error = error
            self.assert_error(status, code)
        self.network_error = None
        self.http_status = 502
        self.http_body = b"private upstream error"
        self.assert_error(502, "anthropic_failure")

    def test_remaining_lambda_time_caps_timeout(self):
        context = Mock()
        context.get_remaining_time_in_millis.return_value = 9000
        self.invoke(context=context)
        self.assertEqual(self.anthropic.call_args.kwargs["timeout"], 6)
        context.get_remaining_time_in_millis.return_value = 2000
        self.assert_error(504, "anthropic_timeout", context=context)
        self.assertEqual(len(self.requests), 1)

    def test_elapsed_deadline_cancels_request(self):
        cancelled = []

        async def slow_request(*_):
            try:
                await asyncio.Future()
            finally:
                cancelled.append(True)

        with patch.object(anthropic_provider, "_request_message", side_effect=slow_request), \
            patch.object(anthropic_provider, "TIMEOUT_SECONDS", 0.01):
            self.assert_error(504, "anthropic_timeout")
        self.assertEqual(cancelled, [True])


if __name__ == "__main__":
    unittest.main()