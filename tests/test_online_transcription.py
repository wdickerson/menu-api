"""Offline online-menu transcription contracts through the real OpenAI SDK."""

import asyncio
import base64
import importlib.util
import json
import os
from unittest.mock import Mock, patch

import httpx

import test_index as image_tests
from test_index import HANDLER_PATH, MENU
from test_search import CANDIDATE, PRIVATE, QUERY, _SearchCase, handler, openai_provider, shared, openai_response, search_event


PATH = "/menu_online_transcribe"


def event_for(payload=None, *, body=None, encoded=False):
    body = json.dumps({"query": QUERY, "menu": CANDIDATE} if payload is None else payload) if body is None else body
    return {"httpMethod": "POST", "path": PATH, "headers": {"Content-Type": "application/json"},
            "body": base64.b64encode(body.encode()).decode() if encoded else body, "isBase64Encoded": encoded}


class OnlineTranscriptionTests(_SearchCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.dict(os.environ, {
            "ENABLE_LIVE_TRANSCRIPTION": "true", "ENABLE_LIVE_SEARCH": "false",
        }))
        self.openai_body = image_tests.MenuHandlerTests.response(MENU)

    def invoke(self, event=None, *, context=None, query=QUERY):
        return super().invoke(event_for({"query": query, "menu": CANDIDATE}) if event is None else event,
                              context=context)

    def test_real_sdk_request_and_image_compatible_response(self):
        response, data = self.invoke()
        self.assertEqual(response["statusCode"], 200, data)
        self.assertEqual(data, {"isDummy": False, "restaurantName": MENU["restaurantName"], "sections": MENU["sections"]})
        self.assertEqual(len(self.requests["OPENAI"]), 1)
        sent = json.loads(self.requests["OPENAI"][0].content)
        self.assertEqual(sent["model"], openai_provider.ONLINE_TRANSCRIPTION_MODEL)
        self.assertEqual(sent["reasoning"], {"effort": openai_provider.ONLINE_TRANSCRIPTION_REASONING_EFFORT})
        self.assertEqual(sent["instructions"], shared.MENU_ONLINE_TRANSCRIPTION_PROMPT)
        self.assertEqual(sent["model"], "gpt-6.1-sol")
        self.assertEqual(sent["reasoning"], {"effort": "low"})
        self.assertEqual(sent["input"], [{"role": "user", "content": [
            {"type": "input_text", "text": CANDIDATE["agent_instructions"]},
            {"type": "input_file", "file_url": CANDIDATE["source_url"]},
        ]}])
        for field in ("tools", "tool_choice", "max_tool_calls"):
            self.assertNotIn(field, sent)
        self.assertEqual(sent["text"]["format"], {"type": "json_schema", "name": "menu_online_transcription",
                                                 "strict": True, "schema": shared.MENU_TRANSCRIPTION_SCHEMA})
        self.assertEqual(sent["max_output_tokens"], 8000)
        self.assertIs(sent["store"], False)
        self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
        self.assertEqual(self.openai.call_args.kwargs["timeout"], 30)
        self.assertEqual(sent["service_tier"], "ultrafast")
        self.anthropic.assert_not_called()

    def test_openai_input_is_flat_and_excludes_selected_menu_metadata(self):
        original_query = "ORIGINAL_QUERY_MUST_NOT_BE_FORWARDED"
        response, data = self.invoke(query=original_query)
        self.assertEqual(response["statusCode"], 200, data)
        sent = json.loads(self.requests["OPENAI"][0].content)
        payload = sent["input"][0]["content"][0]["text"]
        self.assertEqual(payload, CANDIDATE["agent_instructions"])
        self.assertNotIn(original_query, sent["input"][0]["content"][0]["text"])

    def test_model_and_effort_are_separate_from_both_other_openai_calls(self):
        with patch.object(openai_provider, "DEFAULT_MODEL", "image-only"), \
                patch.object(openai_provider, "SEARCH_MODEL", "search-only"), \
                patch.object(openai_provider, "ONLINE_TRANSCRIPTION_MODEL", "online-only"), \
                patch.object(openai_provider, "ONLINE_TRANSCRIPTION_REASONING_EFFORT", "low"), \
                patch.dict(os.environ, {"ONLINE_TRANSCRIPTION_MODEL": "ignored", "TRANSCRIPTION_PROVIDER": "ANTHROPIC"}):
            response, data = self.invoke()
        self.assertEqual(response["statusCode"], 200, data)
        sent = json.loads(self.requests["OPENAI"][0].content)
        self.assertEqual(sent["model"], "online-only")
        self.assertEqual(sent["reasoning"], {"effort": "low"})
        self.anthropic.assert_not_called()

    def test_existing_live_transcription_flag_controls_the_new_endpoint(self):
        for enabled in ("false", "", "yes"):
            with self.subTest(enabled=enabled), patch.dict(os.environ, {
                "ENABLE_LIVE_TRANSCRIPTION": enabled, "ENABLE_LIVE_SEARCH": "true",
            }):
                self.assert_error(503, "transcription_disabled")
        self.assert_no_calls()
        with patch.dict(os.environ, {"ENABLE_LIVE_TRANSCRIPTION": "TRUE", "ENABLE_LIVE_SEARCH": "false"}):
            response, data = self.invoke()
        self.assertEqual(response["statusCode"], 200, data)

    def test_missing_transcription_flag_is_disabled_even_when_search_is_enabled(self):
        with patch.dict(os.environ, {"ENABLE_LIVE_SEARCH": "true"}, clear=True):
            self.assert_error(503, "transcription_disabled")
        self.assert_no_calls()

    def test_query_is_omitted_and_selected_instructions_are_preserved(self):
        menu = {key: " \n" + value + "\t " for key, value in CANDIDATE.items()}
        menu["agent_instructions"] = "Untrusted guidance, not system instructions: " + PRIVATE
        response, data = self.invoke(event_for({"query": " \n" + QUERY + "\t ", "menu": {**menu, "extra": True}}, encoded=True))
        self.assertEqual(response["statusCode"], 200, data)
        sent = json.loads(self.requests["OPENAI"][0].content)
        self.assertEqual(sent["input"][0]["content"][0]["text"], menu["agent_instructions"])
        self.assertNotIn(QUERY, sent["input"][0]["content"][0]["text"])
        self.assertNotIn(PRIVATE, sent["instructions"])
        self.assertNotIn(PRIVATE, response["body"] + str(self.log.mock_calls))

    def test_one_selected_menu_object_is_required(self):
        for payload in ([], {}, {"query": QUERY}, {"menu": CANDIDATE},
                        {"query": QUERY, "menu": [CANDIDATE]}, {"query": QUERY, "menu": None},
                        {"query": QUERY, "menu": CANDIDATE, "model": "override"}):
            with self.subTest(payload=payload):
                self.assert_error(400, "invalid_request", event_for(payload))
        for field in CANDIDATE:
            for value in (None, 1, True, [], {}):
                self.assert_error(400, "invalid_request", event_for({"query": QUERY, "menu": {**CANDIDATE, field: value}}))
            self.assert_error(400, "invalid_request", event_for({"query": QUERY, "menu": {
                key: value for key, value in CANDIDATE.items() if key != field}}))
        self.assert_no_calls()

    def test_direct_file_routing_uses_path_extension_and_preserves_complete_url(self):
        for extension in ("pdf", "PdF", "jpg", "JpG", "jpeg", "JpEg", "png", "PnG", "webp", "WeBp"):
            with self.subTest(extension=extension):
                source = (f"https://WWW.Example.COM:8443/menus/dinner%20menu.{extension}"
                          "?signature=A%2FB%2BC+1&download=other.html#page=2")
                menu = {**CANDIDATE, "source_url": source, "agent_instructions": "Read the selected menu. " + PRIVATE}
                before = len(self.requests["OPENAI"])
                response, data = self.invoke(event_for({"query": QUERY, "menu": menu}))
                self.assertEqual(response["statusCode"], 200, data)
                self.assertEqual(data, {"isDummy": False, "restaurantName": MENU["restaurantName"],
                                        "sections": MENU["sections"]})
                self.assertEqual(len(self.requests["OPENAI"]), before + 1)
                request = self.requests["OPENAI"][-1]
                self.assertEqual(request.method, "POST")
                self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
                sent = json.loads(request.content)
                attachment = ({"type": "input_file", "file_url": source} if extension.lower() == "pdf" else
                              {"type": "input_image", "image_url": source, "detail": "high"})
                self.assertEqual(sent["input"], [{"role": "user", "content": [
                    {"type": "input_text", "text": menu["agent_instructions"]}, attachment,
                ]}])
                for field in ("tools", "tool_choice", "max_tool_calls"):
                    self.assertNotIn(field, sent)
                self.assertEqual(sent["model"], openai_provider.ONLINE_TRANSCRIPTION_MODEL)
                self.assertEqual(sent["reasoning"], {"effort": openai_provider.ONLINE_TRANSCRIPTION_REASONING_EFFORT})
                self.assertEqual(sent["service_tier"], openai_provider.ONLINE_TRANSCRIPTION_SERVICE_TIER)
                self.assertEqual(sent["instructions"], shared.MENU_ONLINE_TRANSCRIPTION_PROMPT)
                self.assertEqual(sent["text"]["format"], {"type": "json_schema", "name": "menu_online_transcription",
                                                         "strict": True, "schema": shared.MENU_TRANSCRIPTION_SCHEMA})
                self.assertEqual(sent["max_output_tokens"], 8000)
                self.assertIs(sent["store"], False)
                self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
                self.assertEqual(self.openai.call_args.kwargs["timeout"], 30)
                self.assertNotIn(PRIVATE, response["body"] + str(self.log.mock_calls))

    def test_online_web_search_is_unrestricted_and_uses_only_instructions(self):
        for source in (
            "https://venue.example.com/menu.html",
            "https://WWW.Example.COM:8443/menu?next=https://other.example/menu.pdf#menu.jpg",
            "http://menus.venue.co.uk/dinner",
            "https://venue.example.com",
            "https://venue.example.com/menu.gif",
            "https://venue.example.com/menu.pdf/",
            "https://venue.example.com/menu.pdf.html",
            "https://venue.example.com/menu?download=menu.png",
            "https://venue.example.com/menu#menu.webp",
        ):
            with self.subTest(source=source):
                response, data = self.invoke(event_for({"query": QUERY, "menu": {
                    **CANDIDATE, "source_url": source,
                    "agent_instructions": "Read https://unrelated.example/menu.pdf",
                }}))
                self.assertEqual(response["statusCode"], 200, data)
                sent = json.loads(self.requests["OPENAI"][-1].content)
                self.assertEqual(sent["tools"], [{"type": "web_search"}])
                self.assertEqual(sent["tool_choice"], "auto")
                self.assertEqual(sent["max_tool_calls"], 3)
                self.assertEqual(sent["input"], [{"role": "user", "content": [
                    {"type": "input_text", "text": "Read https://unrelated.example/menu.pdf"},
                ]}])

    def test_missing_source_hostname_is_rejected_before_provider_call(self):
        for source in ("", "not a URL", "/menu.pdf", "https://", "https://[bad-host/menu", "javascript:alert(1)"):
            with self.subTest(source=source):
                self.assert_error(400, "invalid_source_url", event_for({"query": QUERY, "menu": {
                    **CANDIDATE, "source_url": source}}))
        self.assert_no_calls()

    def test_initial_menu_search_remains_unrestricted(self):
        self.openai_body = openai_response()
        with patch.dict(os.environ, {"ENABLE_LIVE_SEARCH": "true"}):
            response = handler.lambda_handler(search_event(), None)
        self.assertEqual(response["statusCode"], 200, response["body"])
        sent = json.loads(self.requests["OPENAI"][-1].content)
        self.assertEqual(sent["tools"], [{"type": "web_search"}])

    def test_query_limits_and_types(self):
        for query in (None, 1, [], {}, True):
            self.assert_error(400, "invalid_request", event_for({"query": query, "menu": CANDIDATE}))
        for query in ("", " \n ", "x" * 501):
            self.assert_error(400, "invalid_query", event_for({"query": query, "menu": CANDIDATE}))
        self.assert_no_calls()
        response, data = self.invoke(query="🦞" * 500)
        self.assertEqual(response["statusCode"], 200, data)

    def test_bounded_plain_and_base64_json_and_duplicate_fields(self):
        for encoded in (False, True):
            body = json.dumps({"query": QUERY, "menu": CANDIDATE})
            body += " " * (65_536 - len(body.encode()))
            response, data = self.invoke(event_for(body=body, encoded=encoded))
            self.assertEqual(response["statusCode"], 200, data)
            self.assert_error(413, "request_too_large", event_for(body=body + " ", encoded=encoded))
        for body in ('{"query":"a","query":"b","menu":{}}', '{bad', '"not an object"', '[' * 1500):
            code = "invalid_request" if body == '"not an object"' else "invalid_json"
            self.assert_error(400, code, event_for(body=body))

    def test_media_type_and_route_metadata(self):
        event = event_for()
        event["resource"] = PATH
        event["path"] = "/prod" + PATH
        response, data = self.invoke(event)
        self.assertEqual(response["statusCode"], 200, data)
        event["headers"]["Content-Type"] = "multipart/form-data"
        self.assert_error(415, "unsupported_media_type", event)

    def test_no_source_metadata_is_required_for_a_usable_menu(self):
        self.openai_body["status"] = "incomplete"
        self.openai_body["output"].insert(0, {"type": "web_search_call", "id": "ws-test",
            "status": "completed", "action": {"type": "search", "sources": None}})
        response, data = self.invoke()
        self.assertEqual(response["statusCode"], 200, data)

    def test_empty_sections_are_returned_in_order(self):
        empty = {"name": "Monday-Friday 3-6 & Thursday-Saturday 10-Midnight", "items": []}
        for sections in ([empty], [empty, *MENU["sections"], empty]):
            with self.subTest(sections=sections):
                self.openai_body = image_tests.MenuHandlerTests.response({**MENU, "sections": sections})
                response, menu = self.invoke()
                self.assertEqual(response["statusCode"], 200, menu)
                self.assertEqual(menu["sections"], sections)

    def test_unusable_menu_and_no_menu_results_are_safe_errors(self):
        for menu in ({}, {"possible_menus": [CANDIDATE]}, {**MENU, "sections": []}, {**MENU, "sections": [{}]}):
            self.openai_body = image_tests.MenuHandlerTests.response(menu)
            self.assert_error(502, "invalid_openai_response")
        for reason in ("no_menu", "illegible_text"):
            self.openai_body = image_tests.MenuHandlerTests.response({**MENU, "status": reason})
            _, data = self.assert_error(422, "no_menu_found" if reason == "no_menu" else reason)
            self.assertNotIn("in the images", data["error"]["message"])
        self.openai_body = image_tests.MenuHandlerTests.response(MENU)
        self.openai_body["output"][0]["content"][0]["text"] = "not JSON"
        self.assert_error(502, "invalid_openai_response")

    def test_billing_network_and_timeout_errors_have_no_retry_or_fallback(self):
        for upstream, code, expected in ((429, "insufficient_quota", 503), (401, "invalid_api_key", 503),
                                         (429, "rate_limit_exceeded", 503), (500, "server_error", 502)):
            self.status = upstream
            self.error_body = {"error": {"code": code, "message": PRIVATE}}
            before = len(self.requests["OPENAI"])
            response, data = self.invoke()
            self.assertEqual(response["statusCode"], expected, data)
            self.assertNotIn(PRIVATE, response["body"])
            self.assertEqual(len(self.requests["OPENAI"]), before + 1)
        self.status = 200
        self.network_error = httpx.ConnectError(PRIVATE)
        self.assert_error(502, "openai_unavailable")
        self.network_error = httpx.ReadTimeout(PRIVATE)
        self.assert_error(504, "openai_timeout")

    def test_direct_file_http_retrieval_failures_are_safe_without_retry_or_fallback(self):
        for extension, upstream_code, status, code in (
            ("pdf", "invalid_request_error", 502, "openai_failure"),
            ("pdf", "invalid_file", 502, "openai_failure"),
            ("jpg", "failed_to_download_image", 502, "openai_failure"),
            ("jpeg", "invalid_image_url", 502, "openai_failure"),
            ("png", "image_file_not_found", 502, "openai_failure"),
            ("webp", "invalid_image_format", 400, "invalid_image"),
        ):
            with self.subTest(extension=extension, upstream_code=upstream_code):
                source = f"https://venue.example.com/menu.{extension}?signature={PRIVATE}"
                self.status = 400
                self.error_body = {"error": {"code": upstream_code, "message": f"Cannot read {source}"}}
                before = len(self.requests["OPENAI"])
                self.assert_error(status, code, event_for({"query": QUERY, "menu": {
                    **CANDIDATE, "source_url": source}}))
                self.assertEqual(len(self.requests["OPENAI"]), before + 1)
                self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
                self.anthropic.assert_not_called()

    def test_direct_file_failed_responses_are_safe_without_fallback(self):
        for extension, code in (("pdf", "server_error"), ("png", "failed_to_download_image")):
            with self.subTest(extension=extension):
                self.openai_body = image_tests.MenuHandlerTests.response(MENU)
                self.openai_body.update(status="failed", output=[], error={"code": code, "message": PRIVATE})
                before = len(self.requests["OPENAI"])
                self.assert_error(502, "invalid_openai_response", event_for({"query": QUERY, "menu": {
                    **CANDIDATE, "source_url": f"https://venue.example.com/menu.{extension}"}}))
                self.assertEqual(len(self.requests["OPENAI"]), before + 1)

    def test_unavailable_and_unreadable_files_keep_recoverable_menu_errors(self):
        for extension in ("pdf", "jpg", "jpeg", "png", "webp"):
            for status, code in (("no_menu", "no_menu_found"), ("illegible_text", "illegible_text")):
                with self.subTest(extension=extension, status=status):
                    self.openai_body = image_tests.MenuHandlerTests.response({**MENU, "status": status})
                    before = len(self.requests["OPENAI"])
                    _, data = self.assert_error(422, code, event_for({"query": QUERY, "menu": {
                        **CANDIDATE, "source_url": f"https://venue.example.com/menu.{extension}"}}))
                    self.assertIn("Try another match", data["error"]["message"])
                    self.assertEqual(len(self.requests["OPENAI"]), before + 1)

    def test_total_deadline_and_remaining_lambda_budget(self):
        context = Mock()
        context.get_remaining_time_in_millis.return_value = 7000
        response, data = self.invoke(context=context)
        self.assertEqual(response["statusCode"], 200, data)
        self.assertEqual(self.openai.call_args.kwargs["timeout"], 4)
        context.get_remaining_time_in_millis.return_value = 2000
        self.assert_error(504, "openai_timeout", context=context)

        async def stalled(*args):
            await asyncio.Event().wait()

        with patch.object(openai_provider, "_request_online_transcription", side_effect=stalled), \
            patch.object(openai_provider, "ONLINE_TRANSCRIPTION_TIMEOUT_SECONDS", 0.01):
            self.assert_error(504, "openai_timeout")

    def test_local_adapter_preflight_and_shared_enable_flag(self):
        spec = importlib.util.spec_from_file_location("online_dev_server", HANDLER_PATH.parents[1] / "dev_server.py")
        assert spec is not None and spec.loader is not None
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        client = server.create_app(load_environment=False).test_client()
        self.assertEqual(client.options(PATH).status_code, 204)
        with patch.object(server.handler.openai_provider, "transcribe_online_menu", return_value={
                "isDummy": False, "restaurantName": MENU["restaurantName"], "sections": MENU["sections"]}) as transcribe:
            response = client.post(PATH, json={"query": QUERY, "menu": CANDIDATE})
            self.assertEqual(response.status_code, 200)
            transcribe.assert_called_once_with(QUERY, CANDIDATE, None)
        self.assertEqual(client.get(PATH).status_code, 405)