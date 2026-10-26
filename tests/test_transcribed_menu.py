"""Offline sharing contracts using real boto3 serialization and botocore stubs."""

import base64
import copy
import importlib.util
import json
import os
import unittest
from unittest.mock import patch

import boto3
from botocore.exceptions import EndpointConnectionError, NoCredentialsError, ReadTimeoutError
from botocore.stub import Stubber

import test_index as transcription


handler = transcription.handler
storage = handler.transcribed_menu
REAL_TOKEN_URLSAFE = storage.secrets.token_urlsafe
MENU_ID = "rP_KAcJggGmj1w"
OTHER_ID = "mfEQ80qbbkDBTg"
TABLE = "offline-shared-menus"
MENU = {"restaurantName": "Café 🦞", "sections": [
    {"name": "", "items": []},
    {"name": "Dinner", "items": ["Soup. $8.", "Tea. $3."]},
]}
SHARED_MENU = {**MENU, "sharedMenuId": MENU_ID}
UNSET = object()


def document(menu=MENU):
    return json.dumps(menu, ensure_ascii=False, separators=(",", ":"))


def event(*, method="POST", path="/transcribed_menu", payload=MENU, body=UNSET):
    return {"httpMethod": method, "path": path,
            "headers": {"Content-Type": "application/json"},
            "body": document(payload) if body is UNSET else body,
            "isBase64Encoded": False}


class SharingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {
            "DYNAMO_TABLE": TABLE, "AWS_DEFAULT_REGION": "us-east-1",
            "ENABLE_LIVE_TRANSCRIPTION": "false", "ENABLE_LIVE_SEARCH": "false",
        }, clear=True))
        self.database = boto3.resource("dynamodb", region_name="us-east-1",
                                      aws_access_key_id="offline", aws_secret_access_key="offline")
        self.addCleanup(self.database.meta.client.close)
        self.stubber = Stubber(self.database.meta.client)
        self.enterContext(self.stubber)
        self.addCleanup(self.stubber.assert_no_pending_responses)
        self.aws = self.enterContext(patch.object(storage.boto3, "resource", return_value=self.database))
        self.token = self.enterContext(patch.object(storage.secrets, "token_urlsafe", return_value=MENU_ID))
        self.log = self.enterContext(patch.object(handler, "logger"))
        self.no_calls = [
            self.enterContext(patch.object(transcription.shared.boto3, "client")),
            self.enterContext(patch.object(handler.openai_provider.openai_sdk, "AsyncOpenAI")),
            self.enterContext(patch.object(handler.anthropic_provider.anthropic_sdk, "AsyncAnthropic")),
        ]
        for mock in self.no_calls:
            mock.side_effect = AssertionError("Sharing must not invoke providers or Secrets Manager")
            self.addCleanup(mock.assert_not_called)

    def put_params(self, menu=MENU, menu_id=MENU_ID):
        return {"TableName": TABLE, "Item": {"menu_id": menu_id, "menu_json": document(menu)},
                "ConditionExpression": "attribute_not_exists(menu_id)"}

    def get_params(self, menu_id=MENU_ID):
        return {"TableName": TABLE, "Key": {"menu_id": menu_id}, "ConsistentRead": True}

    def expect_get(self, menu=MENU):
        self.stubber.add_response("get_item", {"Item": {
            "menu_id": {"S": MENU_ID}, "menu_json": {"S": document(menu)},
        }}, self.get_params())

    def invoke(self, request=None, *, context=None):
        response = handler.lambda_handler(event() if request is None else request, context)
        self.assertEqual(response["headers"]["Access-Control-Allow-Origin"], "*")
        self.assertEqual(response["headers"]["Access-Control-Allow-Headers"], "*")
        self.assertEqual(response["headers"]["Cache-Control"], "no-store")
        self.assertEqual(response["headers"]["Content-Type"], "application/json")
        return response, json.loads(response["body"])

    def assert_error(self, status, code, request=None):
        response, data = self.invoke(request)
        self.assertEqual(response["statusCode"], status, data)
        self.assertEqual(data["error"]["code"], code, data)
        self.assertEqual(set(data), {"error"})
        self.assertEqual(set(data["error"]), {"code", "message"})
        self.assertTrue(data["error"]["message"])
        for private in ("PRIVATE-AWS-DETAIL", TABLE, MENU["restaurantName"]):
            self.assertNotIn(private, response["body"] + str(self.log.mock_calls))
        return response

    def test_create_returns_id_only_after_conditional_write_without_paid_flags(self):
        self.stubber.add_response("put_item", {}, self.put_params())
        response, data = self.invoke()
        self.assertEqual(response["statusCode"], 201)
        self.assertEqual(data, {"menu_id": MENU_ID})
        self.assertEqual(response["headers"]["Access-Control-Allow-Methods"], "POST,OPTIONS")
        self.token.assert_called_once_with(10)
        config = self.aws.call_args.kwargs["config"]
        self.assertEqual(config.connect_timeout, 2)
        self.assertEqual(config.read_timeout, 3)
        self.assertEqual(config.retries, {"total_max_attempts": 1})

    def test_get_returns_app_menu_with_shared_id_and_uses_strong_read(self):
        self.expect_get()
        response, data = self.invoke(event(method="GET", path=f"/transcribed_menu/{MENU_ID}"))
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(data, SHARED_MENU)
        self.assertEqual(response["headers"]["Access-Control-Allow-Methods"], "GET,OPTIONS")
        self.token.assert_not_called()

    def test_get_uses_requested_id_instead_of_stored_metadata(self):
        self.expect_get({**MENU, "sharedMenuId": OTHER_ID, "menu_id": OTHER_ID})
        response, data = self.invoke(event(method="GET", path=f"/transcribed_menu/{MENU_ID}"))
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(data, SHARED_MENU)

    def test_gateway_resource_uses_case_sensitive_path_parameter(self):
        self.expect_get()
        request = event(method="GET", path=f"/prod/transcribed_menu/{MENU_ID}")
        request.update(resource="/transcribed_menu/{menu_id}", pathParameters={"menu_id": MENU_ID})
        response, menu = self.invoke(request)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(menu, SHARED_MENU)

    def test_gateway_item_requires_path_parameter(self):
        for params in (None, {}, {"menu_id": None}, {"menu_id": 42}):
            request = event(method="GET")
            request.update(resource="/transcribed_menu/{menu_id}", pathParameters=params)
            self.assert_error(400, "invalid_menu_id", request)
        self.aws.assert_not_called()

    def test_collision_regenerates_without_read_or_overwrite(self):
        self.token.side_effect = [MENU_ID, OTHER_ID]
        self.stubber.add_client_error("put_item", "ConditionalCheckFailedException",
                                     expected_params=self.put_params())
        self.stubber.add_response("put_item", {}, self.put_params(menu_id=OTHER_ID))
        response, data = self.invoke()
        self.assertEqual(response["statusCode"], 201)
        self.assertEqual(data, {"menu_id": OTHER_ID})
        self.assertEqual(self.token.call_count, 2)

    def test_collision_retries_are_bounded(self):
        for _ in range(storage.MAX_ID_ATTEMPTS):
            self.stubber.add_client_error("put_item", "ConditionalCheckFailedException",
                                         expected_params=self.put_params())
        self.assert_error(503, "sharing_unavailable")
        self.assertEqual(self.token.call_count, storage.MAX_ID_ATTEMPTS)

    def test_random_ids_match_desired_format(self):
        ids = {REAL_TOKEN_URLSAFE(10) for _ in range(100)}
        self.assertEqual(len(ids), 100)
        for menu_id in ids:
            self.assertIsNotNone(storage.MENU_ID_PATTERN.fullmatch(menu_id))
        self.aws.assert_not_called()

    def test_get_missing_menu_is_404_not_a_database_failure(self):
        self.stubber.add_response("get_item", {}, self.get_params())
        self.assert_error(404, "menu_not_found", event(method="GET", path=f"/transcribed_menu/{MENU_ID}"))

    def test_invalid_ids_do_not_access_database(self):
        for menu_id in ("", "short", MENU_ID + "a", "!" * 14, "é" * 14, "a/b", "../" + MENU_ID,
                        MENU_ID + "/", "a" * 1000):
            with self.subTest(menu_id=menu_id):
                self.assert_error(400, "invalid_menu_id", event(method="GET", path=f"/transcribed_menu/{menu_id}"))
        self.aws.assert_not_called()

    def test_no_edit_or_delete_and_route_specific_allow(self):
        for path, allowed in (("/transcribed_menu", "POST,OPTIONS"),
                              (f"/transcribed_menu/{MENU_ID}", "GET,OPTIONS")):
            for method in ("PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
                           "GET" if allowed.startswith("POST") else "POST"):
                response = self.assert_error(405, "method_not_allowed", event(method=method, path=path))
                self.assertEqual(response["headers"]["Allow"], allowed)
        self.aws.assert_not_called()

    def test_unknown_route_is_not_a_sharing_operation(self):
        self.assert_error(404, "not_found", event(path="/transcribed_menus"))
        self.aws.assert_not_called()

    def test_missing_configuration_is_recoverable_and_lazy(self):
        with patch.dict(os.environ, {"DYNAMO_TABLE": "  "}):
            self.assert_error(503, "api_not_configured")
            self.assert_error(503, "api_not_configured", event(method="GET", path=f"/transcribed_menu/{MENU_ID}"))
        self.aws.assert_not_called()

    def test_invalid_menu_shapes_do_not_access_database(self):
        cases = [None, [], True, "menu", {}, {**MENU, "restaurantName": "  "},
                 {**MENU, "restaurantName": 3}, {**MENU, "sections": []}, {**MENU, "sections": {}},
                 {"restaurantName": "Cafe"}, {"sections": MENU["sections"]}]
        for section in (None, {}, {"name": "a"}, {"items": []}, {"name": 4, "items": []},
                        {"name": "", "items": "a"}, {"name": "", "items": [None]},
                        {"name": "", "items": [" "]}):
            cases.append({**MENU, "sections": [section]})
        for payload in cases:
            with self.subTest(payload=payload):
                self.assert_error(400, "invalid_menu", event(payload=payload))
        self.aws.assert_not_called()

    def test_extra_metadata_is_not_stored(self):
        payload = copy.deepcopy(MENU)
        payload.update(isDummy=True, status="success", menu_id=OTHER_ID,
                       sharedMenuId=OTHER_ID, capturedAt="private")
        payload["sections"][1]["private"] = "not stored"
        self.stubber.add_response("put_item", {}, self.put_params())
        response, _ = self.invoke(event(payload=payload))
        self.assertEqual(response["statusCode"], 201)

    def test_empty_sections_and_blank_headings_preserve_exact_text(self):
        menu = {"restaurantName": "  Cafe  ", "sections": [{"name": "  ", "items": []}]}
        self.stubber.add_response("put_item", {}, self.put_params(menu))
        self.assertEqual(self.invoke(event(payload=menu))[0]["statusCode"], 201)
        self.expect_get(menu)
        self.assertEqual(self.invoke(event(method="GET", path=f"/transcribed_menu/{MENU_ID}"))[1],
                 {**menu, "sharedMenuId": MENU_ID})

    def test_json_media_type_and_decoding_errors(self):
        for body in ("", "{", '{"restaurantName":"one","restaurantName":"two"}',
                     '{"sections":[{"name":"one","name":"two"}]}'):
            self.assert_error(400, "invalid_json", event(body=body))
        for headers in ({}, {"Content-Type": "text/plain"}, {"Content-Type": "multipart/form-data"}):
            request = event()
            request["headers"] = headers
            self.assert_error(415, "unsupported_media_type", request)
        request = event(body="%%%")
        request["isBase64Encoded"] = True
        self.assert_error(400, "invalid_body", request)
        request["body"] = base64.b64encode(b"\xff").decode()
        self.assert_error(400, "invalid_json", request)
        self.assert_error(400, "invalid_body", event(body=42))
        self.aws.assert_not_called()

    def test_base64_json_and_charset_supported(self):
        self.stubber.add_response("put_item", {}, self.put_params())
        request = event(body=base64.b64encode(document().encode()).decode())
        request.update(isBase64Encoded=True, headers={"content-type": "application/json; charset=utf-8"})
        self.assertEqual(self.invoke(request)[0]["statusCode"], 201)

    def test_unicode_body_limit_counts_utf8_bytes(self):
        for encoded in (False, True):
            body = document({**MENU, "restaurantName": "🦞" * (storage.MAX_SHARED_MENU_BYTES // 4)})
            self.assertLess(len(body), storage.MAX_SHARED_MENU_BYTES)
            request = event(body=base64.b64encode(body.encode()).decode() if encoded else body)
            request["isBase64Encoded"] = encoded
            self.assert_error(413, "request_too_large", request)
        self.aws.assert_not_called()

    def test_maximum_body_and_many_empty_sections_fit_dynamodb_limit(self):
        menu = {**MENU, "restaurantName": "x"}
        menu["restaurantName"] += "x" * (storage.MAX_SHARED_MENU_BYTES - len(document(menu).encode()))
        self.assertEqual(len(document(menu).encode()), storage.MAX_SHARED_MENU_BYTES)
        self.stubber.add_response("put_item", {}, self.put_params(menu))
        self.assertEqual(self.invoke(event(payload=menu))[0]["statusCode"], 201)
        menu = {"restaurantName": "Cafe", "sections": [{"name": "", "items": []}] * 10_000}
        self.assertLess(len(document(menu).encode()) + 100, 400 * 1024)
        self.stubber.add_response("put_item", {}, self.put_params(menu))
        self.assertEqual(self.invoke(event(payload=menu))[0]["statusCode"], 201)

    def test_unpaired_surrogate_is_invalid_not_internal_error(self):
        self.assert_error(400, "invalid_menu", event(payload={**MENU, "restaurantName": "\ud800"},
                                                   body=json.dumps({**MENU, "restaurantName": "\ud800"})))
        self.aws.assert_not_called()

    def test_database_failures_are_safe_503_and_not_retried(self):
        for operation in ("put_item", "get_item"):
            request = event() if operation == "put_item" else event(method="GET", path=f"/transcribed_menu/{MENU_ID}")
            params = self.put_params() if operation == "put_item" else self.get_params()
            for code in ("ProvisionedThroughputExceededException", "AccessDeniedException",
                         "ResourceNotFoundException", "InternalServerError", "ValidationException"):
                with self.subTest(operation=operation, code=code):
                    self.stubber.add_client_error(operation, code, "PRIVATE-AWS-DETAIL", expected_params=params)
                    self.assert_error(503, "sharing_unavailable", request)

    def test_credential_and_network_failures_are_safe(self):
        for error in (NoCredentialsError(), EndpointConnectionError(endpoint_url="PRIVATE-AWS-DETAIL"),
                      ReadTimeoutError(endpoint_url="PRIVATE-AWS-DETAIL")):
            self.aws.side_effect = error
            self.assert_error(503, "sharing_unavailable")

    def test_corrupt_storage_does_not_leak_internal_data(self):
        for item in ({"menu_id": {"S": MENU_ID}}, {"menu_json": {"S": "not-json"}},
                     {"menu_json": {"S": "{}"}}, {"menu_json": {"N": "3"}}):
            self.stubber.add_response("get_item", {"Item": item}, self.get_params())
            self.assert_error(503, "sharing_unavailable", event(method="GET", path=f"/transcribed_menu/{MENU_ID}"))

    def test_dummy_transcription_does_not_touch_storage(self):
        request = event(path="/menu_image_transcribe", payload={"dummy": True})
        response, menu = self.invoke(request)
        self.assertEqual(response["statusCode"], 200)
        self.assertTrue(menu["isDummy"])
        self.aws.assert_not_called()


class LocalSharingTests(unittest.TestCase):
    def setUp(self):
        path = transcription.HANDLER_PATH.parents[1] / "dev_server.py"
        spec = importlib.util.spec_from_file_location("menu_sharing_local_adapter", path)
        assert spec is not None and spec.loader is not None
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.enterContext(patch.object(self.adapter, "handler", handler))
        self.client = self.adapter.create_app(load_environment=False).test_client()

    def test_create_and_read_use_same_handler_and_gateway_path_parameters(self):
        with patch.object(storage, "create_menu", return_value={"menu_id": MENU_ID}) as create:
            response = self.client.post("/transcribed_menu", json=MENU)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json(), {"menu_id": MENU_ID})
        create.assert_called_once_with(MENU)
        with patch.object(storage, "get_menu", return_value=SHARED_MENU) as get, \
                patch.object(handler, "lambda_handler", wraps=handler.lambda_handler) as wrapped:
            response = self.client.get(f"/transcribed_menu/{MENU_ID}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), SHARED_MENU)
        get.assert_called_once_with(MENU_ID)
        request, context = wrapped.call_args.args
        self.assertEqual(request["resource"], "/transcribed_menu/{menu_id}")
        self.assertEqual(request["pathParameters"], {"menu_id": MENU_ID})
        self.assertIsNone(context)

    def test_local_preflights_do_not_call_lambda(self):
        with patch.object(handler, "lambda_handler") as call:
            for path, methods in (("/transcribed_menu", "POST,OPTIONS"),
                                  (f"/transcribed_menu/{MENU_ID}", "GET,OPTIONS")):
                response = self.client.options(path)
                self.assertEqual(response.status_code, 204)
                self.assertEqual(response.headers["Access-Control-Allow-Methods"], methods)
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        call.assert_not_called()

    def test_local_menu_limit_and_disallowed_methods(self):
        with patch.object(storage.boto3, "resource") as aws:
            response = self.client.post("/transcribed_menu", data=" " * (storage.MAX_SHARED_MENU_BYTES + 1),
                                        content_type="application/json")
            self.assertEqual(response.status_code, 413)
            for path in ("/transcribed_menu", f"/transcribed_menu/{MENU_ID}"):
                for method in ("PUT", "PATCH", "DELETE"):
                    response = self.client.open(path, method=method)
                    self.assertEqual(response.status_code, 405)
                    self.assertEqual(response.get_json()["error"]["code"], "method_not_allowed")
        aws.assert_not_called()


if __name__ == "__main__":
    unittest.main()