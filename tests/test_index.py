"""Offline tests; run from the API repository root with unittest discovery."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch


HANDLER_PATH = Path(__file__).resolve().parents[1] / "lambda" / "index.py"
spec = importlib.util.spec_from_file_location("menu_handler", HANDLER_PATH)
assert spec is not None and spec.loader is not None
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)


class MenuHandlerTests(unittest.TestCase):
    def invoke(self, body="{}", method="POST", encoded=False):
        with patch.dict(os.environ, {"ALLOWED_ORIGIN": "https://menu.example.com"}):
            result = handler.lambda_handler({
                "httpMethod": method,
                "body": body,
                "isBase64Encoded": encoded,
            }, None)
        self.assertEqual(result["headers"]["Content-Type"], "application/json")
        self.assertEqual(
            result["headers"]["Access-Control-Allow-Origin"],
            "https://menu.example.com",
        )
        self.assertEqual(result["headers"]["Access-Control-Allow-Methods"], "POST,OPTIONS")
        self.assertEqual(result["headers"]["Access-Control-Allow-Headers"], "Content-Type")
        self.assertEqual(result["headers"]["Cache-Control"], "no-store")
        self.assertIsInstance(result["body"], str)
        return result, json.loads(result["body"])

    def test_post_returns_frontend_menu_contract(self):
        result, menu = self.invoke('{"imageUrl": "https://example.com/menu.jpg"}')
        self.assertEqual(result["statusCode"], 200)
        self.assertTrue(menu["isDummy"])
        self.assertEqual(menu["restaurantName"], "Example Restaurant")
        self.assertTrue(menu["sections"])
        for section in menu["sections"]:
            self.assertIsInstance(section["name"], str)
            self.assertTrue(section["items"])
            self.assertTrue(all(isinstance(item, str) for item in section["items"]))

    def test_empty_body_is_accepted(self):
        for body in (None, "", "{}"):
            with self.subTest(body=body):
                result, _ = self.invoke(body)
                self.assertEqual(result["statusCode"], 200)

    def test_missing_body_is_accepted(self):
        result = handler.lambda_handler({"httpMethod": "POST"}, None)
        self.assertEqual(result["statusCode"], 200)

    def test_malformed_json_is_rejected(self):
        result, payload = self.invoke("not json")
        self.assertEqual(result["statusCode"], 400)
        self.assertIn("message", payload)

    def test_json_must_be_an_object(self):
        for body in ("[]", "null", '"hello"', "123", "true"):
            with self.subTest(body=body):
                result, _ = self.invoke(body)
                self.assertEqual(result["statusCode"], 400)

    def test_base64_json_is_accepted(self):
        body = base64.b64encode(b'{"example": true}').decode("ascii")
        result, _ = self.invoke(body, encoded=True)
        self.assertEqual(result["statusCode"], 200)

    def test_invalid_base64_or_utf8_is_rejected(self):
        for body in ("%%%", base64.b64encode(b"\xff").decode("ascii")):
            with self.subTest(body=body):
                result, _ = self.invoke(body, encoded=True)
                self.assertEqual(result["statusCode"], 400)

    def test_other_methods_are_rejected_by_handler(self):
        # Real OPTIONS requests are answered by API Gateway's MOCK integration.
        for method in ("GET", "PUT", "DELETE", "OPTIONS", None):
            with self.subTest(method=method):
                result, _ = self.invoke(method=method)
                self.assertEqual(result["statusCode"], 405)
                self.assertEqual(result["headers"]["Allow"], "POST,OPTIONS")

    def test_request_data_is_not_echoed(self):
        _, menu = self.invoke('{"privateValue": "do-not-echo"}')
        self.assertNotIn("do-not-echo", json.dumps(menu))

    def test_default_origin(self):
        with patch.dict(os.environ, {}, clear=True):
            result = handler.lambda_handler({"httpMethod": "POST"}, None)
        self.assertEqual(
            result["headers"]["Access-Control-Allow-Origin"],
            "https://menu.littleapp.io",
        )


if __name__ == "__main__":
    unittest.main()