"""Unknown request fields are rejected before credentials or provider requests."""

import json
import os
import unittest
from unittest.mock import patch

from test_index import PNG_FILE, handler, openai_provider, shared, upload_event


class RequestFieldTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {
            "ENABLE_LIVE_TRANSCRIPTION": "true", "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
        }, clear=True))
        self.aws = self.enterContext(patch.object(shared.boto3, "client"))
        self.openai = self.enterContext(patch.object(openai_provider.openai_sdk, "AsyncOpenAI"))
        self.anthropic = self.enterContext(patch.object(handler.anthropic_provider.anthropic_sdk, "AsyncAnthropic"))
        for mock in (self.aws, self.openai, self.anthropic):
            mock.side_effect = AssertionError("Unexpected credentials or provider call")

    def assert_rejected(self, event):
        response = handler.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 400, response["body"])
        self.assertEqual(json.loads(response["body"])["error"]["code"], "invalid_request")

    def assert_no_paid_calls(self):
        self.aws.assert_not_called()
        self.openai.assert_not_called()
        self.anthropic.assert_not_called()

    def test_unknown_fields_are_rejected_for_both_providers(self):
        for provider in ("OPENAI", "ANTHROPIC"):
            with patch.object(handler, "TRANSCRIPTION_PROVIDER", provider):
                for field, value in (("unexpected_field", "extra-value"), ("extra_field", "")):
                    with self.subTest(provider=provider, field=field):
                        self.assert_rejected(upload_event(fields=[(field, value)]))
        self.assert_no_paid_calls()

    def test_unknown_fields_are_rejected_after_images_and_as_files(self):
        for provider in ("OPENAI", "ANTHROPIC"):
            with patch.object(handler, "TRANSCRIPTION_PROVIDER", provider):
                for field, value in (("unexpected_field", "extra-value"), ("extra_field", "other-value")):
                    self.assert_rejected(upload_event(files=[PNG_FILE, (field, (None, value))]))
                    self.assert_rejected(upload_event(files=[PNG_FILE, (field, ("extra.txt", value.encode(), "text/plain"))]))
        self.assert_no_paid_calls()

    def test_unknown_fields_are_rejected_in_dummy_and_disabled_modes(self):
        with patch.dict(os.environ, {"ENABLE_LIVE_TRANSCRIPTION": "false"}):
            for provider in ("OPENAI", "ANTHROPIC"):
                with patch.object(handler, "TRANSCRIPTION_PROVIDER", provider):
                    for field, value in (("unexpected_field", "extra-value"), ("extra_field", "other-value")):
                        self.assert_rejected(upload_event(files=[], fields=[("dummy", "true"), (field, value)]))
                        self.assert_rejected(upload_event(fields=[(field, value)]))
        self.assert_no_paid_calls()

    def test_json_dummy_rejects_unknown_fields(self):
        for field, value in (("unexpected_field", "extra-value"), ("extra_field", "other-value")):
            self.assert_rejected({
                "httpMethod": "POST", "path": "/menu_image_transcribe",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"dummy": True, field: value}), "isBase64Encoded": False,
            })
        self.assert_no_paid_calls()


if __name__ == "__main__":
    unittest.main()