"""Check provider separation and cold-start imports in the deployed Lambda layout."""

import base64
import os
import subprocess
import sys
import unittest

from test_index import HANDLER_PATH, PNG_BYTES, handler, openai_provider, shared


class ModuleLayoutTests(unittest.TestCase):
    def test_providers_and_sdks_are_different_modules(self):
        self.assertEqual(openai_provider.__name__, "providers.openai")
        self.assertEqual(openai_provider.openai_sdk.__name__, "openai")
        self.assertEqual(handler.anthropic_provider.__name__, "providers.anthropic")
        self.assertEqual(handler.anthropic_provider.anthropic_sdk.__name__, "anthropic")
        self.assertIs(openai_provider.ApiError, shared.ApiError)
        self.assertIs(handler.anthropic_provider.ApiError, shared.ApiError)
        self.assertIs(handler.ApiError, shared.ApiError)

    def test_image_validation_returns_provider_neutral_data(self):
        images = handler.validate_images([(b"image/png", PNG_BYTES)])
        self.assertEqual(images, [shared.MenuImage(
            media_type="image/png", base64_data=base64.b64encode(PNG_BYTES).decode("ascii"),
        )])

    def test_cold_start_imports_work_in_both_sdk_import_orders(self):
        script = """
import json
import index
import openai
import anthropic
assert index.openai_provider.openai_sdk is openai
assert index.anthropic_provider.anthropic_sdk is anthropic
event = {'httpMethod': 'POST', 'headers': {'Content-Type': 'application/json'},
         'body': '{"dummy":true}', 'isBase64Encoded': False}
for provider in ('OPENAI', 'ANTHROPIC'):
    index.TRANSCRIPTION_PROVIDER = provider
    response = index.lambda_handler(event, None)
    assert response['statusCode'] == 200
    assert json.loads(response['body'])['isDummy'] is True
print('cold-start checks passed')
"""
        for prefix in ("", "import openai, anthropic\n"):
            with self.subTest(sdks_imported_first=bool(prefix)):
                result = subprocess.run(
                    [sys.executable, "-c", prefix + script], cwd=HANDLER_PATH.parent,
                    env={**os.environ, "ENABLE_LIVE_TRANSCRIPTION": "false"},
                    capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "cold-start checks passed")


if __name__ == "__main__":
    unittest.main()