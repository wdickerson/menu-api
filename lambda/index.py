"""Minimal API Gateway REST proxy handler; no AWS SDK or storage required."""

import base64
import binascii
import json
import os


def lambda_handler(event, context):
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": os.environ.get(
            "ALLOWED_ORIGIN", "https://menu.littleapp.io"
        ),
        "Access-Control-Allow-Methods": "POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Cache-Control": "no-store",
    }

    def response(status_code, payload):
        return {
            "statusCode": status_code,
            "headers": headers,
            "body": json.dumps(payload),
        }

    if event.get("httpMethod") != "POST":
        headers["Allow"] = "POST,OPTIONS"
        return response(405, {"message": "Method not allowed. Use POST."})

    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body, validate=True).decode("utf-8")
        payload = json.loads(body)
    except (ValueError, TypeError, binascii.Error):
        return response(400, {"message": "Body must be a valid JSON object."})

    if not isinstance(payload, dict):
        return response(400, {"message": "Body must be a JSON object."})

    return response(200, {
        "isDummy": True,
        "restaurantName": "Example Restaurant",
        "sections": [{
            "name": "Sample menu",
            "items": ["Example soup. $8.", "Example salad. $12."],
        }],
    })