"""Immutable shared menus; no database access until an explicit share/read request."""

from contextlib import contextmanager
import json
import os
import re
import secrets

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from providers.shared import ApiError


# A single compact JSON string avoids DynamoDB's per-map/list document overhead.
# Leave ample room below the 400 KiB item limit for the key and attribute names.
MAX_SHARED_MENU_BYTES = 300_000
MAX_ID_ATTEMPTS = 3
MENU_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{14}")


def validate_shared_menu(payload):
    """Project the app's menu shape without changing its text or empty sections."""
    invalid = ApiError(400, "invalid_menu", "Provide a restaurantName and sections containing names and string items.")
    if (not isinstance(payload, dict)
            or not isinstance(payload.get("restaurantName"), str)
            or not payload["restaurantName"].strip()
            or not isinstance(payload.get("sections"), list) or not payload["sections"]):
        raise invalid
    sections = []
    for section in payload["sections"]:
        if (not isinstance(section, dict) or not isinstance(section.get("name"), str)
                or not isinstance(section.get("items"), list)
                or any(not isinstance(item, str) or not item.strip() for item in section["items"])):
            raise invalid
        sections.append({"name": section["name"], "items": section["items"]})
    menu = {"restaurantName": payload["restaurantName"], "sections": sections}
    try:
        document = json.dumps(menu, ensure_ascii=False, separators=(",", ":"))
        size = len(document.encode("utf-8"))
    except UnicodeError:
        raise invalid from None
    if size > MAX_SHARED_MENU_BYTES:
        raise ApiError(413, "request_too_large", f"Shared menu exceeds {MAX_SHARED_MENU_BYTES:,} UTF-8 bytes.")
    return document


@contextmanager
def menu_table():
    # Lazy configuration/client creation keeps imports and dummy calls AWS-free,
    # and lets the local adapter load dotenv before the first database operation.
    name = os.environ.get("DYNAMO_TABLE", "").strip()
    if not name:
        raise ApiError(503, "api_not_configured", "Menu sharing is not configured.")
    try:
        database = boto3.resource("dynamodb", config=Config(
            connect_timeout=2, read_timeout=3, retries={"total_max_attempts": 1},
        ))
        try:
            yield database.Table(name)
        finally:
            database.meta.client.close()
    except (BotoCoreError, ClientError):
        # Never expose AWS messages, credentials, table identifiers, or menu data.
        raise ApiError(503, "sharing_unavailable", "Menu sharing is temporarily unavailable. Try again.") from None


def create_menu(payload):
    document = validate_shared_menu(payload)
    with menu_table() as table:
        for _ in range(MAX_ID_ATTEMPTS):
            menu_id = secrets.token_urlsafe(10)  # 80-bit / 14-character format.
            try:
                table.put_item(
                    Item={"menu_id": menu_id, "menu_json": document},
                    ConditionExpression="attribute_not_exists(menu_id)",
                )
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
                    raise
            else:
                return {"menu_id": menu_id}
    raise ApiError(503, "sharing_unavailable", "Could not create a share identifier. Try again.")


def get_menu(menu_id):
    if not isinstance(menu_id, str) or MENU_ID_PATTERN.fullmatch(menu_id) is None:
        raise ApiError(400, "invalid_menu_id", "Provide a valid 14-character menu identifier.")
    with menu_table() as table:
        # Shared links must work immediately after a successful POST.
        result = table.get_item(Key={"menu_id": menu_id}, ConsistentRead=True)
    item = result.get("Item")
    if item is None:
        raise ApiError(404, "menu_not_found", "This shared menu could not be found.")
    try:
        # Check/project persisted data as well; do not return internal attributes.
        menu = json.loads(validate_shared_menu(json.loads(item["menu_json"])))
        return {**menu, "sharedMenuId": menu_id}
    except (KeyError, TypeError, ValueError, RecursionError, ApiError):
        raise ApiError(503, "sharing_unavailable", "This shared menu is temporarily unavailable.") from None