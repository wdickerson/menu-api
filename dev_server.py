"""Loopback-only development adapter; not part of the Lambda deployment."""

import argparse
import base64
import importlib.util
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from flask import Flask, Response, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.serving import is_running_from_reloader


API_ROOT = Path(__file__).resolve().parent
LAMBDA_ROOT = API_ROOT / "lambda"
ENV_FILE = API_ROOT / ".env"
if str(LAMBDA_ROOT) not in sys.path:
    sys.path.insert(0, str(LAMBDA_ROOT))
spec = importlib.util.spec_from_file_location("menu_lambda", LAMBDA_ROOT / "index.py")
assert spec is not None and spec.loader is not None
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Cache-Control": "no-store",
}


def create_app(*, load_environment: bool = True) -> Flask:
    # Explicit path: don't search parent directories for secrets. Shell values win.
    # Only this local adapter loads dotenv; the production handler remains unchanged.
    if load_environment:
        load_dotenv(ENV_FILE, override=False, interpolate=False)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = handler.MAX_BODY_BYTES

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge) -> Response:
        return Response(json.dumps({"error": {
            "code": "request_too_large",
            "message": f"Request body exceeds {handler.MAX_BODY_BYTES:,} bytes.",
        }}), status=413, headers=CORS_HEADERS, mimetype="application/json")

    @app.route("/menus", methods=["POST", "OPTIONS", "GET", "HEAD", "PUT", "PATCH", "DELETE"],
               provide_automatic_options=False)
    def menus() -> Response:
        if request.method == "OPTIONS":
            # API Gateway owns preflight in production; emulate it only locally.
            return Response(status=204, headers=CORS_HEADERS)
        # Never parse form fields here: Lambda must enforce the same multipart
        # contract and see exactly the original image bytes, including non-UTF-8.
        raw = request.stream.read(handler.MAX_BODY_BYTES + 1) if request.method == "POST" else b""
        if len(raw) > handler.MAX_BODY_BYTES:
            return too_large(RequestEntityTooLarge())
        event = {
            "httpMethod": request.method,
            "path": request.path,
            "headers": dict(request.headers),
            "multiValueHeaders": {key: request.headers.getlist(key) for key in request.headers.keys()},
            "body": base64.b64encode(raw).decode("ascii"),
            "isBase64Encoded": True,
        }
        result = handler.lambda_handler(event, None)
        body = result["body"]
        if result.get("isBase64Encoded"):
            body = base64.b64decode(body)
        return Response(body, status=result["statusCode"], headers=result["headers"])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    # Reload without exposing Werkzeug's interactive debugger. Watch dotenv too;
    # Only the serving child loads secrets. Loading them in the watcher parent
    # would pass stale dotenv values to every child, defeating override=False.
    create_app(load_environment=is_running_from_reloader()).run(
        host="127.0.0.1", port=args.port, debug=False,
        use_reloader=True, extra_files=[str(ENV_FILE)], load_dotenv=False)


if __name__ == "__main__":
    main()