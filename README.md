# Menu API

Minimal Python Lambda, using the same `index.lambda_handler` and API Gateway REST
proxy convention as the Doodle API. No external dependencies, database, OCR, or
AWS credentials are required to run its unit tests.

## Contract

- **POST /menus** on the deployed API's `prod` stage.
- Send `Content-Type: application/json` and a JSON object. An empty body or `{}`
  is accepted. Fields are currently ignored; nothing is fetched or persisted.
- Returns HTTP 200 with `isDummy: true`, `restaurantName`, and nonempty `sections`
  containing `name` and string-valued `items`, matching the frontend menu shape.
- Malformed JSON, invalid base64, or a non-object JSON value returns HTTP 400.
- The handler returns HTTP 405 for methods other than POST. API Gateway only
  exposes POST and a MOCK OPTIONS preflight; unsupported public routes/methods
  are rejected by API Gateway instead (normally HTTP 403).
- `ALLOWED_ORIGIN` is configured by the stack to the HTTPS frontend origin.
  CORS is not authentication: the endpoint is public.

## Repository and deployment

Publish this directory's **contents** as the separate `menu-api` GitHub repository,
with [lambda/index.py](lambda/index.py) and [tests/test_index.py](tests/test_index.py)
at those paths. Commit the tests: the deployment pipeline runs them before upload.

Run `python -m unittest discover -s tests -v` from the repository root. AWS builds
and runs the function on Python 3.13; the skeleton also works on Python 3.9+ for
local testing.

On a push to `main`, CodeBuild tests the handler, zips the **contents** of the
Lambda directory (so the handler module is at the ZIP root), updates Lambda code,
and waits for the update to succeed. Only code inside that directory is deployed;
tests and documentation are excluded. The initial skeleton needs no dependency
installation. Add a dependency-packaging step before introducing third-party
packages, using Linux wheels compatible with Lambda's Python version and x86_64
architecture rather than packages built on macOS.

CloudFormation owns configuration and an inline bootstrap; this repository owns
subsequent application code releases. Do not deploy infrastructure and API code
concurrently. The frontend currently uses bundled JSON and does not call this API.