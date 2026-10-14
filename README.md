# Menu API — Version 2

## Choose the transcription provider

Set `TRANSCRIPTION_PROVIDER` in [lambda/index.py](lambda/index.py) to exactly
`"OPENAI"` or `"ANTHROPIC"`, then release the API code. This selects an internal implementation
path, **not a second HTTP endpoint**: both use **POST /menus**, the same uploads,
dummy mode, success shape, validation, and live-transcription gate. Provider
selection cannot be overridden by an environment variable or request field.
No frontend, CloudFormation, or IAM changes are needed.

The Anthropic path uses **Claude Haiku 5.5** (`claude-haiku-5-5`, defined by
`DEFAULT_MODEL` in [lambda/providers/anthropic.py](lambda/providers/anthropic.py))
and the Messages API with JSON-schema
structured output. Add **`ANTHROPIC_API_KEY`** to the existing **`menu_api_secrets`**
JSON secret. The existing `MENU_API_SECRET_ID` setting selects that shared secret.
Only the chosen provider's key is read; an OpenAI key is never used as an Anthropic
fallback. Secret caches are scoped to both secret ID and provider key field.
For local development, `ANTHROPIC_API_KEY` in the process environment or local
dotenv file takes precedence over Secrets Manager, just like `OPENAI_API_KEY`.

Models and reasoning settings are configured in the provider source. To compare
settings locally, edit `DEFAULT_MODEL` and
`DEFAULT_REASONING_EFFORT` in [lambda/providers/openai.py](lambda/providers/openai.py),
or `DEFAULT_MODEL` in [lambda/providers/anthropic.py](lambda/providers/anthropic.py);
the development server reloads Python changes automatically. A production settings
change requires an API code release. There is no automatic retry or cross-provider fallback.
All uploaded images are included in a single paid request to the selected provider.
OpenAI uses the Responses API with the shared prompt and strict JSON schema,
high-detail image inputs, an 8,000-token output cap, and response storage disabled.
Its parser skips reasoning metadata, combines menu text in output order, and rejects
refusals, incomplete responses, and unexpected output items before menu validation.
Anthropic requests keep thinking disabled. If thinking metadata is nevertheless
returned, the parser skips it and validates only the menu text.

Anthropic-specific error codes are `anthropic_not_funded`,
`anthropic_authentication_failed`, `anthropic_rate_limited`, and
`anthropic_model_unavailable` (HTTP 503); `anthropic_unavailable` (HTTP 502 for
connection failures, 503 for overload); `anthropic_failure` and
`invalid_anthropic_response` (HTTP 502); and `anthropic_timeout` (HTTP 504).
Menu detection, invalid-image, refusal, and incomplete-transcription errors use
the same codes as the OpenAI path. Upstream error messages and keys are not exposed.
Both paths enforce a 20-second elapsed deadline, capped by remaining Lambda time
minus three seconds, with SDK retries disabled.

Offline tests cover both real provider SDKs with mocked HTTP transports; no paid
requests or AWS calls are made. Install the updated production dependencies before
running the local server or tests.

## Code organization

- [lambda/index.py](lambda/index.py): gateway request parsing, image validation,
   dummy mode, live-transcription gate, provider selection, and HTTP responses.
- [lambda/providers/openai.py](lambda/providers/openai.py): OpenAI defaults and
   Responses request, response parsing, and errors.
- [lambda/providers/anthropic.py](lambda/providers/anthropic.py): Haiku defaults,
   Messages request, response parsing (including thinking metadata), and errors.
- [lambda/providers/shared.py](lambda/providers/shared.py): `MenuImage`, `ApiError`,
   shared prompt/schema, provider-scoped credentials, elapsed deadlines, and menu validation.

Both provider modules expose `transcribe_menu`. Images are validated and encoded
once as provider-neutral `MenuImage` values; each provider builds its own SDK image
blocks. Provider source lives inside a package so the requested module filenames
do not shadow the installed OpenAI and Anthropic SDKs. The local server and tests
add only the Lambda source directory—not the provider directory—to their import path.
The existing deployment pipeline recursively packages this directory; no template
changes are required.

## Start the local API (automatic reload)

From this directory, use **Python 3.13**:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python dev_server.py
```

POST to **http://127.0.0.1:8000/menus** in Postman using the request formats below.
Use `python dev_server.py --port 8001` to change the port. Python source changes
and edits to the local dotenv file restart the server automatically.
The server binds only to loopback and never enables the interactive debugger;
it is a development server, not a production host.

The gitignored local dotenv file is loaded only by [dev_server.py](dev_server.py),
from this directory regardless of your working directory. Existing process
environment variables take precedence. [.env.example](.env.example) contains no
secret and keeps `ENABLE_LIVE_TRANSCRIPTION=false`. Dummy mode needs no key:

```sh
curl http://127.0.0.1:8000/menus \
   -H 'Content-Type: application/json' --data '{"dummy":true}'
```

For intentional **paid** transcription, put the selected provider's real
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the
local dotenv file and set `ENABLE_LIVE_TRANSCRIPTION=true` there. Do not put keys
in request fields, frontend configuration, logs, or source control. Alternatively,
configure `MENU_API_SECRET_ID` and AWS credentials using a secure local mechanism.
Never commit the local dotenv file. Model and reasoning settings live in the
provider source code.

Run all offline checks with development dependencies installed:

```sh
python -m unittest discover -s tests -v
```

The local adapter forwards raw bodies as binary API Gateway events to the unchanged
Lambda handler, preserving validation, upload limits, errors, and response headers.
Local OPTIONS supports non-credentialed browser preflights. It does not emulate AWS
authentication, throttling, event-envelope limits, or gateway timing. The existing
frontend still targets the deployed API; starting this server does not redirect it.
[requirements-dev.txt](requirements-dev.txt), the server, and dotenv loading are
development-only: the production pipeline still installs [requirements.txt](requirements.txt)
and packages only the Lambda directory. Production Lambda never loads dotenv.

## Production API

Python Lambda (`index.lambda_handler`) behind API Gateway REST **POST /menus**.
Transcribes 1–5 menu images using OpenAI or Anthropic vision and strict structured output.
Printed menus, counter boards, food-truck posters, and handwritten beer lists
are supported through the transcription prompt. No database or image storage.
This development-server addition does not change the frontend or its API URL.

## Request

POST to the stack's `ApiPostUrl` using **multipart/form-data** with actual image
files. Let the HTTP client generate the Content-Type header and boundary.

| Field name | Field type | Value |
| --- | --- | --- |
| `images` | File | A JPEG, PNG, or WebP image. Repeat this exact field name for 1–5 files. |
| `dummy` | Text (optional) | Exactly `true` or `false`; defaults to `false`. |

These are the only accepted multipart fields.

Images must belong to the same restaurant; order them in menu reading order.
All images are sent together in **one** request to the selected provider. The server
encodes the original bytes internally (OpenAI data URLs or Anthropic base64 image
blocks); clients do not encode images.

- Use `images`, **not** `images[]`. Each part needs a nonempty filename. Filenames
   are never used as paths or sent to either provider. Unknown fields and duplicate text
   fields are rejected.
- File types: JPEG, PNG, or WebP. No remote URLs, base64 text fields, JSON image
  arrays, PDFs, GIFs, or HEIC. Convert HEIC camera photos to JPEG first. Declared
  image MIME types must match file signatures. Missing/generic
  `application/octet-stream` types are inferred from signatures, not filenames.
   The selected provider performs full image decoding; local signature checks alone cannot prove
  a file is valid or readable.

### Upload limits (no S3 required)

- **4,500,000 bytes (4.5 decimal MB) combined original image files**, whether
   uploading one file or five. This replaces the old 2.8 MB image limit.
- **4,600,000 bytes (4.6 decimal MB) for the complete multipart body**, including
  boundaries, part headers, and text fields.
- At most 6 parts (five files plus `dummy`), eight headers per part, and 4,096
  bytes per part header. Files are parsed in bounded memory, not saved to disk.

API Gateway treats multipart as binary and base64-wraps the body when invoking
Lambda. A maximum-size body becomes **6,133,336 bytes**, leaving **158,120 bytes
(about 154 KiB)** for the event envelope within Lambda's **6 MiB / 6,291,456-byte**
synchronous request limit. The multipart budget also allows 100,000 bytes beyond
the file total for framing. These are deliberately near the practical ceiling,
not a guarantee for arbitrarily large headers/query strings/context metadata.
AWS can reject oversized events before Lambda runs, using its own error format.
API Gateway's 10 MB request limit does not increase Lambda's smaller limit.

No application resizing or recompression occurs: accepted file bytes reach the provider
unchanged. The provider may resize images for its vision model. If photos exceed these
limits, crop/compress them or submit fewer files. Full-resolution photos are not
guaranteed to fit merely because the request contains fewer than five images.

### Postman

1. Select **POST**, using `ApiPostUrl` (`…/prod/menus`), and **No Auth**.
2. Under **Body → form-data**, add a row named **`images`**, switch its type from
   **Text** to **File**, and choose your image. Repeat the row for more images.
3. Optionally add **`dummy`** as a **Text** field with `true` or `false`.
4. Remove any manually configured `Content-Type` header, especially the old
   `application/json` header. Postman must generate the multipart boundary.
5. Click **Send**. Live requests require enabled transcription and a configured
   secret, and incur charges from the selected provider. Never put API keys in Postman.

Future browser clients can use `FormData.append('images', file)` for each file
and send the FormData body directly without setting Content-Type themselves.
The local server does not modify the frontend or its API URL.

### Free dummy mode

Send a multipart **Text** field **`dummy=true`**, with no files. For compatibility,
**`{"dummy": true}`** with `Content-Type: application/json` is also supported.
JSON is limited to 1,024 bytes and is **only** accepted for this exact dummy shape.
No images or credentials are required; neither Secrets Manager nor either provider is
called. With multipart dummy mode, file contents are not transcribed or validated
as images, but body/file-size, dummy-field validation, framing, and count limits still
apply. Unknown fields are rejected.
The stored fixture is
[lambda/dummy-menu.json](lambda/dummy-menu.json).
It is a snapshot of the frontend's Rough & Tumble restaurant name and all 11 menu
sections, excluding frontend-only metadata. It is bundled with the API and does
not depend on frontend files at runtime.

**Breaking change:** JSON base64 image uploads are no longer accepted. Use multipart
file fields for live requests. `{}` is not dummy mode; use the explicit flag.
Live errors never silently fall back to fake data.

## Successful response

HTTP 200 returns the frontend's menu shape plus an `isDummy` marker:

```json
{
  "isDummy": false,
  "restaurantName": "Coffee Shop",
  "sections": [{"name": "Drinks", "items": ["Café latte. $5.", "Tea. $3."]}]
}
```

Item strings contain the visible names, descriptions, prices, sizes, modifiers,
and dietary notes. Original language and reading order are preserved. The model
is instructed to use a descriptive restaurant placeholder when the name is
absent; an unexpectedly blank model name falls back to `Restaurant`.

Readable text is retained and uncertain parts marked `[illegible]`. If no menu
is readable, the request fails instead. Unrelated images can be ignored when
other images contain a readable menu. Output structure is validated locally;
AI transcription can still make factual mistakes, especially on handwriting,
glare, distant text, and prices. Confirm important details against the original.

## Errors

Handler errors use `{"error":{"code":"no_menu_found","message":"..."}}`.
Clients should branch on `code`, not message wording.

| HTTP | Codes | Meaning |
| --- | --- | --- |
| 400 | `invalid_json`, `invalid_body`, `invalid_multipart`, `invalid_request`, `invalid_images`, `invalid_image`, `unsupported_image_type` | Correct the request, multipart framing, or image files. |
| 405 | `method_not_allowed` | Handler accepts POST only; gateway serves OPTIONS. |
| 413 | `request_too_large`, `images_too_large` | Resize/compress images. |
| 415 | `unsupported_media_type` | Use multipart/form-data for files; JSON is allowed only for dummy mode. |
| 422 | `no_menu_found`, `illegible_text`, `transcription_refused` | Take different/clearer menu photos. |
| 503 | `transcription_disabled`, `api_not_configured` | Operator must enable/configure the service. |
| 503 | `openai_not_funded` | OpenAI reports exhausted credits/billing quota; check API billing. |
| 503 | `openai_authentication_failed` | Invalid key or insufficient permissions. |
| 503 | `openai_rate_limited` | Try again later, with backoff. |
| 502 | `openai_failure`, `openai_unavailable`, `invalid_openai_response`, `incomplete_transcription` | Upstream, connectivity, output-format, or output-length failure. |
| 504 | `openai_timeout` | Try fewer/smaller images; see timing limits below. |
| 500 | `internal_error` | Unexpected application error. |

All handler responses include CORS headers and `Cache-Control: no-store`.
Lambda allows any origin and request headers (`Access-Control-Allow-Origin: *`
and `Access-Control-Allow-Headers: *`), ignoring `ALLOWED_ORIGIN`. This permits
non-credentialed requests from a locally running frontend after deploying the API
code. The frontend's FormData uploads do not require a preflight when sent without
custom headers. Credentialed browser requests are not supported by wildcard CORS.

CloudFormation templates are unchanged: API Gateway's separate OPTIONS mock and
gateway-generated errors still use their configured origin. Requests requiring a
preflight (such as JSON or custom-header requests) can still be blocked from
localhost by that gateway configuration; Lambda headers cannot override it.

AWS-level rejections (including throttling, oversized invocation events, and
unsupported routes/methods) use AWS's own response format, not this error envelope.

## Put API credentials in the shared application secret

### AWS deployment — recommended

1. In the [OpenAI API platform](https://platform.openai.com/api-keys), create a
   dedicated project API key with permission to create Responses (`POST /v1/responses`)
   and use the code-owned model. Existing restricted keys must allow Responses,
   not just Chat Completions. Enable API
   billing/credits for that project. A ChatGPT
   subscription does **not** fund API usage. Configure usage alerts and limits;
   project budget alerts alone are not a guaranteed hard spending cap.
2. In **AWS Secrets Manager → us-east-1 → Store a new secret → Other type of
   secret**, add this key/value pair:
   - **Key:** `OPENAI_API_KEY`
   - **Value:** paste the actual OpenAI key directly in the AWS console.
3. Use the default **`aws/secretsmanager`** encryption key. Name the secret
   **`menu_api_secrets`**. No automatic rotation is configured. This is separate from
   the existing `github_menu` secret. Never paste the key into chat, source code,
   CloudFormation parameters, build variables, or frontend configuration.
4. Wait until API deployments are idle, then update the infrastructure template
   with `MenuApiSecretName=menu_api_secrets` and initially
   `EnableLiveTranscription=false`. Acknowledge named IAM resources. Update the
  stack **before** releasing this API code: multipart needs the binary-media
  configuration and a new gateway deployment. Explicitly set the flag to false
  for setup; the current template's default is true.
5. After the stack update completes, push this API repository to the configured
   branch or release its pipeline. Verify pipeline success and POST
    `{"dummy":true}` to `ApiPostUrl`; expect `Rough & Tumble`.
6. Review the public-access warning below, then update the stack parameter
   **`EnableLiveTranscription=true`**. POST real images to verify transcription.
   No frontend deployment or frontend key configuration is needed.

**Existing stacks:** the API requires `MENU_API_SECRET_ID`, as set by the current
template; older secret environment variable names are not supported. Create
`menu_api_secrets` with the working OpenAI key, or set `MenuApiSecretName` to your
existing secret name. Update the stack, wait for completion, then release the
API code. If migrating from older configuration/code, plan a maintenance window:
live transcription may be unavailable until both updates finish. Dummy mode
remains available. Never overlap stack and pipeline deployments.

The secret can contain both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` and, in future, `GEMINI_API_KEY`.
Add provider fields to the same JSON secret without updating CloudFormation or
IAM. The handler reads `OPENAI_API_KEY` when `TRANSCRIPTION_PROVIDER="OPENAI"`
and `ANTHROPIC_API_KEY` when `TRANSCRIPTION_PROVIDER="ANTHROPIC"`. Gemini is not
implemented. Switching providers requires an API code release, not a stack update.
Other API credentials, such as a Google Places API key, and application secrets
can go into this same secret. Adding fields requires no stack update; using them
requires the corresponding application code.

The Lambda receives only the secret **name**, retrieves the key at runtime, and
caches it in memory for at most five minutes. IAM grants `GetSecretValue` for
that named secret only; build/frontend roles do not receive access. The key is
not copied into Lambda environment variables or the CloudFormation template.
For a customer-managed KMS key, additional narrowly scoped `kms:Decrypt` and key
policy permissions would be required; the supplied setup assumes the default key.

To rotate, create a replacement provider key, edit its corresponding field in the secret,
allow at least five minutes for warm Lambda caches to refresh, verify a real
request, then revoke the old key. No code release is required. Do not revoke the
old key before cache expiry unless immediate revocation is necessary.

### Local development

Use **Python 3.13** (matching Lambda/CodeBuild); the multipart dependency requires
at least Python 3.10, so the old Python 3.9 environment no longer suffices.
Install [requirements.txt](requirements.txt) into a virtual environment with
`python -m pip install -r requirements.txt`. Set the selected provider's
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the process
environment using a secure local secret mechanism and set
`ENABLE_LIVE_TRANSCRIPTION=true` for intentional paid calls. A local environment
key takes precedence over Secrets Manager. The production handler does not load
dotenv files; the local server above does. Do not commit any credentials.

Offline tests and dummy mode need neither a key nor AWS credentials. Run
`python -m unittest discover -s tests -v` from this repository root. Tests use the
real OpenAI and Anthropic SDKs with in-memory HTTP transports; no paid requests are made.
Multipart tests use a real HTTP client's encoder and simulate Gateway's binary
event wrapping, including byte-preservation and maximum-size upload tests.

## Configuration and operational limits

| Environment variable | Default / purpose |
| --- | --- |
| `ENABLE_LIVE_TRANSCRIPTION` | Handler fallback `false`; the current stack parameter defaults to `true`. Set it explicitly during deployment. |
| `MENU_API_SECRET_ID` | Shared application secret name or ARN; stack sets this from `MenuApiSecretName`, defaulting to `menu_api_secrets`. |
| `OPENAI_API_KEY` | Optional local override; never set by the stack. |
| `ANTHROPIC_API_KEY` | Optional local Anthropic override; never set by the stack. |
| `ALLOWED_ORIGIN` | `https://menu.littleapp.io`; stack uses the configured frontend domain. |

OpenAI defaults are defined by `DEFAULT_MODEL` and `DEFAULT_REASONING_EFFORT` in
[lambda/providers/openai.py](lambda/providers/openai.py); effort is omitted upstream for Mini.
Anthropic's model is defined by `DEFAULT_MODEL` in
[lambda/providers/anthropic.py](lambda/providers/anthropic.py). Choose a vision
model with structured output and access granted in your provider account. Not every
OpenAI model supports every reasoning effort. Edit the provider source to compare
settings on the automatically reloading local server; production changes require
an API code release. JSON supports the exact `{"dummy":true}` shape. No
infrastructure changes are needed to change provider settings.

Each provider call has a **20-second elapsed deadline**, capped further by remaining
Lambda time. Pending work is cancelled on timeout and the SDK never automatically
retries. Cancellation does not guarantee the provider stops processing/billing the
request. Lambda has 512 MB and a 28-second timeout; API Gateway waits 29 seconds.
Changing model or effort does not increase this deadline. Higher reasoning effort
may time out even on a single image. Long menus or five dense photos may time out:
send fewer photos. Supporting reliably longer jobs would require an asynchronous
API or an increased gateway timeout/quota,
not simply raising the Lambda timeout. Output is limited to 8,000 tokens; truncated
menus return an error rather than silently losing items.

**Public endpoint / cost warning:** CORS is not authentication. Anyone can call
the enabled endpoint and consume your API credits. The existing gateway throttle
(5 requests/second, burst 10) is best-effort and is not a spending cap or per-user
quota. Set `EnableLiveTranscription=false` to keep paid calls disabled. The current
template defaults to `true`; review it explicitly before deploying.
Choose code-configured models with cost in mind and restrict model
access in the OpenAI project if necessary. Add authentication, per-user quotas,
abuse controls, and enforced spending controls before broad public use. Turn the
stack flag off to disable new paid requests.

Images/menu text are not persisted by this application or logged, and raw upstream
errors/keys are not returned or logged. Images are sent to the selected provider
over HTTPS. OpenAI requests use `store=false`; Anthropic requests use the Messages
API. Neither path overrides its provider's applicable retention/abuse-monitoring
policies. Do not enable SDK HTTP debug logging or gateway body logging in production.

## Repository and deployment

Publish this directory's **contents** as the separate `menu-api` repository.
Commit the handler, provider package, dummy fixture, tests, and [requirements.txt](requirements.txt).
CodeBuild installs dependencies into a staging directory using Linux/Python 3.13
x86_64 wheels, runs offline tests, and packages dependencies plus the Lambda
directory's contents at ZIP root. Tests and documentation are not deployed. Do
not upload a package built with macOS native wheels.

**Multipart upgrade:** update CloudFormation and wait for completion, then deploy
the API code and updated requirements through its pipeline. The template adds
only `multipart/form-data` to `BinaryMediaTypes` (not `*/*`) and replaces the
gateway deployment with `ApiGatewayDeploymentV2`, updating the `prod` stage.
JSON responses and JSON dummy requests remain text. The Lambda proxy integration
needs no `ContentHandling` conversion or mapping template. Requests arrive with
`isBase64Encoded: true`; the handler rejects non-binary multipart events rather
than risking corrupted image bytes. Never base64-encode files yourself in Postman.
No upload bucket or new AWS permissions are required.

CloudFormation owns configuration and a historical inline bootstrap; the API
pipeline owns Version 2 code releases. The bootstrap is intentionally unchanged
and still has the old always-dummy behavior. Stack creation/replacement alone is
**not** a Version 2 deployment: wait for the API pipeline. Serialize infrastructure
and code releases, and rerun the API pipeline after a function replacement/reset.