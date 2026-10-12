# Menu API — Version 1

Python Lambda (`index.lambda_handler`) behind API Gateway REST **POST /menus**.
Transcribes 1–10 menu images using OpenAI vision and strict structured output.
Printed menus, counter boards, food-truck posters, and handwritten beer lists
are supported through the transcription prompt. No database or image storage.
The frontend is unchanged and does not call this API yet.

## Request

POST to the stack's `ApiPostUrl` using **multipart/form-data** with actual image
files. Let the HTTP client generate the Content-Type header and boundary.

| Field name | Field type | Value |
| --- | --- | --- |
| `images` | File | A JPEG, PNG, or WebP image. Repeat this exact field name for 1–10 files. |
| `dummy` | Text (optional) | Exactly `true` or `false`; defaults to `false`. |
| `model` | Text (optional) | OpenAI model ID; defaults to `gpt-6-luna`. |
| `reasoning_effort` | Text (optional) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; defaults to `none`. |

Options apply only to that request. Defaults are defined in Lambda code, not the
template or environment. Not every model supports every effort; choose a vision
model with strict structured output and access granted in your OpenAI project.
For Sol, explicitly send `model=gpt-6.1-sol` and `reasoning_effort=low`.

**GPT-4.1 Mini:** send both `model=gpt-4.1-mini` and `reasoning_effort=none`.
For this model only, the effort must be explicitly provided as `none`; missing
or other values return HTTP 400 before any paid call. The server then omits the
reasoning parameter entirely from the OpenAI request, because Mini does not
support it. Other models still receive the selected effort.

Images must belong to the same restaurant; order them in menu reading order.
All images are sent together in **one** OpenAI request. The server converts the
original bytes into data URLs internally; clients no longer encode images.

- Use `images`, **not** `images[]`. Each part needs a nonempty filename. Filenames
   are never used as paths or sent to OpenAI. Unknown fields and duplicate text
   options are rejected. Model IDs are nonempty ASCII identifiers of at most 128
   characters. Keys, prompts, and API-host overrides are never accepted.
- File types: JPEG, PNG, or WebP. No remote URLs, base64 text fields, JSON image
  arrays, PDFs, GIFs, or HEIC. Convert HEIC camera photos to JPEG first. Declared
  image MIME types must match file signatures. Missing/generic
  `application/octet-stream` types are inferred from signatures, not filenames.
  OpenAI performs full image decoding; local signature checks alone cannot prove
  a file is valid or readable.

### Upload limits (no S3 required)

- **4,500,000 bytes (4.5 decimal MB) combined original image files**, whether
  uploading one file or ten. This replaces the old 2.8 MB image limit.
- **4,600,000 bytes (4.6 decimal MB) for the complete multipart body**, including
  boundaries, part headers, and text fields.
- At most 13 parts (ten files plus `dummy`, `model`, and `reasoning_effort`), eight headers per part, and 4,096
  bytes per part header. Files are parsed in bounded memory, not saved to disk.

API Gateway treats multipart as binary and base64-wraps the body when invoking
Lambda. A maximum-size body becomes **6,133,336 bytes**, leaving **158,120 bytes
(about 154 KiB)** for the event envelope within Lambda's **6 MiB / 6,291,456-byte**
synchronous request limit. The multipart budget also allows 100,000 bytes beyond
the file total for framing. These are deliberately near the practical ceiling,
not a guarantee for arbitrarily large headers/query strings/context metadata.
AWS can reject oversized events before Lambda runs, using its own error format.
API Gateway's 10 MB request limit does not increase Lambda's smaller limit.

No application resizing or recompression occurs: accepted file bytes reach OpenAI
unchanged. OpenAI may resize images for its vision model. If photos exceed these
limits, crop/compress them or submit fewer files. Full-resolution photos are not
guaranteed to fit merely because the request contains fewer than ten images.

### Postman

1. Select **POST**, using `ApiPostUrl` (`…/prod/menus`), and **No Auth**.
2. Under **Body → form-data**, add a row named **`images`**, switch its type from
   **Text** to **File**, and choose your image. Repeat the row for more images.
3. Optionally add **`dummy`**, **`model`**, and **`reasoning_effort`** as **Text**
   fields. To try Mini, set `model` to `gpt-4.1-mini` and `reasoning_effort` to `none`.
4. Remove any manually configured `Content-Type` header, especially the old
   `application/json` header. Postman must generate the multipart boundary.
5. Click **Send**. Live requests require enabled transcription and a configured
   secret, and incur OpenAI charges. Never put the OpenAI key in Postman.

Future browser clients can use `FormData.append('images', file)` for each file
and send the FormData body directly without setting Content-Type themselves.
This change does not integrate or modify the frontend.

### Free dummy mode

Send a multipart **Text** field **`dummy=true`**, with no files. For compatibility,
**`{"dummy": true}`** with `Content-Type: application/json` is also supported.
JSON is limited to 1,024 bytes and is **only** accepted for this exact dummy shape.
No images or credentials are required; neither Secrets Manager nor OpenAI is
called. With multipart dummy mode, file contents are not transcribed or validated
as images, but body/file-size, option validation, framing, and count limits still
apply. Valid model/effort options have no effect on the returned dummy menu.
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
| 400 | `unsupported_model_options` | OpenAI rejected an unavailable model or incompatible reasoning effort. |
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

## Put the OpenAI key in the right place

### AWS deployment — recommended

1. In the [OpenAI API platform](https://platform.openai.com/api-keys), create a
   dedicated project API key with permission to create Chat Completions and use
   `gpt-6-luna`. Enable API billing/credits for that project. A ChatGPT
   subscription does **not** fund API usage. Configure usage alerts and limits;
   project budget alerts alone are not a guaranteed hard spending cap.
2. In **AWS Secrets Manager → us-east-1 → Store a new secret → Other type of
   secret**, add exactly this key/value pair:
   - **Key:** `OPENAI_API_KEY`
   - **Value:** paste the actual OpenAI key directly in the AWS console.
3. Use the default **`aws/secretsmanager`** encryption key. Name the secret
   **`openai_menu`**. No automatic rotation is configured. This is separate from
   the existing `github_menu` secret. Never paste the key into chat, source code,
   CloudFormation parameters, build variables, or frontend configuration.
4. Wait until API deployments are idle, then update the infrastructure template
   with `OpenAISecretName=openai_menu` and initially
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

The Lambda receives only the secret **name**, retrieves the key at runtime, and
caches it in memory for at most five minutes. IAM grants `GetSecretValue` for
that named secret only; build/frontend roles do not receive access. The key is
not copied into Lambda environment variables or the CloudFormation template.
For a customer-managed KMS key, additional narrowly scoped `kms:Decrypt` and key
policy permissions would be required; the supplied setup assumes the default key.

To rotate, create a replacement OpenAI key, edit `OPENAI_API_KEY` in the secret,
allow at least five minutes for warm Lambda caches to refresh, verify a real
request, then revoke the old key. No code release is required. Do not revoke the
old key before cache expiry unless immediate revocation is necessary.

### Local development

Use **Python 3.13** (matching Lambda/CodeBuild); the multipart dependency requires
at least Python 3.10, so the old Python 3.9 environment no longer suffices.
Install [requirements.txt](requirements.txt) into a virtual environment with
`python -m pip install -r requirements.txt`. Set `OPENAI_API_KEY` in the process
environment using a secure local secret mechanism and set
`ENABLE_LIVE_TRANSCRIPTION=true` for intentional paid calls. A local environment
key takes precedence over Secrets Manager. The application does not load dotenv
files automatically; do not commit any credentials.

Offline tests and dummy mode need neither a key nor AWS credentials. Run
`python -m unittest discover -s tests -v` from this repository root. Tests use the
real OpenAI SDK with an in-memory HTTP transport; no paid requests are made.
Multipart tests use a real HTTP client's encoder and simulate Gateway's binary
event wrapping, including byte-preservation and maximum-size upload tests.

## Configuration and operational limits

| Environment variable | Default / purpose |
| --- | --- |
| `ENABLE_LIVE_TRANSCRIPTION` | Handler fallback `false`; the current stack parameter defaults to `true`. Set it explicitly during deployment. |
| `OPENAI_SECRET_ID` | Secret name or ARN; stack sets `openai_menu`. |
| `OPENAI_API_KEY` | Optional local override; never set by the stack. |
| `ALLOWED_ORIGIN` | `https://menu.littleapp.io`; stack uses the configured frontend domain. |

Defaults (`gpt-6-luna`, `none`) are defined in [lambda/index.py](lambda/index.py).
Multipart `model` and `reasoning_effort` fields override them for one request;
CloudFormation and environment variables do not select either option. Changing
defaults requires a code release; comparing models needs only request fields.
JSON remains exclusively for the exact `{"dummy":true}` shape, not model options
or live uploads. No infrastructure changes are needed for these request options.

The OpenAI call has a **20-second elapsed deadline**, capped further by remaining
Lambda time. Pending work is cancelled on timeout and the SDK never automatically
retries. Cancellation does not guarantee OpenAI stops processing/billing the
request. Lambda has 512 MB and a 28-second timeout; API Gateway waits 29 seconds.
Changing model or effort does not increase this deadline. Higher reasoning effort
may time out even on a single image. Long menus or ten dense photos may time out:
send fewer photos. Supporting reliably longer jobs would require an asynchronous
API or an increased gateway timeout/quota,
not simply raising the Lambda timeout. Output is limited to 8,000 tokens; truncated
menus return an error rather than silently losing items.

**Public endpoint / cost warning:** CORS is not authentication. Anyone can call
the enabled endpoint and consume your API credits. The existing gateway throttle
(5 requests/second, burst 10) is best-effort and is not a spending cap or per-user
quota. Set `EnableLiveTranscription=false` to keep paid calls disabled. The current
template defaults to `true`; review it explicitly before deploying.
Caller-selected models may be more expensive than the default. Restrict model
access in the OpenAI project if necessary. Add authentication, per-user quotas,
abuse controls, and enforced spending controls before broad public use. Turn the
stack flag off to disable new paid requests.

Images/menu text are not persisted by this application or logged, and raw upstream
errors/keys are not returned or logged. Images are sent to OpenAI over HTTPS with
`store=false`; this does not override OpenAI's applicable retention/abuse-monitoring
policies. Do not enable SDK HTTP debug logging or gateway body logging in production.

## Repository and deployment

Publish this directory's **contents** as the separate `menu-api` repository.
Commit the handler, dummy fixture, tests, and [requirements.txt](requirements.txt).
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
pipeline owns Version 1 code releases. The bootstrap is intentionally unchanged
and still has the old always-dummy behavior. Stack creation/replacement alone is
**not** a Version 1 deployment: wait for the API pipeline. Serialize infrastructure
and code releases, and rerun the API pipeline after a function replacement/reset.