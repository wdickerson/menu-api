# Menu API requirements — v1.0.0

## Requests and responses

- One synchronous `POST /menus` transcribes menu photos into screen-reader-friendly text.
- Live requests use multipart form data: 1–5 repeated `images` files (JPEG, PNG, WebP)
	and optional `dummy` text, exactly `true` or `false` (default). Reject unknown fields,
	duplicate flags, empty filenames, and mismatched/unsupported image signatures.
- Limits: 4,500,000 combined image bytes, 4,600,000 multipart body bytes, six parts.
	Accepted image bytes are forwarded without resizing; URLs and JSON image uploads are unsupported.
- Multipart `dummy=true`, or JSON containing only `{"dummy":true}` (at most 1,024 bytes),
	returns the bundled sample without credentials or provider calls; no images are required.
- HTTP 200 returns `isDummy`, `restaurantName`, and `sections` with `name` and string `items`.
	Validate nonempty live sections/items; trim text and replace a blank restaurant name with `Restaurant`.
- Handler errors use `{"error":{"code":"...","message":"..."}}` with appropriate HTTP status
	for invalid/oversized input, unreadable/non-menu images, refusal, configuration, provider, or timeout failures.

## Processing and operation

- Code selects `OPENAI` (Responses API) or `ANTHROPIC` (Messages API); requests cannot override it.
	Send all images in one structured-output request; no automatic retries, provider fallback, or dummy fallback.
- Live calls require explicit enablement and the selected provider's key from the environment
	or the JSON Secrets Manager secret selected by `MENU_API_SECRET_ID`; environment keys take precedence.
- Provider work has a 20-second deadline, capped by remaining Lambda time minus three seconds.
- Handler responses are non-cacheable and allow noncredentialed cross-origin requests.
	Gateway preflight/errors use the configured origin. The public endpoint has no authentication;
	gateway throttling is best-effort, not a spending cap. The application does not store images.
