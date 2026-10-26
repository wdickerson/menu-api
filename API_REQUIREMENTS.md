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
  Require at least one live section, but allow empty item lists and blank section names.
  Preserve section order, trim text, omit blank item strings, ignore extra provider fields,
  and replace a blank restaurant name with `Restaurant`. Required fields and their types
  are still validated; explicit `no_menu` and `illegible_text` statuses remain errors.
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


# Menu API requirements — v2.0.0

## Image transcription and online search

- Replace `POST /menus` with `POST /menu_image_transcribe`; retain its behavior and
	OpenAI/Anthropic provider support.
- `POST /menu_online_search` accepts a JSON `query` and searches online for matching menus.
	Instruct the model to return 1–8 candidates in `possible_menus`.
- Each candidate has four required string fields: `venue_identifier` identifies the venue
	and location, `menu_identifier` identifies the specific menu, `source_url` links to the
	menu page or file, and `agent_instructions` provides private transcription guidance.
- Require these fields in the search output schema. Validate only the result list and
	four string fields; ignore extras and do not reject otherwise usable results for blank
	strings, count, URL contents, or absent provider/tool/source metadata.
- Search uses OpenAI only, defaulting to `gpt-6.1-sol`, low reasoning effort, and the
	ultrafast service tier, with required unrestricted web search and at most eight tool calls.
	Configure search settings separately from image transcription.

## Online transcription

- `POST /menu_online_transcribe` accepts JSON `{"query":"original query","menu":{...}}`,
	where `menu` is one selected search result containing all four fields. Return the same
	structured menu JSON as image transcription, with the same schema and validation.
- Use OpenAI only, with separately configurable settings defaulting to `gpt-6.1-sol`,
	low reasoning effort, and the ultrafast service tier. Forward only `agent_instructions`
	as user text; it should include the menu URL. Retain source-URL hostname validation.
- Select input from `source_url`: PDF uses `input_file` / `file_url`; JPEG, PNG, and WebP
	use `input_image` / `image_url`. These file paths do not use web search. Other URLs use
	unrestricted optional web search (`tool_choice="auto"`), with at most three tool calls.
- Identify file extensions case-insensitively from the URL path, ignoring query strings
	and fragments; forward the complete URL unchanged. Do not download or upload files
	server-side or attempt fallbacks. Retrieval/read failures return safe, recoverable errors.
- Reuse `ENABLE_LIVE_TRANSCRIPTION` / CloudFormation `EnableLiveTranscription`.
	Online transcription has a 30-second deadline, capped by remaining Lambda time minus
	three seconds; image transcription and search retain 20-second deadlines.
- Cover URL routing, OpenAI request payloads, and retrieval failures with offline tests.

