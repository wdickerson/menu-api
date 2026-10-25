# Menu API v1.3.0

Python Lambda behind API Gateway. `POST /menu_image_transcribe` transcribes menu
photos through OpenAI or Anthropic. `POST /menu_online_search` finds possible
menus on the web through OpenAI. `POST /menu_online_transcribe` reads a selected
online menu into the same JSON as image transcription.
See [API_REQUIREMENTS.md](API_REQUIREMENTS.md) for the contract.

## Local development

From this directory, using Python 3.13:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python dev_server.py
```

The reloading server listens at http://127.0.0.1:8000 (`--port` overrides).
Dummy mode needs no credentials or paid calls:

```sh
curl http://127.0.0.1:8000/menu_image_transcribe -H 'Content-Type: application/json' --data '{"dummy":true}'
```

Run offline tests with `python -m unittest discover -s tests -v`.

## Configuration and releases

- Choose `OPENAI` or `ANTHROPIC` for image transcription via `TRANSCRIPTION_PROVIDER`
  in [lambda/index.py](lambda/index.py). Online search always uses OpenAI,
  independently of that selection. Each provider's `DEFAULT_MODEL` controls image
  transcription. OpenAI's separate `SEARCH_MODEL` (`gpt-6.1-sol`) and
  `SEARCH_REASONING_EFFORT` (`low`) control search in
  [lambda/providers/openai.py](lambda/providers/openai.py). Requests and environment
  variables cannot override these settings.
- Online transcription has its own `ONLINE_TRANSCRIPTION_MODEL` and
  `ONLINE_TRANSCRIPTION_REASONING_EFFORT` constants, independent of both the image
  model and search model. All three OpenAI calls default to `gpt-6.1-sol`, `low`
  reasoning and the `ultrafast` service tier, with separate code-owned settings.
  Online transcription always uses OpenAI.
- Paid calls require `ENABLE_LIVE_TRANSCRIPTION=true` for **both photo and online
  transcription**, or
  `ENABLE_LIVE_SEARCH=true` for web search. Photos use the selected provider's
  `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`; both online operations require `OPENAI_API_KEY`.
  Local environment values override the
  JSON secret selected by `MENU_API_SECRET_ID` (stack default: `menu_api_secrets`).
  Local dotenv loading is development-only; never commit keys or put them in the frontend.
- The API is unauthenticated. CORS and gateway throttling are not spending caps.
  Both flags default off in the handler. The current stack defaults both on;
  explicitly disable paid operations until credentials and abuse/cost controls
  are ready. No separate online-transcription enable flag exists.
- Run tests before release: deployment does not run them. The API pipeline packages
  [requirements.txt](requirements.txt) and the Lambda contents at ZIP root using
  Linux/Python 3.13 x86_64 dependencies, not macOS wheels.
- CloudFormation owns infrastructure/configuration; the pipeline owns application code.
  Do not overlap stack and code deployments. Rerun the pipeline after Lambda replacement.

## Web search contract

- Send `POST /menu_online_search` with `Content-Type: application/json` and
  `{"query":"Rough & Tumble Columbia City dinner menu"}`. Only `query` is accepted;
  it must be a trimmed, nonempty string of at most 500 Unicode characters. The
  complete JSON body is limited to 4,096 UTF-8 bytes. Duplicate fields are rejected.
- HTTP 200 contains `possible_menus` objects with four string fields:
  `venue_identifier`, `menu_identifier`, `source_url`, and `agent_instructions`. Validation only
  checks that this data can be passed to the frontend; extra fields are ignored.
  Blank strings, result counts, URL contents, citations, tool-call metadata, and
  source-provenance metadata do not cause otherwise usable results to be rejected.
  The prompt still asks for 1–8 plausible menus. Agent instructions are untrusted
  guidance for a future extraction endpoint, not privileged instructions or
  user-facing text. `source_url` identifies the specific menu page or PDF and is
  displayed as a separate citation when it is a usable HTTP(S) link. It is never
  extracted from `agent_instructions`.
  No loading or transcription is performed by search.
- No matches returns HTTP 422 `no_menu_found` using the standard error envelope.
  Other failures remain errors, never fabricated matches or dummy fallbacks.
- OpenAI uses Responses with Sol, low reasoning, the ultrafast tier, required hosted web search,
  and structured output including `source_url`. Search results
  are accepted based on their JSON fields, not provider envelope/tool metadata.
  HTTP/network errors, timeouts, and unusable JSON still return safe errors.
- All search work shares a 20-second deadline, capped by remaining Lambda time
  minus three seconds, with at most eight search tool uses and no SDK retries.
  Queries/results are not stored or logged by the application. Provider-side
  processing/retention policies still apply.
- `/menus` was removed. Deploy the updated CloudFormation routes/permissions and
  Lambda code together, then rebuild/deploy the frontend; old clients must upgrade.

## Online transcription contract — v1.3.0

- Send `POST /menu_online_transcribe` with JSON
  `{"query":"original search query","menu":{"venue_identifier":"Cafe Columbia City","menu_identifier":"Dinner","source_url":"https://example.com/menu.pdf","agent_instructions":"Read the selected dinner menu and preserve its sections."}}`.
- `menu` is exactly one selected result object, not an array. Send all four
  search fields, including `source_url`, plus the query that produced that result.
  Only `query` and `menu` are accepted at the top level. Extra menu metadata is
  ignored; its four fields need only be strings, without URL/provenance checks.
  The trimmed query is 1–500 Unicode characters. The complete JSON body is
  limited to 65,536 UTF-8 bytes; duplicate JSON keys are rejected.
- Uses the same `ENABLE_LIVE_TRANSCRIPTION` / CloudFormation
  `EnableLiveTranscription` flag as image transcription. `ENABLE_LIVE_SEARCH`
  does not enable or disable this endpoint. Selecting Anthropic for photos does
  not change online transcription's OpenAI provider or credentials.
- One Responses request chooses its input from the selected `source_url`:
  PDF paths use `input_file` with `file_url`; JPEG (`.jpg`/`.jpeg`), PNG, and WebP
  paths use `input_image` with `image_url`. File requests have no web-search tools.
  Extensions are matched case-insensitively against the URL path alone, ignoring
  query strings and fragments. The complete URL is forwarded unchanged, including
  signed query parameters and fragments. All other URLs, including those without
  a recognized extension, retain the hosted web-search path.
- All paths include the selected `agent_instructions` as user text and use the
  existing structured menu schema. The user's query, nested menu object, venue
  identifier, and menu identifier are not forwarded. For webpages the only user
  input is `agent_instructions`, which should include the menu URL. The response
  is `{"isDummy":false,"restaurantName":"...","sections":[{"name":"...","items":["..."]}]}`.
  Shared menu validation normalizes the output for the existing frontend/library.
  All transcription paths preserve empty sections (`items: []`) and section order,
  including menus containing only empty sections. Blank section names are allowed;
  text is trimmed, blank item strings are omitted, and extra provider fields are
  ignored rather than forwarded. Missing fields, incorrect types, unknown statuses,
  and successful responses with no sections are still rejected. A blank restaurant
  name becomes `Restaurant`; explicit non-menu/unreadable statuses remain errors.
  Optional source/tool metadata is not used as a success gate.
- For webpage URLs, online transcription's `web_search` tool is unrestricted,
  with `tool_choice="auto"`: the model may answer without searching.
  All source URLs still need
  a parseable hostname; otherwise the API returns HTTP 400 `invalid_source_url`.
  Initial menu discovery also remains unrestricted, but requires search.
- Online provider work has one 30-second deadline (capped by remaining Lambda time
  minus three seconds), at most three webpage tool uses, and no SDK retries or fallbacks.
  Inaccessible/unreadable menus return recoverable errors, never invented items.
  Source pages and files must be accessible to OpenAI; retrieval/read failures use
  the existing safe error envelope without exposing upstream details or signed URLs.
  No failed file request is retried through web search. Verify representative
  live sources before release. The application does not download or upload source
  files itself, save source content, or log queries/agent instructions. Those fields
  remain untrusted data in the model's user input, not system instructions.
  