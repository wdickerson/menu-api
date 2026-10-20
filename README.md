# Menu API v1.1.0

Python Lambda behind API Gateway. `POST /menu_image_transcribe` transcribes menu
photos through OpenAI or Anthropic. `POST /menu_online_search` finds possible
menus on the web through OpenAI. See [API_REQUIREMENTS.md](API_REQUIREMENTS.md) for the contract.

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
  transcription. OpenAI's separate `SEARCH_MODEL` (`gpt-6-luna`) and
  `SEARCH_REASONING_EFFORT` (`medium`) control search in
  [lambda/providers/openai.py](lambda/providers/openai.py). Requests and environment
  variables cannot override these settings.
- Paid calls require `ENABLE_LIVE_TRANSCRIPTION=true` for photos or
  `ENABLE_LIVE_SEARCH=true` for web search. Photos use the selected provider's
  `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`; search always requires `OPENAI_API_KEY`.
  Local environment values override the
  JSON secret selected by `MENU_API_SECRET_ID` (stack default: `menu_api_secrets`).
  Local dotenv loading is development-only; never commit keys or put them in the frontend.
- The API is unauthenticated. CORS and gateway throttling are not spending caps.
  Both flags default off in the handler. The stack defaults transcription on and
  search off; explicitly opt into search with `EnableLiveSearch=true`.
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
- OpenAI uses Responses with Luna, medium reasoning, required hosted web search,
  and structured output including `source_url`. Search results
  are accepted based on their JSON fields, not provider envelope/tool metadata.
  HTTP/network errors, timeouts, and unusable JSON still return safe errors.
- All search work shares a 20-second deadline, capped by remaining Lambda time
  minus three seconds, with at most three search tool uses and no SDK retries.
  Queries/results are not stored or logged by the application. Provider-side
  processing/retention policies still apply.
- `/menus` was removed. Deploy the updated CloudFormation routes/permissions and
  Lambda code together, then rebuild/deploy the frontend; old clients must upgrade.
  