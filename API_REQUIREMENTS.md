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


# Menu API requirements — v1.1.0

Change the path of the current POST endpoint from "/menus" to "/menu_image_transcribe". There is no behavioral change to this endpoint.

Add a new POST endpoint: "/menu_online_search"

This endpoint accepts a short search query from the user. It then makes a call to an AI model, asking that AI model to search the internet for the menu the user is looking for. It returns a list of possible menus. It's response should follow a JSON schema containing the list of possible menus. Here is an example schema:

```json
{
  "possible_menus": [
    {
      "venue_identifier": "Marination Columbia City",
      "menu_identifier": "Main + Brunch Menu",
      "source_url": "https://static1.squarespace.com/static/5495bcb6e4b027d892329413/t/6ac678d5936d29404798f722/1791391957651/mcc+main+%2B+brunch+paper.pdf",
      "agent_instructions": "This user is looking for the Marination Columbia City Main + Brunch menu. It is located at https://static1.squarespace.com/static/5495bcb6e4b027d892329413/t/6ac678d5936d29404798f722/1791391957651/mcc+main+%2B+brunch+paper.pdf"
    },
    {
      "venue_identifier": "Marination 6th & Virginia",
      "menu_identifier": "Menu",
      "source_url": "https://static1.squarespace.com/static/5495bcb6e4b027d892329413/t/6a4536e0186f36294d4ac66a/1782920928047/downtown-menu.pdf",
      "agent_instructions": "The user is looking for the Marination 6th & Virginia menu. It is located at https://static1.squarespace.com/static/5495bcb6e4b027d892329413/t/6a4536e0186f36294d4ac66a/1782920928047/downtown-menu.pdf"
    },
    {
      "venue_identifier": "Marination T-Mobile Stadium",
      "menu_identifier": "Section 119 Menu",
      "source_url": "https://static1.squarespace.com/static/5495bcb6e4b027d892329413/t/6a453838a4b6944df73246d8/1782921272236/STADIUM+119+menu.pdf",
      "agent_instructions": "The user is looking for the Marination T-Mobile Stadium Section 119 menu. It is located at https://static1.squarespace.com/static/5495bcb6e4b027d892329413/t/6a453838a4b6944df73246d8/1782921272236/STADIUM+119+menu.pdf"
    },
    {
      "venue_identifier": "Marination T-Mobile Stadium",
      "menu_identifier": "Pen Menu",
      "source_url": "https://static1.squarespace.com/static/5495bcb6e4b027d892329413/t/6a453850b13d3d13fb861b3c/1782921296272/STADIUM+PEN+menu.pdf",
      "agent_instructions": "The user is looking for the Marination T-Mobile Stadium Pen menu. It is located at https://static1.squarespace.com/static/5495bcb6e4b027d892329413/t/6a453850b13d3d13fb861b3c/1782921296272/STADIUM+PEN+menu.pdf"
    }
  ]
}
```

The `venue_identifier` field must clearly identify the venue (including a location description, if necessary).
The `menu_identifier` field must clearly identify the specific possible menu at that venue location.
The `source_url` field contains the URL of the specific menu page or PDF and is included as a required string in the JSON schema sent to OpenAI. The frontend uses it for a source link rather than extracting a URL from agent instructions.
The `agent_instructions` field will not be displayed to users. Rather, once the user confirms which menu they want, this will be passed to another API endpoint for extraction and transcription. (We will build that endpoint in a later version).

The AI model should be instructed to respond with between 1 and 8 possible menus, depending on how confident it is that it found the user's desired menu.

Online menu search uses OpenAI only, with `gpt-6.1-sol`, low reasoning effort, the ultrafast service tier, required web search, and at most eight tool calls. Its settings are configured separately from image transcription. Keep both OpenAI and Anthropic support for image transcription.

Search response validation checks only the result list and its four string fields. Extra fields are ignored, and otherwise usable results are not rejected for blank strings, count, URL contents, or absent provider/tool/source metadata.

# Menu API requirements — v1.2.0

Add a new POST endpoint: "/menu_online_transcribe"

This endpoint accepts exactly one list item of the possible menus
identified by the /menu_online_search endpoint. It will also receive the origina query from the user. From this information (particularly the `agent_instructions` field), this endpoint is responsible for finding the identified menu online, and returning the same structured JSON that the "menu_image_transcribe" returns.

Use an AI model to turn the API input into the structured JSON.
Initially, we will only support OpenAI for this enpoint.
The specific model should be configurable separately from the other two OpenAI calls.
Defaults are `gpt-6.1-sol`, low reasoning effort, and the ultrafast service tier.
Forward only `agent_instructions` as user text; it should include the menu URL.
Web search is unrestricted and optional (`tool_choice="auto"`), with at most three
tool calls. Keep source-URL hostname validation. The online-transcription deadline
is 30 seconds, capped by remaining Lambda time minus three seconds; image and
search deadlines remain 20 seconds.

This is the second and final neccessary API endpoint which enables a client to search with a simple user defined query and end up with a structured JSON menu.

Wire contract: send JSON `{"query":"original query","menu":{"venue_identifier":"...","menu_identifier":"...","source_url":"...","agent_instructions":"..."}}`. The selected result has four fields, including the source URL added in v1.1.0. Reuse `ENABLE_LIVE_TRANSCRIPTION` / CloudFormation `EnableLiveTranscription` for this endpoint; do not add a separate enablement flag.

# Menu API requirements - v1.3.0

**Direct online file transcription**
Update /menu_online_transcribe to choose its OpenAI input based on source_url:

PDF URLs: use input_file with file_url, without web search.
JPEG, PNG, and WebP URLs: use input_image with image_url, without web search.
Webpage URLs: retain the current web-search path.
Identify file extensions case-insensitively from the URL path, ignoring query strings and fragments. Forward the complete URL unchanged. URLs without a recognized extension use the webpage path.

Include agent_instructions as user text. Keep the existing online-transcription settings, deadline, schema, validation, and API contracts.

Do not download or upload files server-side, or add fallback attempts. If OpenAI cannot retrieve or read the file, return a safe, recoverable error.

Add offline tests for routing, OpenAI request payloads, and retrieval failures.

