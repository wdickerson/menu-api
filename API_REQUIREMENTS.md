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

Online menu search uses OpenAI only, with `gpt-6-luna` and medium reasoning effort. Its model and effort are configured separately from image transcription. Keep both OpenAI and Anthropic support for image transcription.

Search response validation checks only the result list and its four string fields. Extra fields are ignored, and otherwise usable results are not rejected for blank strings, count, URL contents, or absent provider/tool/source metadata.
