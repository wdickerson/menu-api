# Menu API v1.0.0

Python Lambda behind API Gateway. `POST /menus` transcribes menu photos through
OpenAI or Anthropic. See [API_REQUIREMENTS.md](API_REQUIREMENTS.md) for the contract.

## Local development

From this directory, using Python 3.13:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python dev_server.py
```

The reloading server listens at http://127.0.0.1:8000/menus (`--port` overrides).
Dummy mode needs no credentials or paid calls:

```sh
curl http://127.0.0.1:8000/menus -H 'Content-Type: application/json' --data '{"dummy":true}'
```

Run offline tests with `python -m unittest discover -s tests -v`.

## Configuration and releases

- Choose `OPENAI` or `ANTHROPIC` via `TRANSCRIPTION_PROVIDER` in
  [lambda/index.py](lambda/index.py); model/request settings live in the
  [OpenAI](lambda/providers/openai.py) and [Anthropic](lambda/providers/anthropic.py) modules.
- Paid calls require `ENABLE_LIVE_TRANSCRIPTION=true` and the selected provider's
  `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Local environment values override the
  JSON secret selected by `MENU_API_SECRET_ID` (stack default: `menu_api_secrets`).
  Local dotenv loading is development-only; never commit keys or put them in the frontend.
- The API is unauthenticated. CORS and gateway throttling are not spending caps.
  The handler defaults live calls off, but the stack defaults them on; set the flag explicitly.
- Run tests before release: deployment does not run them. The API pipeline packages
  [requirements.txt](requirements.txt) and the Lambda contents at ZIP root using
  Linux/Python 3.13 x86_64 dependencies, not macOS wheels.
- CloudFormation owns infrastructure/configuration; the pipeline owns application code.
  Do not overlap stack and code deployments. Rerun the pipeline after Lambda replacement.
  