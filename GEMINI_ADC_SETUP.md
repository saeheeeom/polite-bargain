# Gemini access setup (Vertex AI + ADC)

This project's GCP org **disallows issuing Gemini API keys**, so Gemini calls
go through **Vertex AI**, authenticated with **Application Default
Credentials (ADC)** instead of a key in `.env`. You've already been added as
a collaborator on the GCP project this bills against
(`project-22d4bba1-e867-49bf-8a6`, shared $300 credit) — this doc is the
setup you need to actually call the API from your machine.

`OPENROUTER_API_KEY` is unrelated to all of this — OpenRouter still uses a
regular key, get your own from https://openrouter.ai and put it in `.env` as
before.

## What is ADC, briefly

Instead of a static key, Google client libraries look for credentials tied to
your identity: a file saved locally by `gcloud auth application-default
login`, or (on GCP infra) an attached service account. IAM on the GCP project
controls what you can actually do, rather than a secret string that works for
anyone who has it.

## Setup steps

1. **Install the gcloud CLI** if you don't have it.
   - macOS: `brew install --cask google-cloud-sdk`
   - Other platforms: https://cloud.google.com/sdk/docs/install

2. **Log in and save ADC credentials** (opens a browser — use the Google
   account Saehee added to the project):
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Set the active project:**
   ```bash
   gcloud config set project project-22d4bba1-e867-49bf-8a6
   ```
   If you see a warning like *"Your active project does not match the quota
   project in your local Application Default Credentials file"*, run:
   ```bash
   gcloud auth application-default set-quota-project project-22d4bba1-e867-49bf-8a6
   ```
   This matters — it's what determines which project actually gets billed
   when the Python client library makes a call. (Vertex AI API is already
   enabled on the project, you shouldn't need to touch that.)

4. **Add to your `.env`** (no `GEMINI_API_KEY` needed anymore):
   ```
   GOOGLE_CLOUD_PROJECT=project-22d4bba1-e867-49bf-8a6
   GOOGLE_CLOUD_LOCATION=global
   ```
   Use `global`, not `us-central1` — we hit a `404 Publisher model ... was
   not found` error in `us-central1`; `gemini-3.6-flash` is currently only
   served from the `global` Vertex AI endpoint.

5. **Verify it works:**
   ```bash
   python -m src.agents
   ```
   This runs the smoke test at the bottom of `agents.py` — you should get a
   real JSON negotiation-turn response back, not an auth error.

## Running the experiment

```bash
python -m src.run_experiment --provider gemini --workers 8
```

`--workers` runs multiple negotiations concurrently (see `run_experiment.py`
— it's resume-safe, so if it's interrupted just re-run the same command and
it'll skip whatever's already saved to `results/transcripts/`). Since we're
sharing the same $300 credit: actual cost is tiny (the full 432-negotiation
experiment is roughly $1-2 based on observed token usage), so budget isn't a
real constraint — just worth coordinating in case we're both running large
batches at the same time.
