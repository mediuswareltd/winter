# Getting started

Run **Winter** on your machine.

## 1. Quick start

### Docker (recommended)

Needs only [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on Windows/macOS).

```bash
git clone https://github.com/<you>/winter.git
cd winter
docker compose up -d --build
```

Open **http://localhost:8000**. That one command:

- builds the Winter image (Python, ffmpeg, CPU PyTorch)
- starts [Ollama](https://ollama.com) and pulls a small local model (`qwen2.5:3b`, ~1.9 GB), so you can
  chat **with no API key**
- keeps your videos, index and downloaded models in Docker volumes, so restarts are instant

The first start takes a few minutes for the downloads. Follow along with `docker compose logs -f`.
The header shows `LOCAL qwen2.5:3b` once the model is ready; until then, answers list the matching
passages.

**Use Claude instead** (better answers, and it can read the keyframes):

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY=...
docker compose up -d          # Claude is used whenever a key is set
```

**Common tweaks**, all set in `.env`:

| Want | Set |
|---|---|
| A different port | `WINTER_PORT=8080` |
| A bigger/smaller local model | `VIDEORAG_OLLAMA_MODEL=llama3.2:3b` (or `qwen2.5:7b` with 16 GB+ RAM) |
| No local model at all | run `docker compose up -d winter` (starts only the app) |
| NVIDIA GPU for the local model | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` |

**Day to day:**

```bash
docker compose stop                  # stop, keep everything
docker compose up -d --build         # update after a git pull
docker compose exec winter python -m videorag list    # CLI inside the container
docker compose down -v               # remove containers AND all data/models
```

### Without Docker

Requirements: Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) on `PATH`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e .

cp .env.example .env          # optional: add your ANTHROPIC_API_KEY
winter serve                  # open http://127.0.0.1:8000
```

The Whisper, BGE and CLIP models (about 1 GB in total) download on first use and are cached afterwards.

For answers without an API key, install [Ollama](https://ollama.com/download) and run
`ollama pull qwen2.5:3b`. Winter detects the running model automatically.

### How answers are generated

| Setup | Who writes the answer |
|---|---|
| `ANTHROPIC_API_KEY` set | Claude, using transcripts **and** keyframe images |
| No key, Ollama running with the model pulled | The local model, using transcripts only |
| Neither | No generated answer; Winter lists the best-matching moments |

Force one with `VIDEORAG_LLM=claude|ollama|none`. A 3B local model is fine for factual lookups but
cites less consistently than Claude.

### CLI

```bash
winter ingest talk.mp4 "https://www.youtube.com/watch?v=..."
winter list
winter search "rate limiting" -k 5
winter ask "What did they recommend for rate limiting?"
```

## 2. Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

The tests use stub embeddings, so they run in a few seconds with no GPU, models or API key.

## Next

- Full detail: [Guide](/guide)
- Source: [github.com/mediuswareltd/winter](https://github.com/mediuswareltd/winter)
