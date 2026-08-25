# RAG Pipeline

A local RAG (retrieval-augmented generation) system for querying a large PDF collection. Everything below is done through `docker compose` and `curl` — no Python setup on your machine, no CLI juggling.

## Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | Runs the whole thing as one service |
| `Dockerfile` | Builds the image (Python + torch CPU + embedding model baked in) |
| `app.py` | The HTTP API (upload, index, ask) |
| `rag_pipeline.py` | Core logic: parsing, chunking, embedding, retrieval — imported by `app.py` |
| `generate_test_pdfs.py` | Creates 20 synthetic test PDFs with known facts (optional, for testing) |
| `test_pdfs/` | Those 20 test PDFs + `_answer_key.json` with the ground-truth facts |

## 1. One-time setup

```bash
cp env.example .env
```

Edit `.env`:
- `LLM_PROVIDER` — `anthropic` (default) or `gemini`
- `LLM_KEY` — API key matching whichever provider you picked ([Anthropic console](https://console.anthropic.com) / [Google AI Studio](https://aistudio.google.com/apikey))
- `API_KEY` — pick any secret string; this protects your endpoints since anyone who can reach the server can otherwise trigger paid API calls

To switch providers later, just edit `.env` and restart:
```bash
docker compose up -d   # picks up the new .env, no rebuild needed
```

## 2. Start it

```bash
docker compose up -d --build
```

First build takes a few minutes (downloads torch + the embedding model, baked into the image so it only happens once). `-d` runs it in the background; drop it to watch logs live.

Check it's alive:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## 3. Upload PDFs via curl

```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: $(grep API_KEY .env | cut -d= -f2)" \
  -F "file=@test_pdfs/company_01.pdf"
```

Repeat per file. To upload every test PDF in one go:
```bash
for f in test_pdfs/*.pdf; do
  curl -s -X POST http://localhost:8000/upload \
    -H "X-API-Key: $(grep API_KEY .env | cut -d= -f2)" \
    -F "file=@$f" > /dev/null
  echo "uploaded $f"
done
```

For your real 2000+ PDFs, looping curl calls one file at a time works but is slow for that volume — dropping the files directly into the `./pdfs` folder that `docker-compose.yml` mounts (it appears inside the container at `/app/pdfs` automatically) is faster for a big batch. Either path lands files in the same place; use whichever is more convenient for a given batch.

## 4. Trigger indexing

```bash
curl -X POST http://localhost:8000/index \
  -H "X-API-Key: $(grep API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Empty `{}` body indexes everything in the shared PDFs folder. Runs in the background so this returns immediately — check progress:

```bash
curl http://localhost:8000/index/status
```
```json
{"state": "running", "detail": "Indexing /app/pdfs"}
```
Poll until `"state": "done"`.

Indexing is incremental — files already indexed (by content hash) are skipped on future `/index` calls, so re-running this after adding more PDFs only processes what's new.

## 5. Ask questions

```bash
curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: $(grep API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Northwind Robotics'\''s Q3 revenue in 2023?"}'
```

```json
{
  "answer": "According to company_01.pdf (page 2), Northwind Robotics reported Q3 2023 revenue of $69.1 million...",
  "sources": [{"file": "company_01.pdf", "page": 2}]
}
```

Check `test_pdfs/_answer_key.json` to confirm the answer matches the real embedded fact — that's how you verify retrieval is actually working, not just producing plausible-sounding text.

## 6. Stop / restart

```bash
docker compose down        # stops the container, keeps your data (./pdfs, ./chroma_db)
docker compose up -d       # starts it again, no re-indexing needed
docker compose up -d --build   # rebuild after changing app.py or rag_pipeline.py
```

Your indexed data lives in `./chroma_db` on your machine (created automatically), so it survives container restarts and rebuilds.

## Interactive docs

FastAPI auto-generates a testable API browser at `http://localhost:8000/docs` — useful if you'd rather click through requests than type curl commands.

## Tuning

Open `rag_pipeline.py`, top of file:

| Setting | Effect |
|---|---|
| `CHUNK_SIZE_WORDS` | Bigger = more context per chunk, less precise retrieval. Default 500. |
| `TOP_K` | How many chunks get sent to the LLM per question. Default 5. |
| `EMBEDDING_MODEL` | Swap for a bigger model if retrieval quality feels off. |
| `LLM_PROVIDER` / `LLM_KEY` / `LLM_MODEL` (in `.env`, not `rag_pipeline.py`) | Which LLM answers questions, and with which model. See `env.example`. |

Changes require `docker compose up -d --build` to take effect.

## Troubleshooting

- **`401 Unauthorized`** — missing or wrong `X-API-Key` header; check it matches `.env`.
- **`404` on `/ask`** — nothing indexed yet, or indexing hasn't finished (check `/index/status`).
- **PDF missing from answers** — check container logs (`docker compose logs -f`) for a `[SKIP]` warning at that filename; it likely looks like a scanned image with no extractable text and needs OCR (not yet wired in — say the word if you hit this and want it added).
- **Build fails with "no space left on device"** — run `docker system prune -a --volumes` first; the CPU-only torch build in the Dockerfile should already keep the image small (~1–1.5GB) but leftover build cache from earlier attempts can still fill disk.
