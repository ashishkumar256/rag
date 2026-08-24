# RAG Pipeline - Setup & Run Guide

## Files in this project

| File | Purpose |
|---|---|
| `rag_pipeline.py` | The full pipeline: parse, chunk, embed, store, retrieve, answer |
| `generate_test_pdfs.py` | Creates 20 synthetic test PDFs with known facts, for testing |
| `test_pdfs/` | 20 generated PDFs + `_answer_key.json` (ground-truth facts) |

## 1. Install dependencies

Requires Python 3.9+.

```bash
pip install pypdf sentence-transformers chromadb anthropic
```

First run will download the embedding model (~130MB) automatically -- that's normal, only happens once.

## 2. Get an Anthropic API key

Only needed for the *answering* step (retrieval works without it).

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Get a key at https://console.anthropic.com if you don't have one.

## 3. Test with the 20 synthetic PDFs (recommended before your real 2000)

The test PDFs are already generated in `test_pdfs/`. If you want to regenerate them:

```bash
python generate_test_pdfs.py
```

Then index them:

```bash
python rag_pipeline.py index test_pdfs
```

You'll see output like:
```
Indexing company_01.pdf...
Indexing company_02.pdf...
...
Done. 187 new chunks indexed. 0 files unchanged and skipped.
```

This creates a `chroma_db/` folder (your vector store) and `indexed_files.json` (tracks what's been indexed, for incremental runs).

Then ask a question. Open `test_pdfs/_answer_key.json` to see the real facts and pick one to test:

```bash
python rag_pipeline.py ask "What was Northwind Robotics's Q3 revenue in 2023?"
```

Expected: an answer citing `company_01.pdf, page 2` with the correct dollar figure. Compare against `_answer_key.json` to confirm it's right, not just plausible-sounding.

Try a few more:
```bash
python rag_pipeline.py ask "Who is the CEO of Bluepeak Logistics?"
python rag_pipeline.py ask "Which company is headquartered in Nairobi?"
python rag_pipeline.py ask "What is the revenue of a company that doesn't exist, like Acme Corp?"
```

That last one tests whether the model correctly says "not found" instead of hallucinating -- important to verify before trusting it on your real documents.

## 4. Run on your real 2000+ PDFs

```bash
python rag_pipeline.py index /path/to/your/pdf/folder
```

This will take a while the first time (expect anywhere from tens of minutes to a few hours depending on total page count and your CPU -- there's no API rate limit since embeddings run locally, so it's just raw compute time).

**Important:** the script skips PDFs whose content looks like a scanned image (very little extractable text) and prints a `[SKIP]` warning with the filename. Collect that list -- those files need OCR before they can be indexed. See the "Handling scanned PDFs" section below.

Because of the hash-based cache (`indexed_files.json`), re-running `index` on the same folder later only processes new or changed files:

```bash
# add new PDFs to the folder, then just re-run -- unchanged files are skipped
python rag_pipeline.py index /path/to/your/pdf/folder
```

Then query as usual:

```bash
python rag_pipeline.py ask "your real question here"
```

## 5. Handling scanned PDFs (if the [SKIP] warnings show up)

```bash
pip install pytesseract pdf2image
# also requires the tesseract binary installed at the OS level:
#   macOS:   brew install tesseract poppler
#   Ubuntu:  sudo apt install tesseract-ocr poppler-utils
```

This isn't wired into `rag_pipeline.py` yet by default since OCR is much slower and you likely don't want it running on all 2000 files automatically. Once you know which files are scanned (from the `[SKIP]` list), that's the next step to add -- happy to build that in when you get there.

## 6. Tuning knobs (in `rag_pipeline.py`, top of file)

| Setting | Effect |
|---|---|
| `CHUNK_SIZE_WORDS` | Bigger = more context per chunk but less precise retrieval. Start at 500, adjust based on your document type. |
| `TOP_K` | How many chunks get sent to the LLM per question. More = harder to miss the answer, but more noise. Start at 5. |
| `EMBEDDING_MODEL` | Swap for a bigger model (e.g. `BAAI/bge-base-en-v1.5`) if retrieval quality feels off -- slower but more accurate. |

## Troubleshooting

- **"No indexed content found"** -- you ran `ask` before `index`, or indexed a different folder.
- **Answers seem to ignore your PDFs** -- check the `[SKIP]` warnings from indexing; the file you're asking about may not have been indexed (likely scanned).
- **Slow indexing** -- normal on CPU for large batches. If you have a GPU, sentence-transformers will use it automatically if `torch` with CUDA is installed.# rag
