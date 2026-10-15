![CineSearch — retro movie-search cover](readme-assets/cinesearch-cover.webp)

# CineSearch — Movie RAG Engine

**Find movies through keyword and semantic search, then generate answers grounded in retrieved descriptions.**

CineSearch is a Retrieval-Augmented Generation (RAG) engine. It starts with normalized keyword matching and grows into a multi-stage search system that combines BM25, dense sentence embeddings, chunk retrieval, reciprocal rank fusion, reranking, LLM query enhancement, evaluation metrics, and grounded generation.

Browse movies in a keyboard-driven terminal interface, or use the individual Python commands to inspect each retrieval and generation technique.

[Try movie search](#-quick-start)

## <img src="readme-assets/icons/features.svg" width="24" height="24" alt=""> Features

- Browse movies with arrow keys, full descriptions, stable controls, and saved display preferences.
- Choose cover-inspired, light, high-contrast, or terminal-native colors; use plain text for linear reading.
- Retrieve titles with normalized keyword matching, TF-IDF, and BM25.
- Search descriptions and overlapping sentence chunks using semantic embeddings.
- Combine rankings with weighted scoring or RRF, with optional query enhancement and reranking.
- Evaluate retrieval using Precision@K, Recall@K, and F1 against a golden dataset.
- Generate answers, summaries, and citation-aware responses from retrieved context.

## <img src="readme-assets/icons/installation.svg" width="24" height="24" alt=""> Installation

Clone with `git clone https://github.com/yaqyn/cinesearch-rag-engine.git`, then run the commands below from `cinesearch-rag-engine/`.

### <img src="readme-assets/icons/requirements.svg" width="20" height="20" alt=""> Requirements

- Bash and curl for `./launch` setup on Linux or macOS
- Internet access on first launch to install dependencies and fetch the catalog
- Python 3.13+ and [uv](https://docs.astral.sh/uv/) for manual setup; the launcher provisions these automatically
- The Boot.dev movie dataset at `data/movies.json`; the launcher downloads it when missing
- A CPU; embedding models and PyTorch are configured for CPU execution (no CUDA required)
- An OpenRouter API key only for LLM-powered commands

For automatic setup, run `./launch`. For manual setup:

```bash
uv sync
```

For manual setup, place the dataset at `data/movies.json` with this shape:

```json
{
  "movies": [
    {"id": 1, "title": "Example film", "description": "A short movie description."}
  ]
}
```

For generation, query enhancement, LLM reranking, and LLM evaluation, create an ignored `.env` file:

```dotenv
OPENROUTER_API_KEY="your_api_key_here"
```

The default free model can be overridden without editing code:

```bash
export OPENROUTER_MODEL="your-provider/your-model"
```

The first semantic command downloads the embedding model. Later runs reuse the local model and cached vectors.

### <img src="readme-assets/icons/configuration.svg" width="20" height="20" alt=""> Build the local indexes

```bash
uv run cli/keyword_search_cli.py build
uv run cli/semantic_search_cli.py embed_chunks
```

If the movie dataset changes, rebuild the keyword index and remove stale embedding files before regenerating them:

```bash
rm -f cache/movie_embeddings.npy cache/chunk_embeddings.npy cache/chunk_metadata.json
uv run cli/keyword_search_cli.py build
uv run cli/semantic_search_cli.py embed_chunks
```

## <img src="readme-assets/icons/launch.svg" width="24" height="24" alt=""> Quick start

Clone and launch:

```bash
git clone https://github.com/yaqyn/cinesearch-rag-engine.git
cd cinesearch-rag-engine
./launch
```

`./launch` installs uv into `~/.local/bin` if missing (without changing your shell
configuration), provisions the project's Python, installs locked dependencies,
downloads the [Boot.dev movie catalog](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/movies.json?generation=1759751668026541)
when missing, builds missing keyword indexes, and opens the browser. First setup
needs internet access and may take several minutes because CPU PyTorch is included.
Later launches reuse installed dependencies, your dataset, and cached indexes.
No API key is needed for local search. Existing `data/movies.json` files are preserved.

The launcher forwards arguments to CineSearch:

```bash
./launch browse --plain
./launch browse --theme contrast
./launch health
./launch search "family movie about bears" --mode keyword --limit 5
```

After setup, `uv run cinesearch` also opens the browser directly. Search rejects
blank queries and limits outside 1–100 before loading data or models.

The browser uses the cover's plum, parchment, and amber palette. Choose among
cover, light, high-contrast, and terminal-native themes with **F2**, or start with
`uv run cinesearch browse --theme contrast`. `NO_COLOR` starts the browser using
your terminal's own colors. Theme and spacing preferences are saved in
`$XDG_CONFIG_HOME/cinesearch/terminal.json` (default: `~/.config/cinesearch/terminal.json`).

| Control | Action |
| --- | --- |
| Enter in the search field / Ctrl+S anywhere | Search |
| Tab / Shift+Tab | Move between search, mode, button, results, and description |
| Up / Down | Select a mode or movie; descriptions follow the highlighted movie |
| Enter / Right in results | Focus the description |
| Left in the description | Return to results |
| Home / End / PageUp / PageDown in results | Jump through movies |
| Down in the search field | Return to existing results |
| PageUp / PageDown in the description | Read longer descriptions |
| Esc / Ctrl+F | Return to search and close help |
| F1 | Show or hide keyboard help |
| F2 / F3 | Cycle theme / toggle comfortable or compact spacing |
| F10 / Ctrl+Q / Ctrl+C | Exit |

Controls stay anchored while browsing descriptions. Results and descriptions
stack in narrow terminals. Full descriptions wrap rather than being truncated. Font family and text size follow your terminal settings;
use its zoom shortcut (often Ctrl+Plus / Ctrl+Minus) for larger text. No mouse,
color recognition, or animation is required. Very small windows may need terminal
zoom-out or `--plain`. Screen-reader compatibility depends on the terminal;
`--plain` provides a linear keyword-search workflow without full-screen redraws.
Keyword mode runs locally; semantic and hybrid modes may download models on first
use. Existing individual technique CLIs remain available below.

`health` reports the absolute cache location and available derived artifacts. Embedding caches now carry a dataset/model manifest, so changing the corpus or model automatically triggers a safe rebuild instead of silently using stale vectors.

## <img src="readme-assets/icons/usage.svg" width="24" height="24" alt=""> Usage

### <img src="readme-assets/icons/usage.svg" width="20" height="20" alt=""> Keyword and BM25

```bash
uv run cli/keyword_search_cli.py search "british bear"
uv run cli/keyword_search_cli.py bm25search "family movie about bears" --limit 5
uv run cli/keyword_search_cli.py tf 4651 merida
uv run cli/keyword_search_cli.py bm25tf 4651 merida
uv run cli/keyword_search_cli.py bm25idf merida
```

Keyword search preprocesses text consistently: lowercase conversion, punctuation removal, whitespace tokenization, stop-word filtering, and Porter stemming.

### <img src="readme-assets/icons/usage.svg" width="20" height="20" alt=""> Semantic search

```bash
uv run cli/semantic_search_cli.py search "a dangerous journey through the wilderness"
uv run cli/semantic_search_cli.py search_chunked "a dangerous journey through the wilderness" --limit 5
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py semantic_chunk "First sentence. Second sentence! Third sentence?" --max-chunk-size 2 --overlap 1
```

### <img src="readme-assets/icons/usage.svg" width="20" height="20" alt=""> Hybrid search

Weighted search mixes normalized BM25 and semantic scores:

```bash
uv run cli/hybrid_search_cli.py weighted-search "bear adventure" --alpha 0.5 --limit 5
```

RRF combines rankings without requiring lexical and semantic scores to share a scale:

```bash
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --limit 5
```

Optional query enhancement and reranking:

```bash
uv run cli/hybrid_search_cli.py rrf-search "britsh bear movie" --enhance spell
uv run cli/hybrid_search_cli.py rrf-search "that bear movie with marmalade" --enhance rewrite
uv run cli/hybrid_search_cli.py rrf-search "scary bear movie" --enhance expand
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --rerank-method batch --limit 5
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --rerank-method cross_encoder --limit 5
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --evaluate
```

`individual`, `batch`, `spell`, `rewrite`, `expand`, and `evaluate` use OpenRouter and may be affected by model availability, rate limits, or free-tier quotas. `cross_encoder` runs locally but downloads its model on first use.

### <img src="readme-assets/icons/usage.svg" width="20" height="20" alt=""> Grounded generation

The generation CLI retrieves movie context first, then asks an LLM to use that context:

```bash
uv run cli/augmented_generation_cli.py rag "movies about action and dinosaurs"
uv run cli/augmented_generation_cli.py summarize "movies about action and dinosaurs" --limit 5
uv run cli/augmented_generation_cli.py question "Who are the main characters in Jurassic Park?"
uv run cli/augmented_generation_cli.py citations "action movie with lasers"
```

Each command prints retrieved titles before the generated response, making the grounding context visible instead of treating the model response as an unexplained black box.

## <img src="readme-assets/icons/architecture.svg" width="24" height="24" alt=""> Architecture

Retrieval combines keyword indexes and semantic vectors before optional reranking or generation.

![Search workflow](readme-assets/workflow.svg)

```text
data/movies.json
        │
        ├── InvertedIndex ── tokenization, stop words, stemming, BM25
        │
        ├── SemanticSearch ── movie embeddings, cosine similarity
        │
        ├── ChunkedSemanticSearch ── sentence chunks, chunk embeddings
        │
        └── HybridSearch ── weighted fusion or reciprocal rank fusion
                              │
                              ├── optional reranking
                              └── optional LLM generation
```

Keyword indexes and embedding artifacts are persisted under `cache/`. They are generated locally and ignored by Git because they are large, derived files.

## <img src="readme-assets/icons/structure.svg" width="24" height="24" alt=""> Project structure

| Path | Responsibility |
| --- | --- |
| `launch` | One-command dependency, dataset, and keyword-index setup, then launch |
| `cli/bootstrap.py` | Atomic dataset download, validation, and first-run index preparation |
| `cli/terminal.py` | Keyboard browser, themes, preferences, and plain-text mode |
| `cli/inverted_index.py` | Tokenization, persistent inverted index, TF-IDF, and BM25 |
| `cli/lib/semantic_search.py` | Dense embeddings, sentence chunking, semantic retrieval, and chunk retrieval |
| `cli/lib/hybrid_search.py` | Weighted hybrid search and Reciprocal Rank Fusion |
| `cli/keyword_search_cli.py` | Keyword and BM25 command-line interface |
| `cli/semantic_search_cli.py` | Embedding, chunking, and semantic-search interface |
| `cli/hybrid_search_cli.py` | Hybrid search, enhancement, reranking, and evaluation interface |
| `cli/augmented_generation_cli.py` | RAG answers, summaries, questions, and citations |
| `cli/evaluation_cli.py` | Golden-dataset Precision, Recall, and F1 evaluation |
| `cli/cinesearch.py` | Unified product entrypoint and local artifact health check |
| `cli/lib/config.py` | Repository-root paths, cache paths, and embedding signatures |
| `tests/` | Ranking, input validation, keyboard navigation, layout, preferences, and setup regressions |
| `data/` | Local movie corpus and evaluation data |
| `cache/` | Generated indexes, metadata, and embedding arrays |

Run the local regression suite with `uv run python -m unittest discover -s tests -v`.

## <img src="readme-assets/icons/evaluation.svg" width="24" height="24" alt=""> Evaluation

The golden dataset contains representative natural-language queries and relevant movie titles. The evaluator reports retrieval quality at a configurable K:

```bash
uv run cli/evaluation_cli.py
uv run cli/evaluation_cli.py --limit 10
```

- **Precision@K**: relevant retrieved titles divided by K.
- **Recall@K**: relevant retrieved titles divided by all relevant titles in the golden dataset.
- **F1**: the harmonic mean of precision and recall.

## <img src="readme-assets/icons/security.svg" width="24" height="24" alt=""> Data and credentials

The movie corpus, generated indexes and embeddings, virtual environment, and credential files are stored locally. LLM-powered commands send prompts and selected retrieved context to OpenRouter; local keyword and embedding search need no API key. The small stop-word list and golden evaluation dataset are included in the repository.

`cli/provider_probe.py` is an optional live-provider diagnostic, not an automated test. It only sends a request when run directly and requires your own `OPENROUTER_API_KEY`. Importing it does not load credentials or call the provider.

## <img src="readme-assets/icons/learning.svg" width="24" height="24" alt=""> Development notes

The implementation uses separate command-line tools for retrieval, generation, and evaluation. Generated data and credentials remain local.

Run command help at any time:

```bash
uv run cli/keyword_search_cli.py --help
uv run cli/semantic_search_cli.py --help
uv run cli/hybrid_search_cli.py --help
uv run cli/augmented_generation_cli.py --help
```

Built by **[Abdulrahman M. Yaqyn](https://yaqyn.github.io/folio-static-site/)**.

> **Learning Journey Project · Boot.dev**
> Developed through the Boot.dev curriculum and extended through independent work.
