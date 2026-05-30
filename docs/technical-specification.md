# SettleBot — Technical Specification Document

**Document reference:** SB-TECH-2026-001

**Version:** 1.0

**Date:** May 30, 2026

**Prepared for:** SettleBot Project

**Status:** Active

---

# 1. Introduction

## 1.1 Purpose

This document defines the technical implementation specification for SettleBot — a RAG-based settlement assistant for international students. It translates the system architecture into concrete technology choices, coding standards, project structure, AI system design rules, API conventions, testing strategy, and operational procedures. It is the authoritative reference for all development activity on this project, followed by both human developers and AI coding agents.

## 1.2 Scope

This document covers the RAG API backend, the Flask web frontend, the vector database implementation, the LLM integration layer, the document ingestion pipeline, the evaluation system, authentication, and the testing strategy. It does not introduce features beyond those defined in the project requirements.

## 1.3 Intended audience

All developers and AI agents working on this codebase must follow this specification without deviation. Any proposed deviation requires explicit written approval before implementation begins. Adherence to this document is non-negotiable.

---

# 2. Technology Stack

## 2.1 RAG API Backend

| Concern | Technology |
|---|---|
| Language | Python 3.12 |
| Web framework | FastAPI |
| AI/LLM | OpenAI API (gpt-4.1-mini for responses, gpt-4.1-nano for language/chunking) |
| Embeddings | OpenAI text-embedding-3-small (unified — no other embedding model may be used) |
| Vector database | ChromaDB (persistent local client) |
| Data validation | Pydantic v2 |
| Retry logic | tenacity |
| Structured logging | structlog with JSON renderer |
| NLP support | spaCy (en_core_web_sm), NLTK |
| Web search | Tavily (optional, graceful degradation when absent) |
| Configuration | PyYAML + python-dotenv via `config/settings.py` |
| CLI | Typer with Rich |
| Testing | pytest |

## 2.2 Flask Frontend

| Concern | Technology |
|---|---|
| Language | Python 3.12 |
| Web framework | Flask 3.x |
| Templating | Jinja2 |
| ORM | Flask-SQLAlchemy (sync) |
| Database | SQLite (development), configurable via environment variable |
| Schema migrations | Flask-Migrate (Alembic) |
| Authentication | Flask-Login |
| Password hashing | Werkzeug security (generate_password_hash / check_password_hash) |
| Token generation | itsdangerous URLSafeTimedSerializer |
| Forms | Flask-WTF / WTForms |
| Email | utilities/email_utils.py (send_email — must not be modified) |
| CSS | Custom CSS, no framework |
| Testing | pytest |

## 2.3 Infrastructure

| Concern | Technology |
|---|---|
| Vector storage | ChromaDB persistent client at `database/chroma_db/` |
| File storage | Local filesystem under `data/` and `uploads/` |
| Document ingestion | LangChain community loaders (WebBaseLoader, SitemapLoader) |
| Process management | Systemd or equivalent |
| Reverse proxy | Nginx |

---

# 3. RAG Backend Specification

## 3.1 Project structure

```
backend (project root)/
  api/
    main.py               FastAPI app. Registers routers and middleware only.
  services/
    embeddings.py         Embedding generation and caching.
    vector_db.py          ChromaDB interactions.
    intent_recognizer.py  Intent classification via cosine similarity.
    language_processor.py Language detection and translation.
    response_generator.py LLM response assembly and generation.
    document_processor.py Document ingestion and chunk creation.
    semantic_chunking.py  Chunk strategy implementations.
    evaluator.py          RAG evaluation runner.
  config/
    config.yaml           Non-secret configuration. No credentials here.
    settings.py           Pydantic settings loaded from config.yaml + env vars.
    locale/               One JSON file per deployment locale.
  utilities/
    path.py               Path resolution helpers.
  tests/
    test_basic.py
    conftest.py
  cli.py                  Typer CLI for index management and interactive queries.
  app.py                  Uvicorn entry point.
  requirements.txt
```

## 3.2 Coding standards

These standards apply to every line of code written in this project. They are not optional. AI coding agents and human developers must follow all of them without exception.

### 3.2.1 No logic in endpoint functions

Endpoint functions must contain no business logic. Their sole responsibilities are receiving a validated request, delegating to a service function, and returning the result. Conditional statements, database queries, LLM calls, loops, and calculations must not appear inside endpoint functions.

**Correct:**

```python
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key),
) -> QueryResponse:
    """
    Receives a settlement query and delegates processing to query_service.
    :param request: QueryRequest - The validated query payload.
    :param api_key: str - The authenticated API key.
    :return: QueryResponse - The generated response with intent and context metadata.
    """
    return await query_service.process(request)
```

**Not permitted:**

```python
@app.post("/query")
async def process_query(request: QueryRequest):
    language_result = language_processor.detect_and_process_query(request.query)
    intent_info = intent_recognizer.get_intent_info(language_result["english_query"])
    # Any logic here violates this standard
```

### 3.2.2 No print() statements in production code

`print()` must not be used anywhere in `api/`, `services/`, `config/`, or `utilities/`. All output must go through structured logging. This includes timing measurements.

**Not permitted:**

```python
print(f"Elapsed time is {end_time - start_time} seconds")
print(query_response)
print("processed")
```

**Correct:**

```python
logger.info("llm_call_completed", elapsed_seconds=round(end_time - start_time, 3))
logger.debug("query_processed", intent=intent_type, confidence=confidence)
```

### 3.2.3 Import ordering

Each import must appear on its own line. Imports are organised into three groups separated by a blank line: standard library, third-party packages, local imports. Within each group, lines are ordered from shortest to longest by total character count. If two lines are the same length, order alphabetically.

**Correct:**

```python
import os
import time
import logging

import numpy as np
from openai import OpenAI
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from config.settings import settings
from services.embeddings import EmbeddingService
```

**Not permitted:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings, ROOT_DIR
import os, time
```

### 3.2.4 Docstrings on every function, method, and class

Every function, method, and class must have a docstring. Functions with parameters or return values must document each parameter and the return value using `:param name: type - description` and `:return: type - description` format.

```python
def embed_query(self, query: str) -> Optional[np.ndarray]:
    """
    Generates a normalised embedding vector for a single query string.
    Prepends settlement context prefix if no locale indicator is present.
    :param query: str - The raw user query to embed.
    :return: Optional[np.ndarray] - The embedding array, or None if generation fails.
    """
```

### 3.2.5 No magic numbers or strings

Numeric literals and string constants that carry meaning must be defined as named constants. Inline literals for thresholds, limits, model names, and timeouts are not permitted in service code.

**Correct (define in `config/constants.py`):**

```python
INTENT_OFF_TOPIC_THRESHOLD = 0.40
INTENT_EMBEDDING_CACHE_SIZE = 512
TAVILY_MAX_RESULTS = 3
TAVILY_TIMEOUT_SECONDS = 4
LLM_RESPONSE_MAX_TOKENS = 2048
LLM_RESPONSE_TEMPERATURE = 0.2
QUERY_MAX_LENGTH = 2000
QUERY_MIN_LENGTH = 3
EMBEDDING_CONTEXT_PREFIX = "International student in {city} {country}: "
```

**Not permitted:**

```python
if max_similarity < 0.40:
    return IntentType.OFF_TOPIC
response = self.client.chat.completions.create(model="gpt-4.1-mini", max_tokens=4096)
```

### 3.2.6 No silent exception swallowing

Every exception must be handled explicitly. Bare `except` clauses are not permitted. Every caught exception must be either re-raised, transformed into a structured HTTP error, or logged with full context before suppression.

**Not permitted:**

```python
try:
    result = self._detect_and_translate_llm(query)
except Exception:
    pass
```

**Correct:**

```python
try:
    result = self._detect_and_translate_llm(query)
except json.JSONDecodeError as exc:
    logger.error("language_detection_json_parse_failed", query_preview=query[:50], error=str(exc))
    raise
except openai.OpenAIError as exc:
    logger.error("language_detection_llm_failed", error=str(exc))
    return self._fallback_language_result(query)
```

### 3.2.7 No hardcoded configuration values

Database paths, API keys, model names, storage paths, locale names, and all environment-specific values must be read from the `settings` object in `config/settings.py`. Hardcoded values for these in any service file are not permitted.

**Not permitted:**

```python
self.client = OpenAI(api_key="sk-...")
self.db_path = ROOT_DIR / "database" / "chroma_db"
response = self.client.chat.completions.create(model="gpt-4.1-mini", ...)
```

**Correct:**

```python
self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
self.db_path = ROOT_DIR / settings.vector_db.location
response = self.client.chat.completions.create(model=settings.llm.model, ...)
```

### 3.2.8 Async consistency

FastAPI route handlers are async. Blocking I/O operations must not be called directly inside async functions. ChromaDB's Python client is synchronous — calls to it must be wrapped with `asyncio.to_thread()` in async request handlers, or isolated to synchronous service methods called from thread-pool executors.

**Not permitted inside an async endpoint:**

```python
results = self.collection.query(query_embeddings=[embedding.tolist()], n_results=30)
```

**Correct:**

```python
results = await asyncio.to_thread(
    self.collection.query,
    query_embeddings=[embedding.tolist()],
    n_results=30,
)
```

### 3.2.9 Type annotations on every function signature

Every function and method must have complete type annotations on all parameters and the return type. Functions returning nothing must be annotated `-> None`. `Any` must not be used unless unavoidable, and its use must be accompanied by a comment.

**Not permitted:**

```python
def classify_intent(self, query, patterns):
    pass
```

**Correct:**

```python
def classify_intent(self, query: str, patterns: Dict[IntentType, np.ndarray]) -> IntentResult:
    """..."""
```

### 3.2.10 No hardcoded locale values

No service file may contain hardcoded references to `"nairobi"`, `"kenya"`, `"KSh"`, `"matatu"`, `"M-Pesa"`, `"Westlands"`, or any other city-specific, country-specific, or currency-specific term as string literals in logic. All locale-specific values must come from a `LocaleConfig` object passed through service constructors.

**Not permitted:**

```python
if "nairobi" not in query_lower and "kenya" not in query_lower:
    alternatives.append(f"{original_query} in Nairobi Kenya")
return f"International student in Nairobi Kenya: {query}"
```

**Correct:**

```python
if self.locale.city not in query_lower and self.locale.country not in query_lower:
    alternatives.append(f"{original_query} in {self.locale.city} {self.locale.country}")
return f"International student in {self.locale.city} {self.locale.country}: {query}"
```

### 3.2.11 All LLM calls must use tenacity retry with explicit timeout

Every call to the OpenAI API must be wrapped with tenacity retry logic and an explicit timeout. Unwrapped bare `client.chat.completions.create()` or `client.embeddings.create()` calls are not permitted in service code.

**Correct:**

```python
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from tenacity import retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
)
def _call_chat_completion(self, messages: List[Dict], max_tokens: int) -> str:
    """
    Calls the OpenAI chat completion API with retry logic on transient errors.
    :param messages: List[Dict] - The message list to send.
    :param max_tokens: int - Maximum tokens in the response.
    :return: str - The assistant's response content.
    """
    response = self.client.chat.completions.create(
        model=settings.llm.model,
        messages=messages,
        temperature=settings.llm.temperature,
        max_tokens=max_tokens,
        timeout=15,
    )
    return response.choices[0].message.content
```

### 3.2.12 Grounding rule — no facts without source

The LLM system prompt for response generation must include the grounding rule as its first instruction block, before any intent-specific or locale-specific content. This rule must not be removed or modified without explicit written approval.

Required grounding rule text (included in every system prompt):

```
GROUNDING RULE (NON-NEGOTIABLE):
You may only state a phone number, email address, physical address,
fee amount, or operating hours if that value appears VERBATIM in the
RETRIEVED SETTLEMENT INFORMATION or ESSENTIAL SETTLEMENT INFORMATION
sections provided below.
If you cannot find a contact detail in the provided context, write:
"Contact details not available — verify at [official source URL if known]."
Never invent, approximate, or infer contact information.
```

### 3.2.13 Embedding model unification

One embedding model must be used throughout the entire system. The model name is read from `settings.embedding.model`. No service file may hardcode a model name for embeddings. The intent recognizer, embedding service, and vector database must all use the same model. The intent prototype embedding cache must be invalidated and rebuilt whenever `settings.embedding.model` changes.

**Not permitted:**

```python
response = self.openai_client.embeddings.create(
    input=examples, model="text-embedding-ada-002"  # hardcoded
)
```

**Correct:**

```python
response = self.openai_client.embeddings.create(
    input=examples, model=settings.embedding.model
)
```

### 3.2.14 No per-chunk LLM calls during document ingestion

`SemanticChunker._enrich_chunks_with_llm_analysis()` must not call the LLM API per chunk. Chunk-level quality scores must be computed using the deterministic keyword-based `_calculate_settlement_relevance_score()` method from `DocumentProcessor`. A single document-level LLM analysis call (`_analyze_text_with_llm`) per document is permitted.

The cost difference is significant: a 20-chunk document must incur at most 1 LLM call for metadata enrichment, not 21.

### 3.2.15 Tavily web content must be sanitised before LLM injection

Web search results from Tavily must be sanitised to remove potential prompt injection patterns before being inserted into the LLM context window. Use the `sanitise_web_content()` utility function from `utilities/sanitisation.py`. Raw `result["content"]` must never be concatenated directly into user or system prompts.

```python
from utilities.sanitisation import sanitise_web_content

safe_content = sanitise_web_content(result["content"])
context_parts.append(safe_content)
```

### 3.2.16 Error response structure

All error responses from the FastAPI backend must use a consistent structure:

```json
{
  "success": false,
  "error": {
    "code": "INTENT_RECOGNITION_FAILED",
    "message": "Could not classify the query intent. Please rephrase your question."
  }
}
```

The `code` field must use a constant defined in `config/constants.py`. The `message` field must be plain language suitable for display to the user.

---

## 3.3 LocaleConfig specification

All locale-specific values flow through a `LocaleConfig` object. No service may access locale data except through this object. The config is loaded at startup from a JSON file in `config/locale/`.

```python
@dataclass
class LocaleConfig:
    """
    Encapsulates all locale-specific configuration for a SettleBot deployment.
    Passed through service constructors. Never hardcode values that belong here.
    """
    city: str
    country: str
    currency_code: str
    currency_symbol: str
    timezone: str
    emergency_number: str
    primary_languages: List[str]
    key_neighborhoods: List[str]
    trusted_web_domains: List[str]
    collection_name: str
    web_search_geo_bias: str
    fact_store_path: str
```

A sample locale file for Nairobi lives at `config/locale/nairobi.json`. Adding support for a new city requires only creating a new locale JSON file and updating the `SETTLEBOT_LOCALE` environment variable. No Python code changes are required.

---

## 3.4 Vector database conventions

### 3.4.1 Collection namespacing

Each locale has its own ChromaDB collection named `settlebot_{city}`. This prevents cross-contamination between deployments. The collection name is always read from `locale.collection_name`, never hardcoded.

### 3.4.2 Metadata fields per chunk

Every indexed chunk must carry these metadata fields:

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | Parent document identifier |
| `chunk_id` | str | Unique chunk identifier within the document |
| `chunk_index` | int | Position of this chunk within the document |
| `settlement_score` | float | Relevance score [0.0–1.0] to settlement domain |
| `topic_tags` | str (JSON) | List of settlement topics detected in this chunk |
| `location_entities` | str (JSON) | Location names extracted from this chunk |
| `source_url` | str | Original URL or file path |

### 3.4.3 Query embedding must match index embedding

The model used to embed documents at ingestion time must be identical to the model used to embed queries at search time. They must both use `settings.embedding.model`. Changing the embedding model requires rebuilding the entire index.

### 3.4.4 Retrieval pipeline

Retrieval must follow this sequence:
1. Embed query using `EmbeddingService.embed_query()`
2. Query ChromaDB for `top_k * 2` candidates
3. Apply settlement score boost
4. Sort by boosted score
5. Return top `top_k` results

A BM25 sparse retrieval component with Reciprocal Rank Fusion is the target upgrade (see `FEEDBACK.md` Phase 6).

---

## 3.5 Intent recognition conventions

### 3.5.1 Intent prototype embeddings

Intent prototype embeddings are cached to `.embeddings_cache/`. The cache is valid only if all `IntentType` values (excluding `OFF_TOPIC`) match the cached metadata. If the intent list changes, `rebuild_cache()` must be called.

### 3.5.2 Off-topic threshold

The off-topic threshold is defined in `config/constants.py` as `INTENT_OFF_TOPIC_THRESHOLD`. It must not be hardcoded in `intent_recognizer.py`. The default value is `0.40`.

### 3.5.3 Evaluator method name

The evaluator calls intent classification through `intent_recognizer.get_intent_info(query)`. The method `recognize_intent()` does not exist and must not be called. Any test or script that calls `recognize_intent()` is a bug and must be fixed immediately.

### 3.5.4 IntentType and evaluator alignment

All `expected_intent` values in the evaluator test set (`evaluator.py`) must match values in the `IntentType` enum exactly. Intent types that exist in the test set but not in the enum are bugs. The enum is the source of truth.

---

## 3.6 API design conventions

### 3.6.1 URL structure

All endpoints are prefixed with `/api/v1/` for versioned endpoints. The existing unversioned endpoints (`/query`, `/health`, `/documents/*`, etc.) are grandfathered until a versioning migration is completed.

```
POST   /query
GET    /health
GET    /system/status
POST   /intent/analyze
POST   /search
GET    /search/topics
GET    /search/locations
POST   /documents/upload
GET    /documents
GET    /documents/{doc_id}
DELETE /documents/{doc_id}
POST   /documents/process-url
POST   /documents/process-sitemap
POST   /chunking/process-text
GET    /chunking/strategies
GET    /embeddings/stats
POST   /embeddings/generate
GET    /vector-db/stats
POST   /vector-db/rebuild-index
POST   /language/detect
POST   /language/translate
GET    /language/supported
POST   /evaluation/run
GET    /evaluation/status/{task_id}
```

### 3.6.2 Request tracing

Every request must carry a correlation ID. If the incoming request has an `X-Request-ID` header, use it. Otherwise generate a UUID. The ID must appear in all log entries for that request and in the response headers.

```python
@app.middleware("http")
async def add_correlation_id(request: Request, call_next: Callable) -> Response:
    """
    Injects a correlation ID into the request context and response headers.
    :param request: Request - The incoming HTTP request.
    :param call_next: Callable - The next middleware or route handler.
    :return: Response - The response with X-Request-ID header attached.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

### 3.6.3 Rate limiting

All endpoints must be protected by rate limiting using `slowapi`. The `/query` endpoint limit is `30/minute` per IP address. The `/documents/upload` endpoint limit is `10/minute` per IP address.

### 3.6.4 Input validation

`QueryRequest.query` must enforce `min_length=3` and `max_length=2000`. No query longer than 2000 characters may reach service logic. File upload endpoints must validate the filename to prevent path traversal: `safe_name = Path(file.filename).name`.

---

## 3.7 Logging conventions

All logging uses `structlog` with a JSON renderer. No `logging.basicConfig()` calls may configure the root logger. The application configures structlog once at startup in `app.py`.

```python
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
```

Log levels:
- `DEBUG` — per-step timing, retrieved chunk details, token counts
- `INFO` — request received, intent classified, response generated
- `WARNING` — fallback triggered, Tavily unavailable, cache miss
- `ERROR` — LLM call failed, ChromaDB unavailable, exception caught

---

# 4. Flask Frontend Specification

## 4.1 Project structure

```
front-end/
  app/
    __init__.py           App factory. Registers blueprints and extensions.
    models/               SQLAlchemy models: user.py, conversation.py, message.py.
    accounts/             Profile management blueprint.
    administration/       Admin management blueprint.
    anonymous/            Public-facing chat blueprint.
    authentication/       Login, logout, password reset blueprint.
    main/                 Dashboard and home blueprint.
    static/               CSS, JS, images.
    templates/            Jinja2 templates, one subfolder per blueprint.
  migrations/             Alembic migration files.
  utilities/
    email_utils.py        Email utility. Must not be modified.
    securities.py         Gravatar and security helpers.
  config.py               Flask configuration classes.
  flasky.py               Entry point.
```

## 4.2 Coding standards

### 4.2.1 No logic in view functions

View functions must contain no business logic. They receive the request, call a model method or service, and render a template or return a redirect. Conditional statements, database queries beyond model method calls, and calculations must not appear directly in view functions.

### 4.2.2 Model method conventions

All database interactions are encapsulated as class methods or instance methods on ORM models. The pattern is: `User.create(details)`, `user.update(details)`, `user.delete()`. Views never issue SQLAlchemy queries directly.

Methods that change database state return a `tuple[bool, str]` where the first element is a success flag and the second is a human-readable message:

```python
success, message = current_user.update(details)
if success:
    flask.flash(message, "success")
else:
    flask.flash(message, "error")
```

### 4.2.3 Column naming

SQLAlchemy model columns use `camelCase` naming consistent with the existing codebase (`userId`, `fullName`, `emailAddress`, `dateCreated`, `lastLogin`, `isActive`). Do not introduce `snake_case` column names in new model columns.

### 4.2.4 No hardcoded secret keys

The Flask `SECRET_KEY` must be read from environment variables. The fallback literal in `config.py` is acceptable only for local development. In production, `SECRET_KEY` must be set as an environment variable.

### 4.2.5 Email utility usage

The `send_email()` function in `utilities/email_utils.py` is the only permitted email mechanism. Its signature must not be modified. It must be called with keyword arguments for all parameters.

```python
from utilities.email_utils import send_email

send_email(
    to=[user.emailAddress],
    subject="Welcome to SettleBot",
    template="email/welcome",
    user=user,
)
```

### 4.2.6 Password handling

Passwords are stored using Werkzeug's `generate_password_hash`. The `password` attribute on `User` is write-only and raises `AttributeError` on read. This pattern must be preserved in any model that stores passwords.

### 4.2.7 CSRF protection

All state-changing forms must include CSRF protection via Flask-WTF. The `@csrf.exempt` decorator must not be applied to POST endpoints that accept user-submitted data without explicit approval.

### 4.2.8 Docstrings

All view functions, model methods, and utility functions must have docstrings following the same format as the backend: `:param name: type - description` and `:return: type - description`.

---

# 5. Testing Specification

## 5.1 Principles

Tests are mandatory. No feature is considered complete until its tests pass. Tests must not depend on external services including the OpenAI API or Tavily. All LLM calls must be mocked. Tests must have a defined completion condition and must not run indefinitely. A maximum timeout of 30 seconds applies per test.

## 5.2 RAG backend testing

Tests use pytest. A shared `conftest.py` provides fixtures for a test ChromaDB client with an in-memory collection, mocked OpenAI clients, and a test FastAPI app with the test client.

### Module 1: Embeddings

| Test | Description |
|---|---|
| test_embed_query_returns_array | Valid query returns a numpy array of correct dimension |
| test_embed_query_empty_returns_none | Empty string returns None without raising |
| test_embed_query_prepends_locale_prefix | Query without locale indicator gets context prefix |
| test_embed_query_no_duplicate_prefix | Query already containing locale indicator is not double-prefixed |
| test_embed_batch_returns_correct_count | Batch of N texts returns N embeddings |
| test_cache_prevents_regeneration | Same chunks file with unchanged hash skips regeneration |
| test_cache_invalidated_on_model_change | Changing the model name triggers regeneration |

### Module 2: Intent Recognition

| Test | Description |
|---|---|
| test_classify_housing_query | "Where can I live near campus?" returns housing_inquiry |
| test_classify_safety_query | "Is Westlands safe at night?" returns safety_concern |
| test_classify_off_topic_below_threshold | Query with max similarity below threshold returns off_topic |
| test_get_intent_info_method_exists | Calling get_intent_info() does not raise AttributeError |
| test_recognize_intent_method_does_not_exist | Calling recognize_intent() raises AttributeError (regression guard) |
| test_cache_loads_on_init | IntentRecognizer loads embeddings from cache on second instantiation |
| test_settlement_relevance_boosted | Query with locale keyword yields higher confidence than identical query without it |

### Module 3: Language Processing

| Test | Description |
|---|---|
| test_detect_english_returns_english | English query detected as english |
| test_detect_non_english_returns_translation | Non-English query returns english_query populated |
| test_disabled_detection_returns_passthrough | detection_enabled=False returns original query unchanged |
| test_fallback_on_llm_failure | OpenAI error triggers fallback result with english passthrough |
| test_translation_quality_validates_term_preservation | Preserved terms score reduces if critical terms absent |

### Module 4: Vector Database

| Test | Description |
|---|---|
| test_search_returns_top_k_results | Search against a seeded collection returns at most top_k items |
| test_search_empty_collection_returns_empty | Search on empty collection returns empty list without raising |
| test_settlement_boost_applied | Chunk with higher settlement_score ranks above lower-scored chunk |
| test_location_filter_returns_matching_chunks | Location filter returns only chunks with that location in metadata |
| test_topic_filter_returns_matching_chunks | Topic filter returns only chunks tagged with that topic |
| test_index_chunks_adds_to_collection | Indexing a valid chunks file increases collection count |
| test_health_check_reports_healthy | health_check returns overall_health=True when collection is accessible |

### Module 5: Response Generation

| Test | Description |
|---|---|
| test_off_topic_returns_canned_response | OFF_TOPIC intent returns the standard off-topic response |
| test_response_has_three_sections | Generated response contains all three required section headers |
| test_grounding_rule_in_system_prompt | System prompt passed to LLM contains the grounding rule verbatim |
| test_emergency_number_from_context_only | Phone numbers in response are only those present in injected context |
| test_crisis_high_adds_emergency_info | crisis_level=high triggers emergency info in response |
| test_translation_called_for_non_english | Non-English original query triggers translate_response call |
| test_duplicate_language_detection_not_called | language_processor.detect is called exactly once per query |

### Module 6: Document Processing

| Test | Description |
|---|---|
| test_process_txt_file | Valid .txt file produces chunks and metadata |
| test_unsupported_extension_raises | .exe file raises ValueError |
| test_settlement_score_is_float_in_range | Settlement score for any chunk is in [0.0, 1.0] |
| test_location_entities_extracted | Chunk containing "Westlands" yields "Westlands" in location_entities |
| test_cost_entities_extracted | Chunk containing "KSh 15,000" yields a match in cost_entities |
| test_document_index_persists | After process_document, doc_id appears in list_documents() |
| test_delete_document_removes_files | delete_document removes chunk file and processed file |

### Module 7: Semantic Chunking

| Test | Description |
|---|---|
| test_settlement_optimized_produces_chunks | Non-empty text produces at least one chunk |
| test_empty_text_returns_empty_list | Empty string returns empty list without raising |
| test_chunk_word_count_within_bounds | No chunk word count exceeds max_chunk_size / 4 |
| test_no_llm_call_per_chunk | _enrich_chunks_with_llm_analysis makes at most 1 LLM call regardless of chunk count |
| test_fallback_chunking_on_llm_failure | OpenAI error triggers fallback chunking with no exception raised |

### Module 8: Evaluator

| Test | Description |
|---|---|
| test_evaluator_runs_without_crash | run_comprehensive_evaluation completes without AttributeError or crash |
| test_get_intent_info_not_recognize_intent | Evaluator calls get_intent_info, not recognize_intent |
| test_intent_types_match_enum | All expected_intent values in eval CSV match IntentType enum values |
| test_bleu_score_range | All BLEU scores are in [0.0, 1.0] |
| test_evaluation_report_has_required_keys | Report contains overall_metrics, priority_metrics, intent_performance |

### Module 9: API Endpoints

| Test | Description |
|---|---|
| test_query_endpoint_requires_api_key | Request without Authorization header returns 401 |
| test_query_endpoint_rejects_long_query | Query longer than 2000 characters returns 422 |
| test_query_endpoint_rejects_empty_query | Query shorter than 3 characters returns 422 |
| test_health_endpoint_no_auth_required | /health returns 200 without authentication |
| test_upload_rejects_path_traversal_filename | Filename with ../ in path is sanitised to basename only |
| test_upload_rejects_forbidden_extension | Upload of .exe file returns 400 |
| test_search_patterns_returns_501 | /analytics/search-patterns returns 501 Not Implemented |

## 5.3 Flask frontend testing

Tests use pytest. All API calls to the RAG backend are mocked. Database tests use a separate SQLite in-memory database created per test function.

### Module 10: Authentication

| Test | Description |
|---|---|
| test_login_valid_credentials | Valid credentials set session and redirect to dashboard |
| test_login_invalid_password | Wrong password returns error flash, no session |
| test_login_inactive_account | Inactive account returns appropriate error message |
| test_logout_clears_session | Logout removes user_role from session |
| test_password_reset_token_valid | Valid token allows password change |
| test_password_reset_token_expired | Expired token returns error without changing password |

### Module 11: User model

| Test | Description |
|---|---|
| test_create_user_persists | User.create() persists to database and returns User instance |
| test_password_not_readable | Accessing user.password raises AttributeError |
| test_verify_password_correct | verifyPassword with correct password returns True |
| test_verify_password_incorrect | verifyPassword with wrong password returns False |
| test_update_returns_success_tuple | Successful update returns (True, message) |
| test_delete_removes_record | delete() removes user from database |

---

# 6. Environment Configuration

All environment-specific values are stored in environment variables. No credentials, secret keys, API keys, or paths appear in source code or version control. A `.env.example` file listing every required variable is maintained in the repository. The actual `.env` file is listed in `.gitignore`.

**Required RAG backend environment variables:**

```
OPENAI_API_KEY
TAVILY_API_KEY              # Optional. System degrades gracefully when absent.
SETTLEBOT_API_KEY           # Must not be "your_secure_random_key_here"
SETTLEBOT_LOCALE            # e.g. "nairobi" — maps to config/locale/{locale}.json
```

**Required Flask frontend environment variables:**

```
SECRET_KEY
DATABASE_URL                # Defaults to SQLite development.db if absent
ORGANIZATION_NAME
MAIL_USERNAME
MAIL_PASSWORD
MAIL_SERVER
MAIL_PORT
```

---

# 7. Version Control and CI Conventions

## 7.1 Branch strategy

Three permanent branches: `main` (production), `staging` (release candidate), `develop` (active development). Feature work uses short-lived branches named `feature/short-description` created from `develop`. Bug fixes use `fix/short-description`. Hotfixes branch from `main` and are named `hotfix/short-description`.

## 7.2 Commit message format

Every commit message must follow this structure exactly: a subject line, a blank line, and a body. The body is not optional.

```
type(scope): short description in present tense

Body explaining what changed and why. Not what the code does line by line,
but why the change was made, what problem existed before, or what decision
was taken. Use as many lines as needed. Wrap at 72 characters per line.
```

**Types:** `feat`, `fix`, `test`, `refactor`, `docs`, `chore`

**Scopes:** `query-pipeline`, `intent`, `embeddings`, `vector-db`, `language`, `response-gen`, `chunking`, `document-ingest`, `evaluator`, `api`, `auth`, `frontend`, `config`, `locale`, `cli`, `tests`, `docs`

**Subject line rules:**
- Present tense, lowercase after the closing parenthesis
- No full stop at the end
- Maximum 72 characters
- Describes what changes, not how

**Body rules:**
- Separated from subject by exactly one blank line
- Explains why the change was made and what problem existed before
- Where a change addresses a documented finding, reference it (e.g. `addresses FEEDBACK.md Phase 1`)
- Maximum 72 characters per line
- Must be present on every commit without exception

**Example:**

```
fix(query-pipeline): remove duplicate language detection call

Language detection was executing twice per query: once in
api/main.py:578 and again inside response_generator.generate_response().
This doubled LLM cost and added 1-2 seconds to every response.

The call at api/main.py lines 577-589 is removed. The response
generator's internal call at line 660 is the single source of truth
for language detection results.

addresses FEEDBACK.md Phase 1
```

## 7.3 CI pipeline

The pipeline runs on every push to `develop`, `staging`, and `main`. It stops on the first failure.

```
1.  Install backend dependencies
2.  Verify OPENAI_API_KEY is set in CI secrets
3.  Run Ruff linting — fail on any error
4.  Run Pyright type checking — fail on any error
5.  Run backend tests
6.  Fail if any test in Module 2, 5, or 9 fails (these are regression-critical)
7.  Run Flask frontend tests
8.  Build production bundle verification
9.  Deploy to staging (staging branch — automatic)
10. Deploy to production (main branch — manual trigger required)
```

---

# 8. Known Issues and Mandatory Fixes

Issues documented in `FEEDBACK.md` that represent production blockers. These must be resolved before any public-facing deployment. See `FEEDBACK.md` for full findings and proposed fixes.

| Priority | Finding | File | Status |
|---|---|---|---|
| P0 | `print(query_response)` dumps full response to stdout on every request | `api/main.py:645` | Open |
| P0 | API key is the literal placeholder string committed to config | `config/config.yaml:122` | Open |
| P0 | Evaluator calls non-existent `recognize_intent()` method | `evaluator.py:466,1352` | Open |
| P0 | LLM hallucination of phone numbers — grounding rule absent from prompt | `response_generator.py` | Open |
| P0 | File upload path traversal via `file.filename` without sanitisation | `api/main.py:799` | Open |
| P0 | Language detection called twice per request | `api/main.py:578` | Open |
| P1 | All remaining `print()` statements (7 locations) | Multiple files | Open |
| P1 | No tenacity retry on LLM calls in language_processor and response_generator | Multiple files | Open |
| P1 | No query length validation on `/query` endpoint | `api/main.py` | Open |
| P1 | No rate limiting on any endpoint | `api/main.py` | Open |
| P1 | No request correlation ID | `api/main.py` | Open |

---

*Document reference: SB-TECH-2026-001 · Version 1.0 · SettleBot Project · May 2026*
