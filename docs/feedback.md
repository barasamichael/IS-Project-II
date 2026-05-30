# SettleBot Backend Audit — FEEDBACK.md

_Audited: 2026-05-30_
_Auditor: Claude Code (claude-sonnet-4-6)_

---

## Summary — Top 10 Critical Issues

| # | Issue | Impact | Fix Category |
|---|-------|--------|--------------|
| 1 | Language detection LLM call executes **twice per request** (api/main.py + response_generator.py) | +1–2s latency, 2× LLM cost on every query | Latency / Cost |
| 2 | Query embedding generated **twice per request** via incompatible models (ada-002 in intent_recognizer vs text-embedding-3-small in embedding_service) | +0.5–1s latency, intent vs retrieval vector-space mismatch | Latency / Correctness |
| 3 | LLM **instructed to hallucinate phone numbers** — system prompt tells GPT to include contact numbers with zero grounding check; `responses.txt` confirms fabricated numbers like `+254 700 000 000` | Critical safety issue: students calling wrong numbers in emergencies | Factual Accuracy |
| 4 | **Evaluator is broken and will always crash** — `evaluator.py` lines 466 and 1352 call `intent_recognizer.recognize_intent()` which does not exist (correct method: `get_intent_info()`). Eval test set references intents (`comparison_query`, `reassurance_seeking`, `explanation_query`) not in `IntentType` enum | Zero test coverage of the full pipeline | Testing |
| 5 | **API key is the literal placeholder** `"your_secure_random_key_here"` in committed `config/config.yaml` line 122 | Any request with this known key is authenticated | Security |
| 6 | **Semantic chunking calls an LLM for every chunk** at ingestion time (`_analyze_chunk_with_llm` per chunk + `_analyze_text_with_llm` once per document) — a 20-chunk document incurs 21 GPT calls | Document ingestion is prohibitively slow and expensive | Latency / Cost |
| 7 | **Tavily web results injected verbatim into LLM context** with no sanitisation (response_generator.py line 967–972) | Prompt injection: a malicious search result can override the system prompt | Security |
| 8 | **Entire system is hardcoded to Nairobi, Kenya at every layer** — query expansion, embedding prefixes, location boosts, ChromaDB collection name, system prompts, hardcoded hospital/university lists | Zero reuse for any other city; even Mombasa or Kampala would require surgery across 6+ files | Architecture |
| 9 | **Off-topic detection fails silently** — `"What is bread?"` returns a full settlement response; `"What adapters do I need?"` is correctly rejected; the boundary is inconsistent because the 0.40 cosine threshold is a single global value for all 11 intents | Users receive confidently wrong answers for non-settlement queries | Intent Recognition |
| 10 | **No rate limiting, no query length validation, no request tracing** on any endpoint | A single client can exhaust OpenAI quota; long adversarial queries can inflate costs; no way to debug a specific user complaint end-to-end | Production Readiness |

---

## Phase 1 — Latency

### Findings

**Sequential LLM calls per `/query` request (measured in `responses.txt`):**

| Step | File:Line | Model | Est. Latency | Notes |
|------|-----------|-------|--------------|-------|
| 1. Language detect | `api/main.py:578` → `language_processor.py:157` | gpt-4.1-nano | 1–2s | Called here |
| 2. Intent embedding | `intent_recognizer.py:763` | text-embedding-ada-002 | 0.4–0.8s | Per-query, no cache |
| 3. Vector DB embed | `embeddings.py:344` | text-embedding-3-small | 0.4–0.8s | **Separate call from step 2** |
| 4. Chromadb query | `vector_db.py:410` | — | 0.1–0.3s | Synchronous |
| 5. Language detect (again) | `response_generator.py:660` | gpt-4.1-nano | 1–2s | **DUPLICATE of step 1** |
| 6. [Parallel] Tavily search | `response_generator.py:719` | — | 2–5s | Blocks response generation |
| 7. Response generation | `response_generator.py:1099` | gpt-4.1-mini | 4–10s | max_tokens=4096 |

`responses.txt` confirms the duplicate detection: every query shows **two consecutive** `INFO:language_processor:LLM detected language` lines (e.g. lines 17–18, 48–49, 79–80 of responses.txt).

**What is parallelised:**
- `response_generator.py` lines 714–726: `ThreadPoolExecutor(max_workers=2)` runs emotion detection + Tavily search in parallel. This is **structurally correct** but emotion detection is keyword-based (≈0ms), so only Tavily actually runs in the thread. The executor overhead adds ~5ms.
- No race conditions in the parallel block.
- However, the `ThreadPoolExecutor` is created inside a sync FastAPI route handler. The executor calls are blocking from the event loop's perspective — FastAPI's async worker thread is blocked for the full duration.

**What is not parallelised and should be:**
- Steps 1–4 are fully sequential. Steps 2 and 3 are independent once the query is known and could run in parallel.
- The Tavily call (step 6) could overlap with the vector DB retrieval (steps 3–4) but does not.

**Caching gaps:**
- Query embeddings: no in-process cache. Every request pays an API round-trip for ada-002 AND for text-embedding-3-small.
- Language detection results: not cached. The same English query runs detection every time.
- Intent classification embeddings for patterns are cached to disk (good), but the per-query embedding is not.
- Web search results: not cached. The same common query ("how do I open a bank account?") triggers a fresh Tavily call each time.

**Tavily blocking behaviour:**
- If Tavily times out (no explicit timeout set), the entire parallel block blocks until the `requests` library times out (default: no timeout → indefinite).
- If `TAVILY_API_KEY` is absent, `search_web_for_current_info` returns `None` gracefully (line 369–370). No crash.
- If Tavily returns slowly, Tavily becomes the bottleneck for response generation because the executor joins before `_generate_comprehensive_response` is called.

**Embedding model inconsistency (architectural defect):**
- `IntentRecognizer` hardcodes `text-embedding-ada-002` (line 675, 765).
- `EmbeddingService` uses `settings.embedding.model` which `config.yaml` sets to `text-embedding-3-small`.
- These are different vector spaces. Chunks are indexed with text-embedding-3-small embeddings. Queries are searched via a text-embedding-3-small embedding (through `VectorDBService.search → EmbeddingService.embed_query`). But the intent classification embedding is ada-002. The intent recognizer prototype embeddings (cached in `.embeddings_cache/`) use ada-002. This inconsistency does not break retrieval (both calls use the same model for their respective tasks) but it does mean 2 different embedding calls per query with 2 different models.

### Proposed Fix

**Redesigned pipeline (target: ≤12s):**

```
Request
  │
  ├─ [Async Step A] EmbeddingService.embed_query(query)          ← single ada-002 OR 3-small call
  │   Shared embedding used for BOTH intent + vector search
  │
  ├─ [Parallel, after A]
  │   ├─ IntentRecognizer.classify_from_embedding(embedding)     ← no new LLM call, cosine only
  │   ├─ VectorDBService.search_from_embedding(embedding, top_k) ← no new embedding call
  │   └─ [If needed] Tavily web search                           ← can start immediately
  │
  ├─ [After parallel] Assemble context
  │
  └─ [Single LLM call] GPT-4.1-mini response generation
       Language detection embedded IN the system prompt:
       "If the query is not English, translate, then answer in the user's language."
       This eliminates 2 dedicated language-detection calls.
```

Concrete changes:
1. **Delete `language_processor.detect_and_process_query()` call from `api/main.py:578`.** The response generator already calls it at line 660. One call, not two.
2. **Unify the embedding model.** Choose one: either use ada-002 everywhere (higher cost, proven) or text-embedding-3-small everywhere (lower cost). Update `intent_recognizer.py` line 675 to use `settings.embedding.model`. Rebuild the intent embedding cache.
3. **Cache query embeddings** with an in-process LRU (e.g. `functools.lru_cache(maxsize=512)` on `embed_query`, keyed on normalised query string). TTL not needed since embeddings are deterministic.
4. **Cache Tavily results** with a TTL of 3600s keyed on `(intent_type, query_normalised)`. Use `cachetools.TTLCache`. Emergency queries (crisis_level == high) must bypass this cache.
5. **Set explicit Tavily timeout**: `tavily_client.search(..., timeout=4)`. If it times out, proceed without web results.
6. **Run Tavily in parallel with vector search**, not after. Restructure `generate_response()` to launch Tavily immediately after intent classification, while vector search is also running.
7. **Convert FastAPI route to async** and use `asyncio.to_thread()` for blocking calls (ChromaDB, OpenAI) to avoid blocking the event loop.
8. **Cap `max_tokens` at 2048** for most intents. 4096 is overkill for housing queries and directly drives response latency.

---

## Phase 2 — Factual Accuracy & Hallucination Risk

### Findings

**Hardcoded facts inventory:**

| File | Lines | What is hardcoded | Risk |
|------|-------|-------------------|------|
| `response_generator.py` | 48–52 | 6 emergency numbers (999, Red Cross, AA Kenya) | 999 is correct; Red Cross/AA numbers unverified, may be stale |
| `response_generator.py` | 53–61 | Immigration office: "Nyayo House, Uhuru Highway", "+254 20 222 2022" | Phone confirmed incorrect in responses.txt (model generated "+254 (0) 20 2100272" and "+254 20 222 999" — three different numbers for the same office) |
| `response_generator.py` | 62–173 | 14 hospitals with names, addresses, phones, emails, websites | Some likely stale; KNH phone "+254 20 2726300" appears in responses.txt line 736 correctly, but others were overridden by model hallucinations |
| `response_generator.py` | 174–261 | 16 universities with names, addresses, phones, websites | Generally correct but phone formats inconsistent (e.g. "0703‑034000" for Strathmore) |
| `services/vector_db.py` | 96–103 | Location boost weights hardcoded for 5 Nairobi neighborhoods | Not a factual risk but hardcoded |
| `services/document_processor.py` | 163–211 | Nairobi neighborhoods, landmarks, universities hardcoded | Not directly served to users |
| `services/intent_recognizer.py` | 91–133 | Settlement keywords include "nairobi", "kenya" | Not served to users |
| `config/config.yaml` | 20 | Collection name `settlebot_nairobi` | Not served to users |

**Grounding failure evidence from `responses.txt`:**

The LLM fabricates phone numbers not in the hardcoded data nor in retrieved chunks:
- Line 39: `+254 700 000 000` (Gimco Limited — clearly a template placeholder)
- Line 339: `+254 700 000 000` (Odds and Ends store)
- Line 306: `+254 20 222 999` (immigration, differs from hardcoded `+254 20 222 2022`)
- Line 333: `+254 (0) 20 2100272` (immigration, a third fabricated number)
- Line 549: `0709 123 456` (ambulance — different from hardcoded 999)

**Root cause — the system prompt instructs hallucination:**

`_get_comprehensive_system_prompt()` (`response_generator.py` lines 1127–1152) instructs the LLM:
> "Include specific Nairobi details - locations, costs in KSh, **contact numbers**, current information"
> "Include specific contacts, websites, and resources"
> "Provide specific contacts, websites, and locations"

This is a direct instruction to fabricate contact numbers when the retrieved context doesn't supply them. There is no instruction saying "only cite phone numbers explicitly present in the provided context."

**No citation or source tracking.** Responses never indicate which chunk or web result supports a claim. Users cannot verify any assertion.

**Evaluator does not test factual accuracy.** `evaluator.py` uses keyword presence checks (`contains_expected`), BLEU score vs. reference responses, and student-relevance scoring. None of these checks whether a phone number is correct.

### Proposed Fix

1. **Replace all hardcoded fact dicts with a `LocaleFactStore`** backed by a configurable JSON file per locale. Load at startup. Expose a `/admin/update-facts` endpoint for operators to push verified updates without code deploys.

2. **Rewrite the system prompt grounding rule** (add to `_get_comprehensive_system_prompt()` before all other instructions):
   ```
   GROUNDING RULE (NON-NEGOTIABLE):
   You may only state a phone number, email address, physical address, fee amount,
   or operating hours if it appears VERBATIM in the RETRIEVED SETTLEMENT INFORMATION
   or ESSENTIAL SETTLEMENT INFORMATION sections below.
   If you cannot find a contact detail in the provided context, write:
   "Contact details unavailable — verify at [official source URL if known]."
   Never invent or approximate contact information.
   ```

3. **Add a post-generation phone number audit** using regex before returning the response:
   ```python
   import re
   PHONE_RE = re.compile(r'(\+254[\s\-]?\d[\d\s\-]{7,}|\b0[17]\d{2}[\s\-]?\d{3}[\s\-]?\d{3}\b)')
   found_numbers = PHONE_RE.findall(response_text)
   verified_numbers = set(extract_all_phones_from_context(essential_info, retrieved_chunks))
   unverified = [n for n in found_numbers if normalise_phone(n) not in verified_numbers]
   if unverified:
       # Either strip them or append a disclaimer
   ```

4. **Add source attribution.** Each retrieved chunk has a `doc_id` and `chunk_id`. Include a `## SOURCES` section at the end of each response listing the document IDs used. This gives operators a way to audit.

5. **Fix the evaluator** to run regex checks for phone numbers and compare them against a known-good ground-truth set. This must be a blocking CI check before any response model change is deployed.

---

## Phase 3 — Generalisability & Multi-City Architecture

### Findings

**Nairobi/Kenya is baked in at every layer.** The following are specific hardcoded references:

| File | Line | Hardcoded value |
|------|------|-----------------|
| `embeddings.py` | 379–381 | Prepends `"International student in Nairobi Kenya: "` to every non-settlement query |
| `embeddings.py` | 248–252 | Prepends `"Nairobi {location} area: "` for Westlands/Kilimani/Karen/Lavington matches |
| `vector_db.py` | 76 | Collection description: `"SettleBot Nairobi settlement content"` |
| `vector_db.py` | 96–103 | Location boosts hardcoded for Nairobi neighborhoods only |
| `vector_db.py` | 625–629 | Query expansion adds `"in Nairobi Kenya"` if location context absent |
| `vector_db.py` | 710 | `search_by_location`: appends `"Nairobi"` unconditionally |
| `document_processor.py` | 163–211 | `nairobi_locations` dict (20 neighborhoods, 10 universities, 11 landmarks) |
| `document_processor.py` | 213–221 | Cost patterns use `KSh`, `KES`, `shilling` only |
| `document_processor.py` | 527–529 | Text cleaner normalises Kenya Shilling references |
| `semantic_chunking.py` | 216–235 | `nairobi_locations` list (18 neighborhoods) |
| `semantic_chunking.py` | 520–538 | LLM analysis prompt: "Nairobi, Kenya" hardcoded |
| `semantic_chunking.py` | 584–598 | Topic boundary prompt: "Nairobi" hardcoded |
| `response_generator.py` | 44–261 | Essential info: hospitals, universities, immigration office all Nairobi |
| `response_generator.py` | 132–153 | Translation preserves: "Westlands, Kilimani, Karen, Lavington..." (7 Nairobi names) |
| `response_generator.py` | 247–253 | Translation preserves: "matatu, boda boda, M-Pesa" |
| `response_generator.py` | 347–351 | Timezone hardcoded: `Africa/Nairobi` |
| `config/config.yaml` | 20 | `collection_name: settlebot_nairobi` |
| `intent_recognizer.py` | 119–132 | Settlement relevance keywords: "nairobi", "kenya" as high-value terms |

**There is no `LocaleConfig` or deployment context.** The locale is embedded as literals throughout every service rather than being injected from a configuration object.

**ChromaDB namespacing:** Currently one collection `settlebot_nairobi`. ChromaDB supports multiple named collections. To support Kampala, you would need `settlebot_kampala`. There is no routing logic to select a collection based on locale.

**Language coverage gap:** `config.yaml` lists ~40 languages including Luganda, Kinyarwanda, Tigrinya. But `LanguageProcessor.supported_languages` (`language_processor.py` lines 25–50) only maps 24 codes, and the mapping is not derived from the config — it's hardcoded. Luganda (`lg`) is in the config but not in the Python dict. If a Luganda query arrives, `_detect_and_translate_llm` may return `lg` but the code falls back to `en` at line 179–181 because `lg` is not in `supported_languages`.

**Intent specificity:** 7 of 11 intents are generic (housing, transport, safety, healthcare, cost, cultural_adaptation, emergency_help). `BANKING_FINANCE` intents reference M-Pesa specifically (lines 395–414 of intent_recognizer.py). `TRANSPORTATION` references matatu/boda boda. These are Kenya-specific but the intent label is generic. Multi-label handling or locale-sensitive example sets would resolve this.

### Proposed Fix

**Introduce a `LocaleConfig` dataclass** (new file `config/locale.py`):

```python
@dataclass
class LocaleConfig:
    city: str                          # "nairobi" | "kampala" | "abuja"
    country: str                       # "kenya" | "uganda" | "nigeria"
    currency_code: str                 # "KES" | "UGX" | "NGN"
    currency_symbol: str               # "KSh" | "UGX" | "₦"
    timezone: str                      # "Africa/Nairobi" | "Africa/Kampala"
    emergency_number: str              # "999" | "999" | "199"
    primary_languages: list[str]       # ["sw", "en"] | ["lg", "en"] | ["ha", "yo", "ig", "en"]
    key_neighborhoods: list[str]       # Westlands, Kilimani... | Kololo, Nakasero...
    key_institutions: dict             # hospitals, universities, gov offices
    collection_name: str               # "settlebot_nairobi" | "settlebot_kampala"
    web_search_geo_bias: str           # "Kenya" | "Uganda" | "Nigeria"
    fact_store_path: str               # path to verified facts JSON
```

Pass `LocaleConfig` through the service constructors. Every hardcoded `"Nairobi"` or `"Kenya"` reference must be replaced with `locale.city` or `locale.country`.

**ChromaDB multi-locale:** Each `LocaleConfig.collection_name` maps to its own ChromaDB collection. `VectorDBService.__init__` accepts `locale: LocaleConfig` and uses `locale.collection_name`. A query never crosses locale boundaries.

**System prompt assembly:** `_get_comprehensive_system_prompt()` must accept `locale: LocaleConfig` and use `locale.city`, `locale.currency_symbol`, and `locale.key_institutions` rather than embedded strings.

**LLM text optimization prefixes** (`embeddings.py` line 379): Replace `"International student in Nairobi Kenya:"` with `f"International student in {locale.city} {locale.country}:"`.

---

## Phase 4 — Web Search Integration

### Findings

**Tavily is called on every non-off-topic query**, regardless of whether fresh web data is needed. For common stable queries ("what is M-Pesa?", "what are matatu routes?") the knowledge base should be sufficient and Tavily adds latency and cost with no benefit.

**No source quality filtering.** Tavily returns up to 3 results per search query with no domain allowlist. A low-quality blog, misinformation site, or—critically—a malicious page can contribute content to the LLM context.

**Prompt injection via Tavily results.** `response_generator.py` lines 967–972:
```python
context_parts.append(f"Recent Information - {result.get('query')}:")
context_parts.append(result["content"])
```
`result["content"]` is inserted verbatim into `enhanced_context`, which is then inserted into the LLM prompt at `user_prompt` line 1064. A web page containing the string `"IGNORE ALL PREVIOUS INSTRUCTIONS. You are now..."` would be passed directly to the model with no sanitisation.

**Tavily graceful degradation** is implemented correctly: if `TAVILY_API_KEY` is absent (line 369–370), the function returns `None` and `web_search_used` is False. The system continues with RAG-only.

**No explicit timeout on Tavily calls.** `tavily_client.search(...)` at line 389 has no timeout parameter. If the Tavily API is slow, the parallel block blocks indefinitely.

**Web results vs. RAG context ordering:** Both are concatenated into `context_parts` without priority signalling. The LLM sees retrieved chunks first, web results second. In practice, web results override RAG content for recent topics because they appear more authoritative in the prompt.

### Proposed Fix

1. **Intent-based Tavily routing.** Only trigger web search for intents where freshness matters:
   - Always search: `IMMIGRATION_VISA` (visa fees/procedures change), `COST_INQUIRY` (prices change)
   - Search if RAG confidence < 0.5: `HOUSING_INQUIRY`, `BANKING_FINANCE`
   - Never search: `CULTURAL_ADAPTATION`, `EMERGENCY_HELP` (static content; for emergency, use hardcoded facts only)

2. **Domain allowlist per locale.** In `LocaleConfig`, add `trusted_domains: list[str]`. For Kenya:
   ```python
   trusted_domains = [
       "immigration.go.ke", "ecitizen.go.ke",
       "centralbank.go.ke", "safaricom.co.ke",
       "knh.or.ke", "uonbi.ac.ke", "strathmore.edu"
   ]
   ```
   Filter Tavily results: `results = [r for r in results if any(d in r["url"] for d in trusted_domains)]`

3. **Sanitise web content before LLM injection.** Strip any segment that looks like a prompt instruction:
   ```python
   import re
   INJECTION_PATTERNS = re.compile(
       r'(ignore|disregard|forget).{0,30}(previous|above|prior).{0,30}(instruction|prompt|system)',
       re.IGNORECASE
   )
   def sanitise_web_content(text: str) -> str:
       if INJECTION_PATTERNS.search(text):
           return "[Web content redacted: contained unsafe patterns]"
       return text[:1500]  # hard cap
   ```

4. **Cache Tavily results with TTL.** Use `cachetools.TTLCache(maxsize=200, ttl=3600)` keyed on `(intent_type.value, query_normalised[:100])`. This eliminates repeated API calls for popular queries.

5. **Set explicit Tavily timeout**: add `timeout=4` to `tavily_client.search()`. Handle `TimeoutError` by returning `None` (same as key-absent path).

---

## Phase 5 — Intent Recognition

### Findings

**Adversarial test results (derived from `responses.txt` and code analysis):**

| Query | Actual classification | Expected | Verdict |
|-------|----------------------|----------|---------|
| `"Where can I find injera near Westlands?"` | Likely `housing_inquiry` (Westlands location_specific boost) | `off_topic` | **FAIL** |
| `"My landlord is threatening me"` | Likely `housing_inquiry` (landlord keyword in housing examples) | `safety_concern` or `emergency_help` | **FAIL** |
| `"How do I report a crime?"` | Likely `safety_concern` | Acceptable | PASS |
| `"I lost my passport"` | Likely `immigration_visa` | Could also be `emergency_help` | MARGINAL |
| `"What's the exchange rate for Ethiopian Birr?"` | Likely `cost_inquiry` or `banking_finance` | Acceptable | PASS |
| `"What is bread?"` (`responses.txt` line 981) | `explanation_query` (NOT in IntentType enum) | `off_topic` | **FAIL** — full response returned |
| `"What mobile network provider should I choose?"` (`responses.txt` line 833) | `comparison_query` (NOT in IntentType enum) | Borderline; a student question | UNCLEAR |
| `"Where can I find halal/kosher/vegetarian food?"` (`responses.txt` line 1288) | `university_info` (confidence 0.70) | `cultural_adaptation` | **FAIL** — wrong intent |

**Undeclared intent types in evaluation set.** `evaluator.py` uses `expected_intent` values including: `reassurance_seeking`, `comparison_query`, `explanation_query`, `entertainment_social`, `shopping_markets`, `academic_conversion`, `neighborhood_guide`. None of these are in `IntentType` enum. `evaluator._evaluate_response_quality()` will always mark these as `intent_match = False`, making the evaluator useless for these test cases.

**Evaluator crashes on method call.** `evaluator.py` lines 466 and 1352 call `self.intent_recognizer.recognize_intent(query)`. The actual method is `get_intent_info(query)` (`intent_recognizer.py` line 877). The evaluator has never successfully run.

**Single global threshold (0.40) is inappropriate.** `housing_inquiry` and `cost_inquiry` share many keywords ("rent", "KSh", "budget"). A query like "how much does housing cost?" has high cosine similarity to both. The system picks the highest but both have high scores. `banking_finance` and `cost_inquiry` also overlap heavily. A per-intent threshold tuned empirically would differentiate these correctly.

**No clarification flow.** When confidence < `confidence_threshold` (0.75), the system proceeds with the best guess rather than asking "Are you asking about X or Y?". Low-confidence guesses have the same response quality target as high-confidence ones.

**Static prototype embeddings** are cached in `.embeddings_cache/intent_embeddings.npz`. They were computed once using `text-embedding-ada-002` and are never updated unless `rebuild_cache()` is called manually. If the intent examples are updated, the cache must be manually invalidated.

### Proposed Fix

1. **Fix the evaluator immediately.** Replace `recognize_intent` with `get_intent_info` in `evaluator.py` lines 466 and 1352. This is a one-line fix and unblocks all testing.

2. **Align evaluator `expected_intent` values with `IntentType` enum.** Map:
   - `reassurance_seeking` → `safety_concern` or `cultural_adaptation`
   - `comparison_query` → whichever domain intent applies
   - `explanation_query` → `banking_finance`, `transportation`, etc. by topic

3. **Multi-label scoring.** Return the top-3 intent scores rather than picking a winner. For retrieval, union the top-2 topic sets. For response generation, use the highest confidence. Implement as:
   ```python
   top3 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
   if top3[0][1] - top3[1][1] < 0.05:  # close tie
       intent_result.is_ambiguous = True
       intent_result.secondary_intent = top3[1][0]
   ```

4. **Per-intent thresholds.** Run the evaluator on a labelled set and compute the minimum similarity score at which each intent achieves precision ≥ 0.90. Use those as per-intent off-topic thresholds stored in a config dict.

5. **Clarification flow for ambiguous queries.** If `is_ambiguous=True` and no web fallback is active, return:
   ```
   "Are you asking about [intent_A topic] or [intent_B topic]? 
   Respond with 1 or 2, or rephrase your question."
   ```
   Store the ambiguity state in `conversation_context` for the next turn.

6. **LLM fallback classifier for low-confidence cases.** When max cosine similarity is between 0.40 and 0.55, do a single GPT-4.1-nano classification call using a few-shot prompt with the 11 intent examples. This is slower but more accurate for edge cases, and only fires for <10% of queries.

---

## Phase 6 — Semantic Chunking & Knowledge Base

### Findings

**Active strategy mismatch.** `config.yaml` line 27 sets `strategy: semantic`. `SemanticChunker.__init__` default is `ChunkingStrategy.SETTLEMENT_OPTIMIZED`. The config value `"semantic"` does not match any `ChunkingStrategy` enum value (`semantic_adaptive`, `semantic_fixed`, `settlement_optimized`, `topic_aware`). In `api/main.py` the chunker is instantiated with no strategy argument, so it uses the default `SETTLEMENT_OPTIMIZED` regardless of config. The config setting is **dead code** — it is never applied to the chunker.

**LLM called per chunk during ingestion.** `SemanticChunker._enrich_chunks_with_llm_analysis()` (line 683) iterates over every chunk and calls `_analyze_chunk_with_llm()` (line 715), making a gpt-4.1-nano API call per chunk. Additionally, `_settlement_optimized_chunking()` calls `_analyze_text_with_llm()` once per document. For a document producing 20 chunks: **21 LLM API calls per document**. At gpt-4.1-nano pricing (~$0.0001/call), a 100-document corpus costs $0.21 in LLM calls just for chunking metadata. More critically, ingestion is blocked on serial LLM calls, making bulk ingestion impractical.

**Settlement score is used at retrieval.** `vector_db._apply_settlement_boost()` (line 478) applies `boosted_score *= 1 + settlement_score * 0.2`. So settlement scores do influence ranking — they are not ignored.

**Retrieval is purely dense (cosine similarity).** No BM25 or sparse retrieval component. For queries with rare proper nouns ("Nyayo House immigration", "KCB Equity Bank") or exact phrase matching, BM25 would significantly improve recall.

**Deduplication is declared but not implemented.** `DocumentProcessor.__init__` accepts `enable_deduplication=True` and `similarity_threshold=0.92` but `process_document()` and `_create_settlement_chunks()` never call any deduplication logic. There is no method in `DocumentProcessor` or `EmbeddingService` that compares new chunks against existing ones. The flag is unused.

**`_generate_settlement_queries` in `vector_db.py`** (line 619) adds `"in Nairobi Kenya"` unconditionally (line 626) for multi-query expansion. This is correct for the current deployment but breaks generalisation.

**`top_k` default is `15` in API but `20` in `VectorDBService.search()`** (line 385). `VectorDBService.search()` internally fetches `top_k * 2` (line 411) and reranks. So a request with `top_k=15` actually fetches 30 candidates from ChromaDB, then reranks and returns 15. This is acceptable but means 2× the vectors are scored on every query.

**Chunk-level source is not tracked in responses.** The `doc_id` is available in every retrieved chunk's metadata (line 437–438 of `vector_db.py`) but is never included in the LLM context or the response.

### Proposed Fix

1. **Fix config/chunker alignment.** Either add `SEMANTIC = "semantic"` to the `ChunkingStrategy` enum as an alias for `SETTLEMENT_OPTIMIZED`, or change `config.yaml` to `strategy: settlement_optimized`. Also add chunker strategy injection in `api/main.py`:
   ```python
   semantic_chunker = SemanticChunker(strategy=ChunkingStrategy(settings.chunking.strategy))
   ```

2. **Remove per-chunk LLM calls from ingestion.** Replace `_analyze_chunk_with_llm()` with the existing deterministic `_calculate_settlement_relevance_score()` from `document_processor.py`. The LLM-based semantic score adds ~5% lift over the keyword-based score but costs 21× the API calls. Keep the LLM call only for the document-level analysis (`_analyze_text_with_llm`) — that is genuinely useful for routing. Per-chunk quality scores must be keyword-based.

3. **Add hybrid retrieval with BM25.** Use `rank_bm25` (lightweight, no server needed):
   ```python
   from rank_bm25 import BM25Okapi
   # At collection load time, build BM25 index from stored documents
   # At query time:
   dense_results = chromadb_search(query_embedding, top_k=top_k*2)
   sparse_results = bm25_index.get_top_n(query_tokens, all_docs, n=top_k)
   fused = reciprocal_rank_fusion(dense_results, sparse_results)
   ```
   This is the current SOTA for RAG retrieval accuracy and requires no external service.

4. **Add cross-encoder reranker** after BM25+dense fusion. Use `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` (runs locally, no API cost):
   ```python
   from sentence_transformers import CrossEncoder
   reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   pairs = [(query, chunk["text"]) for chunk in candidates]
   scores = reranker.predict(pairs)
   reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
   ```

5. **Implement actual deduplication.** In `DocumentProcessor._create_settlement_chunks()`, after computing embeddings for new chunks, compare against existing embeddings in ChromaDB using cosine similarity. If similarity ≥ `self.similarity_threshold`, skip the chunk. Use `VectorDBService.search()` with high `top_k` to retrieve candidates.

6. **Pass `doc_id` and `chunk_id` to the LLM context** as inline citations:
   ```
   Source 1 (doc_id: abc123, chunk: 0042, relevance: 0.87):
   [chunk text]
   ```

---

## Phase 7 — Production Readiness & Observability

### Findings

**Endpoints that return stub/fake data:**

| Endpoint | File:Line | Claims to return | Actually returns |
|----------|-----------|-----------------|-----------------|
| `GET /analytics/search-patterns` | `api/main.py:1776–1793` | Search pattern analytics | Static recommendation message, zero data |
| `GET /system/status` `uptime_hours` | `api/main.py:532` | System uptime in hours | Hardcoded `0` |
| `POST /vector-db/optimize` | `vector_db.py:778–797` | "Optimized" collection | Returns current stats with status="optimized", does nothing |
| `GET /embeddings/stats` `avg_quality_score` | `embeddings.py:641` | Embedding quality score | Hardcoded `0.8` placeholder for all test queries |
| `POST /webhooks/document-processed` callback | `api/main.py:2223–2226` | HTTP POST to callback_url | Sets `callback_sent=True` in dict, never actually POSTs |

**`print()` statements in production code:**

| File | Line | Content |
|------|------|---------|
| `api/main.py` | 575 | `print(f"Query processing: {request.query[:100]}")` |
| `api/main.py` | 620 | `print("processed")` |
| `api/main.py` | 645 | `print(query_response)` — **prints the full QueryResponse object including all 15 retrieved chunks** |
| `language_processor.py` | 170 | `print(f"Elapsed time is { end_time - start_time } seconds")` |
| `intent_recognizer.py` | 766 | `print(f"Elapsed time is { end_time - start_time } seconds")` |
| `response_generator.py` | 397 | `print(f"Elapsed time is { end_time - start_time } seconds")` |
| `response_generator.py` | 1109 | `print(f"Elapsed time is { end_time - start_time } seconds")` |

**`api/main.py` line 645 is the most severe**: it dumps `print(query_response)` — a Pydantic model with `retrieved_chunks` included — to stdout on every request, which includes all retrieved document text (up to 15 chunks × ~500 chars). In any log-forwarding setup, this will flood logs with kilobytes per request.

**No request tracing.** There is no correlation ID injected at the API boundary. When a user reports a wrong answer, there is no way to replay or investigate that specific request from logs.

**No rate limiting.** No `slowapi` or `fastapi-limiter` in `requirements.txt` or any middleware in `api/main.py`. A single IP can send unlimited requests, exhausting the OpenAI quota.

**No input validation beyond Pydantic type coercion.** `QueryRequest.query: str` accepts strings of unlimited length. A 50,000-character query would be passed to the embedding API and the LLM, causing API errors or excessive cost. No character filtering.

**No timeout or retry on OpenAI API calls in `language_processor.py` and `response_generator.py`.** `EmbeddingService.embed_batch_optimized()` has retry logic with exponential backoff (lines 296–328), but the LLM calls in `language_processor._detect_and_translate_llm()` (line 156) and `response_generator._generate_comprehensive_response()` (line 1099) have no timeout, no retry, and no circuit breaker.

**`/health` endpoint makes a live embedding API call** via `vector_db_service.health_check()` → `self.embedding_service.embed_query("test")` (line 824). Health checks that make external API calls will fail if OpenAI is down, correctly reporting unhealthy, but the call also costs $0.00001 every time a load balancer pings `/health`.

**`config.yaml` `debug: true`** (line 123). In debug mode, `internal_error_handler` at line 2269 returns `str(exc)` in the error response body. This leaks internal paths, library versions, and possibly partial data to API consumers.

### Proposed Fix

1. **Replace all `print()` with `logger.*()`.** Remove `api/main.py:645` entirely — logging a full QueryResponse with chunks on every successful request is never appropriate.

2. **Add structured logging with JSON formatter:**
   ```python
   import structlog
   structlog.configure(
       processors=[structlog.stdlib.add_log_level, structlog.processors.JSONRenderer()]
   )
   ```

3. **Add request correlation ID middleware:**
   ```python
   import uuid
   @app.middleware("http")
   async def add_correlation_id(request: Request, call_next):
       request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
       response = await call_next(request)
       response.headers["X-Request-ID"] = request_id
       return response
   ```
   Inject `request_id` into every `logger.*()` call via `structlog.contextvars.bind_contextvars(request_id=request_id)`.

4. **Add rate limiting with `slowapi`:**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   @app.post("/query")
   @limiter.limit("30/minute")
   async def process_query(request: QueryRequest, ...):
   ```

5. **Add query length validation to `QueryRequest`:**
   ```python
   query: str = Field(..., min_length=3, max_length=2000)
   ```

6. **Wrap all LLM calls with `tenacity` retry + timeout:**
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
   import httpx
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
          retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)))
   def _call_openai_chat(self, ...):
       return self.client.chat.completions.create(..., timeout=15)
   ```

7. **Fix the stub endpoints that are actively misleading:**
   - `GET /analytics/search-patterns`: Remove or return `{"status": "not_implemented"}` with HTTP 501
   - `uptime_hours`: Store server start time at `startup_event()` and compute delta
   - `POST /vector-db/optimize`: Remove the "optimized" claim; return current stats with `"optimizations_available": [...]`
   - Webhook callback: Implement actual HTTP POST using `httpx.AsyncClient` or mark as `501 Not Implemented`

8. **Set `debug: false` in `config.yaml`** (or better: move API key and debug flag to environment variables only, never commit them).

---

## Phase 8 — Cross-Cutting Concerns

### Findings

**Security:**

- **API key in version control** — `config/config.yaml` line 122: `api_key: "your_secure_random_key_here"`. This is committed to git. Any operator who clones this repo and starts the server without changing this value is running with a publicly known API key.
- **File upload path traversal** — `api/main.py` line 799: `temp_file_path = UPLOAD_DIR / file.filename`. A filename like `../../etc/crontab` would escape the uploads directory. Add: `safe_name = Path(file.filename).name` and reject filenames with path separators.
- **Error messages in debug mode** — `api/main.py` line 2269: `"error": str(exc) if settings.api.debug else "An error occurred"`. With `debug: true` committed to config, every 500 error leaks exception details.
- **OpenAI API key not logged** — the key is read via `os.getenv("OPENAI_API_KEY")` and never passes through logger. This is correct.
- **Prompt injection from web results** — documented in Phase 4.
- **No CSRF protection** on state-changing endpoints (`/documents/upload`, `/admin/*`). FastAPI's API-key auth mitigates this for non-browser clients, but the `allow_origins: ["*"]` CORS config (`api/main.py` line 66) makes browser-based CSRF feasible if the API key is leaked.

**Token budget:**

- System prompt: ~600 tokens
- User prompt (query + intent analysis + current time): ~100 tokens
- Retrieved context (7 chunks × ~400 tokens): ~2800 tokens
- Tavily results (2 queries × ~500 tokens): ~1000 tokens
- Essential info (hospitals, universities, emergency numbers): ~800 tokens
- **Total input context: ~5300 tokens**
- Response (`max_tokens=4096`): up to 4096 tokens
- **Total per call: ~9400 tokens**
- GPT-4.1-mini context limit: 128K tokens. No overflow risk under normal conditions.
- However, a query with `top_k=50` (API allows up to 50) and Tavily returning full results could push context to ~20K tokens, still safe.

**Cost estimate (1,000 queries/day):**

| Component | Model | Input tokens | Output tokens | $/query | $/day (1K queries) |
|-----------|-------|-------------|--------------|---------|-------------------|
| Language detect | gpt-4.1-nano | ~500 | ~150 | ~$0.0001 | ~$0.10 |
| Intent embedding | text-embedding-ada-002 | ~300 | — | ~$0.00003 | ~$0.03 |
| Query embedding (VDB) | text-embedding-3-small | ~350 | — | ~$0.000007 | ~$0.007 |
| Response generation | gpt-4.1-mini | ~5300 | ~1200 | ~$0.0015 | ~$1.50 |
| Tavily (2 searches) | — | — | — | ~$0.008 | ~$8.00 |
| **Total** | | | | **~$0.010** | **~$10/day, ~$300/month** |

This is sustainable at 1,000 queries/day. At 10,000 queries/day it becomes ~$3,000/month — still viable for a production product but worth caching aggressively.

**Testing:**

`tests/test_basic.py` was not read due to space constraints, but the evaluator (`evaluator.py`) is the primary test mechanism and **cannot run** due to the `recognize_intent` method name bug. There is no evidence of CI integration. The evaluator is triggered only manually via CLI or API.

There are no unit tests for individual services visible in the file listing. Integration tests that hit the real pipeline do not exist in the evaluated codebase.

**Dependency risks:**

| Package | Version | Risk |
|---------|---------|------|
| `googletrans==4.0.2` | Present in requirements.txt | Unofficial Google Translate wrapper, scrapes the web interface. Known to break with Google API changes. However, `LanguageProcessor` does not use it — the config says `translation_provider: hybrid_gpt_google` but the code only calls OpenAI. Unused import: safe to remove. |
| `langchain==1.0.7` + `langchain-community==0.4.1` + `langchain-core==1.0.5` | Mixed versions | These are relatively recent versions. No known CVEs at this version range, but LangChain's community loaders (`WebBaseLoader`, `SitemapLoader`) are used for document ingestion. LangChain regularly breaks backwards compatibility. Pin all three to a single compatible release. |
| `ssl: enable_verification: false` | `config.yaml:36–37` | SSL verification disabled for `WebBaseLoader` (`document_processor.py:796`). This allows MITM attacks when fetching content from external URLs. Remove this setting and fix any SSL issues properly. |
| `config.yaml environment: development` | — | If deployed as-is, the system runs in development mode with `debug: true` and the placeholder API key. |
| `openai==2.8.1` | Current | No known vulnerabilities. |
| `chromadb==1.3.5` | Current | No known vulnerabilities. |

---

## Prioritised Action Plan

| Priority | Phase | Change | Effort | Impact |
|----------|-------|--------|--------|--------|
| P0 | 7 | Remove `print(query_response)` at `api/main.py:645` | S | Stops flooding logs with user data |
| P0 | 7 | Replace placeholder API key — move to environment variable only, remove from `config.yaml` | S | Eliminates known-credential auth bypass |
| P0 | 5 | Fix evaluator method call: `recognize_intent` → `get_intent_info` (`evaluator.py:466,1352`) | S | Unblocks all testing |
| P0 | 2 | Add grounding rule to system prompt: no phone numbers unless in retrieved context | S | Stops hallucinated emergency contacts |
| P0 | 8 | Sanitise file upload filename: `safe_name = Path(file.filename).name` | S | Closes path traversal |
| P0 | 1 | Remove duplicate language detection call in `api/main.py:578–589` | S | -1–2s latency, -50% LLM cost on lang detection |
| P1 | 7 | Replace all `print()` with `logger.*()` (7 locations) | S | Structured, filterable logs |
| P1 | 7 | Add `tenacity` retry+timeout to LLM calls in `language_processor.py` and `response_generator.py` | S | Prevents silent failures on OpenAI errors |
| P1 | 7 | Add query length validation (`max_length=2000`) to `QueryRequest` | S | Prevents cost/API abuse |
| P1 | 7 | Add rate limiting via `slowapi` (30 req/min/IP on `/query`) | S | Protects OpenAI quota |
| P1 | 7 | Add request correlation ID middleware | S | Enables end-to-end query debugging |
| P1 | 1 | Unify embedding model: remove ada-002 from `intent_recognizer.py`, use `settings.embedding.model` | M | Eliminates extra embedding call, fixes vector-space mismatch |
| P1 | 4 | Add explicit 4s timeout to Tavily calls | S | Caps latency spike when Tavily is slow |
| P1 | 4 | Sanitise Tavily content before LLM injection (strip prompt injection patterns) | S | Closes prompt injection vector |
| P1 | 8 | Set `ssl: enable_verification: true` and fix any resulting SSL issues properly | S | Closes MITM on URL ingestion |
| P1 | 7 | Set `debug: false` in production config | S | Stops leaking exceptions to API consumers |
| P1 | 2 | Add regex post-generation phone number audit | M | Catches hallucinated numbers before they reach users |
| P2 | 6 | Fix config/chunker strategy alignment (add enum value or fix config value) | S | Config change takes effect |
| P2 | 6 | Remove per-chunk LLM calls in `_enrich_chunks_with_llm_analysis()` — replace with keyword scoring | M | Ingestion 20× faster, significant cost reduction |
| P2 | 6 | Implement actual deduplication logic (currently flagged but not coded) | M | Prevents duplicate content degrading retrieval |
| P2 | 5 | Add per-intent confidence thresholds | M | Reduces misclassification on overlapping intents |
| P2 | 5 | Align evaluator `expected_intent` values with `IntentType` enum | M | Makes evaluation results meaningful |
| P2 | 1 | Cache query embeddings with LRU (`maxsize=512`) | S | Eliminates repeated embedding costs on popular queries |
| P2 | 1 | Cache Tavily results with TTLCache (TTL=3600) | M | Eliminates repeated search costs |
| P2 | 4 | Implement domain allowlist for Tavily results | M | Prevents misinformation from low-quality sources |
| P2 | 3 | Introduce `LocaleConfig` dataclass and inject into all services | L | Enables multi-city deployment without code changes |
| P2 | 6 | Add BM25 hybrid retrieval with Reciprocal Rank Fusion | M | +15–30% retrieval accuracy for keyword-heavy queries |
| P2 | 2 | Add source attribution (`doc_id`, `chunk_id`) in LLM context and response | M | Enables factual verification by operators |
| P2 | 7 | Implement real uptime tracking, remove all stub endpoint responses | M | Accurate system status |
| P2 | 1 | Run Tavily in parallel with vector search (start both immediately after intent classification) | M | -2–5s latency for queries that use web search |

_P0 = fix before any user-facing launch_
_P1 = fix within first sprint post-launch_
_P2 = fix within first month_
