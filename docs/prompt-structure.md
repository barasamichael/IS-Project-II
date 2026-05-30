# SettleBot — AI Coding Agent Prompt Engineering Standards

**Document reference:** SB-PROMPT-2026-001

**Version:** 1.0

**Date:** May 30, 2026

**Status:** Active — applies to all prompts written for this project

---

## Purpose

This document defines the mandatory structure, content rules, and engineering standards for every prompt written to instruct an AI coding agent working on SettleBot. A prompt that does not follow these rules must be rewritten before it is used. Deviations from this standard are the primary cause of agent drift, hallucination, scope creep, and regressions.

---

## 1. Mandatory prompt sections

Every prompt must contain all of the following sections in this order. No section may be omitted. No section may be merged with another.

---

### 1.1 HEADER

A single line stating the project name, the continuation context, and the governing principle. The governing principle must always be stated as: `quality locked, production rigor first`.

**Format:**
```
HEADER:
SettleBot continuation — [short description of what this prompt does] (quality locked, production rigor first).
```

**Rule:** The header must say exactly what the prompt does in one line. "Fix things" is not acceptable. "Remove duplicate language detection call from api/main.py and verify with targeted test" is acceptable.

---

### 1.2 MILESTONE

States the milestone from SB-MILE-2026-001 this prompt addresses.

**Format:**
```
MILESTONE:
Milestone: [Milestone number and name from SB-MILE-2026-001]
```

**Rule:** Every prompt must map to a specific milestone in SB-MILE-2026-001. A prompt that does not map to a defined milestone must not be executed until the work is captured in the milestone document.

---

### 1.3 SCOPE

A plain language statement of exactly what this prompt implements and nothing more.

**Format:**
```
SCOPE:
Implement only [specific thing].
- [bullet describing what is in scope]
- [bullet describing what is in scope]

This prompt must not implement [explicit exclusion], [explicit exclusion], or [explicit exclusion].
```

**Rules:**
- The scope section must state what is included and what is excluded.
- Exclusions are not optional. Every prompt must explicitly rule out adjacent work the agent might reasonably attempt.
- The scope must be narrow enough to complete in one focused session. If it requires more than a few hours of focused work, split it into two prompts.
- The final sentence of the scope must always begin with "This prompt must not implement".

---

### 1.4 NON-NEGOTIABLE RULES

A list of rules the agent must follow without exception. These are not suggestions. The agent must be told explicitly that these rules override its own judgment.

**Format:**
```
NON-NEGOTIABLE RULES:
- First read and align with:
  - [file path]
  - [file path]
- Start with `git status --short` and stop if unrelated workspace changes are present.
- [rule]
- [rule]
```

**Rules:**
- The first item must always instruct the agent to read the relevant existing files before making any changes. List every file by path. Do not write "read the relevant files".
- The second item must always be the git status check. The agent must stop if unrelated workspace changes are present.
- Every rule must be a complete sentence describing a single constraint. Rules must not be grouped.
- Rules must not begin with "try to", "consider", or "where possible". Rules are absolute.
- LLM-specific rules (grounding constraint, model name from settings, no print() statements) must always be included when the prompt touches `services/response_generator.py`, `services/intent_recognizer.py`, or `services/language_processor.py`.

---

### 1.5 GOAL

One paragraph stating what the completed prompt achieves in human terms. Not what files are changed. What the system can do after the prompt is executed that it could not do before, or what risk is eliminated.

**Format:**
```
GOAL:
[One to three sentences describing the system-level outcome.]
```

**Rule:** The goal must be written in terms of system capability, safety, or correctness. It must not describe code or file changes. Code and file changes belong in other sections.

---

### 1.6 INVARIANT TARGET

A numbered list of specific, testable end states that must be true when the prompt is complete. These are the conditions for closing the work.

**Format:**
```
INVARIANT TARGET:
1. [Specific testable condition]
2. [Specific testable condition]
```

**Rules:**
- Every item must be independently verifiable. "The system works better" is not acceptable. "A POST /query request with a 2001-character query body returns HTTP 422" is acceptable.
- Items must not overlap. Each describes a distinct state.
- Every prompt must have at minimum five invariant targets.

---

### 1.7 EXACT FILES TO TOUCH

An explicit list of every file the agent is permitted to modify. The agent must not modify any file not on this list.

**Format:**
```
EXACT FILES TO TOUCH:
- path/to/file.py       Short plain-language description of why this file is touched.
```

**Rules:**
- Every file must have a one-line description of why it is being modified.
- The list must be exhaustive. If the agent needs to touch a file not on this list, it must stop and report, not improvise.
- Test files must be listed separately.
- `config/config.yaml` and `config/constants.py` must be listed separately if modified.

---

### 1.8 ONLY TOUCH THESE IF STRICTLY REQUIRED

A secondary file list for files that may need modification only if the primary implementation makes it unavoidable.

**Format:**
```
ONLY TOUCH THESE IF STRICTLY REQUIRED BY THE IMPLEMENTATION:
- path/to/file.py
```

**Rules:**
- Files on this list must not be touched unless the primary implementation makes it logically unavoidable.
- The agent must not add files to this list. If it needs a file not on either list, it must stop and report.
- This section must always be present. Write "None." if no files qualify.

---

### 1.9 PLAN

A numbered sequence of steps the agent must follow in order.

**Format:**
```
PLAN (QUALITY-LOCKED):
1. [Step — must describe an action, not an outcome]
2. [Step]
...
N. Stage only intended files and create one scoped commit.
```

**Rules:**
- The first step must always be to read the current state of the relevant files.
- The last step must always be to stage only the intended files and create one commit.
- Every step must be an action, not an outcome.
- Steps must be sequential. The agent must not skip or reorder them.
- Implementation details belong in the Implementation Requirements section, not here.

---

### 1.10 IMPLEMENTATION REQUIREMENTS

The detailed technical specification for what must be built. Organised into labelled parts.

**Format:**
```
IMPLEMENTATION REQUIREMENTS:

PART A — [SHORT LABEL]
- [Requirement]
- [Requirement]

PART B — [SHORT LABEL]
- [Requirement]
```

**Rules:**
- Each part must address one coherent concern.
- Every requirement must describe what the system must do, not how. "The grounding rule must appear before all other content in the system prompt" describes what. "Concatenate the constant string before calling _get_comprehensive_system_prompt" describes how.
- Requirements must not contradict `docs/technical-specification.md`. If a requirement conflicts with the spec, the prompt is wrong.
- Magic numbers and inline string literals in requirements are not permitted. Reference the constant name from `config/constants.py`.
- Requirements must not use "try", "consider", "if possible", "where appropriate", or "as needed". Every requirement is absolute. Conditional requirements must state the condition explicitly.

---

### 1.11 OUT OF SCOPE

An explicit list of things this prompt must not implement even if they seem related.

**Format:**
```
OUT OF SCOPE FOR THIS PROMPT:
- [Thing that is excluded]
```

**Rules:**
- Every prompt must have at least five out-of-scope items.
- Items must be specific. "Unrelated work" is not acceptable. "LocaleConfig architecture (Milestone 8)" is acceptable.
- If an adjacent feature is tempting for an AI agent to add, it must appear on this list.

---

### 1.12 TESTS AND VALIDATION

The specific tests that must be written or run to verify the prompt is complete.

**Format:**
```
TESTS / VALIDATION (REQUIRED):

Positive:
- [What must succeed]

Negative:
- [What must be rejected or fail as expected]

Determinism:
- [What must produce identical results on repeated execution]

Regression:
- [What must not break that was working before]
```

**Rules:**
- All four subsections are mandatory.
- Every item must describe a specific observable behaviour. "The query works" is not acceptable. "POST /query with a valid housing query returns a response containing `## DIRECT ANSWER`" is acceptable.
- Test names from `docs/technical-specification.md` Section 5 must be referenced by name where applicable.
- Regression items must be listed explicitly and verified.

---

### 1.13 VALIDATION (TOKEN-EFFICIENT)

The exact commands the agent must run to verify the implementation before committing.

**For the FastAPI RAG backend:**
```
VALIDATION (TOKEN-EFFICIENT):
- git diff --stat
- git status --short
- ruff check [files]
- ruff format --check [files]
- pytest -q [test files]
- git status --short
```

**For the Flask frontend:**
```
VALIDATION (TOKEN-EFFICIENT):
- git diff --stat
- git status --short
- ruff check [files]
- ruff format --check [files]
- pytest -q [test files]
- git status --short
```

**Rules:**
- Ruff check must run on every Python file touched.
- Ruff format check must run on every Python file touched.
- Pytest must specify the exact test file or test names. Running the full test suite is not permitted in this section.
- The final command must always be `git status --short` to confirm no unintended files were modified.
- These commands must run in the order listed. The agent must not proceed to the commit if any command fails.

---

### 1.14 GIT WORKFLOW

The exact git procedure the agent must follow to commit the work.

**Format:**
```
GIT WORKFLOW:
- Stage only the files listed in EXACT FILES TO TOUCH.
- Run: git diff --cached --name-only
- Verify that only intended files appear.
- Commit using the format defined in docs/technical-specification.md Section 7.2.

Commit message:

type(scope): short description in present tense

Body explaining what changed and why. Reference the milestone.
addresses SB-MILE-2026-001 Milestone N
```

**Rules:**
- The agent must stage files explicitly by name. `git add .` is not permitted.
- The agent must run `git diff --cached --name-only` and verify the list before committing.
- If any unintended file appears in the staged list, the agent must unstage it and report.
- The commit message must have a subject line, a blank line, and a body. All three are required.
- The body must reference the milestone number: `addresses SB-MILE-2026-001 Milestone N`.
- One prompt produces exactly one commit.

---

### 1.15 SUCCESS CRITERIA

The final checklist that defines when the prompt is closed.

**Format:**
```
SUCCESS CRITERIA:
- [Specific observable outcome]
- [Specific observable outcome]
- Focused validation passes with no unrelated file changes and one scoped commit only.
```

**Rules:**
- The last item must always be: "Focused validation passes with no unrelated file changes and one scoped commit only."
- All other items must correspond directly to invariant targets from section 1.6.
- Success criteria must not introduce new requirements.
- If any success criterion is not met, the prompt is not complete.

---

## 2. Cross-cutting rules for all prompts

---

### 2.1 One prompt, one commit

Every prompt produces exactly one commit. If the work logically requires two commits, write two prompts.

---

### 2.2 Read before write

The agent must always read the current state of every relevant file before making any changes. An agent that writes without reading produces hallucinations.

---

### 2.3 The agent does not decide scope

The agent executes what the prompt defines. It does not add features it judges to be useful. It does not refactor code it judges to be untidy unless that refactor is explicitly required. Scope is defined before the agent starts.

---

### 2.4 Fail loudly, not silently

If the agent encounters a condition that prevents it from completing the prompt as written — a missing file, a conflicting implementation, an ambiguous requirement — it must stop and report the problem. It must not improvise and proceed.

---

### 2.5 No self-referential commit metadata

The agent must not add any reference to AI, machine generation, or any automated tool in commit messages, code comments, or docstrings. All output must read as authored by the developer configured in the local git environment.

---

### 2.6 Constants over literals

No prompt may instruct the agent to use inline numeric or string literals for values that belong in `config/constants.py`. References in prompts must use the constant name. The agent must use the constant in code.

---

### 2.7 Coding standards are always in force

The coding standards defined in `docs/technical-specification.md` Section 3.2 apply to every line of code produced by every prompt without exception.

---

### 2.8 LLM grounding rule is always in force

Any prompt that modifies `services/response_generator.py` must verify that the system prompt produced by `_get_comprehensive_system_prompt()` still contains the grounding rule text from `docs/technical-specification.md` Section 3.2.12 as its first instruction block. If the grounding rule is absent or modified, the prompt is incomplete regardless of other success criteria.

---

### 2.9 No hardcoded locale values

Any prompt that modifies services, config, or utilities must not introduce new string literals for city names, country names, currency symbols, neighbourhood names, timezone strings, or institution names. All such values must come from the `LocaleConfig` object or from `config/constants.py`. A prompt that introduces a hardcoded locale value violates `docs/technical-specification.md` Section 3.2.10 and must be rewritten.

---

### 2.10 Prompts are ordered by milestone dependency

Prompts must be written and executed in the milestone order defined by `docs/milestones.md`. A prompt for Milestone 4 must not be executed before all prompts for Milestone 1 are closed.

---

## 3. Prompt quality checklist

Before executing any prompt, verify all of the following. If any item is false, rewrite the prompt.

| Check | Pass condition |
|---|---|
| Header is one line and specific | The header names exactly what is implemented |
| Milestone is mapped | The prompt maps to a milestone in SB-MILE-2026-001 |
| Scope states exclusions | At least three things are explicitly excluded |
| Non-negotiable rules list files by path | No vague file references |
| Git status check is present | Present in Non-Negotiable Rules |
| Read-before-write is enforced | Files to read are listed before implementation begins |
| Invariant targets are testable | Each target can be verified pass or fail |
| Exact files list is exhaustive | No file is missing from the list |
| Out of scope has five or more items | Adjacencies are explicitly ruled out |
| All four test subsections present | Positive, Negative, Determinism, Regression |
| Validation commands are exact | No vague instructions like "run the tests" |
| Commit message has subject and body | Both parts are drafted in the prompt |
| No AI self-reference in commit | Commit body contains no AI mention |
| One commit only | The prompt does not produce multiple commits |
| Success criteria match invariants | No new requirements appear in success criteria |
| Grounding rule verified | If response_generator.py is touched, grounding rule check is in invariants |
| No locale literals introduced | No city/country/currency strings appear in requirements |

---

## 4. Prompt samples

These samples are the reference standard for all prompts written on this project. All other prompts follow one of these three patterns. When in doubt about format or tone, compare against these samples.

---

### Sample A — Single-file fix (simple)

This sample shows the pattern for a focused, single-concern fix where the change is small and the risk is high if done incorrectly. Use this for P0 and P1 items from `docs/feedback.md`.

```
HEADER:
SettleBot continuation — remove duplicate language detection call from api/main.py
(quality locked, production rigor first).

MILESTONE:
Milestone: 1 — Immediate Security and Safety

SCOPE:
Remove only the duplicate language detection block in api/main.py.
- Delete lines 578–589 in api/main.py that call language_processor.detect_and_process_query()
  before the response generator is invoked
- Update the query handling to use only the detection result from inside generate_response()
- Add a targeted test confirming exactly one language detection call occurs per query

This prompt must not implement rate limiting, the grounding rule, the evaluator
fix, or any other Milestone 1 items.

NON-NEGOTIABLE RULES:
- First read and align with:
  - api/main.py
  - services/response_generator.py
  - services/language_processor.py
- Start with `git status --short` and stop if unrelated workspace changes are present.
- Reproduce the exact current block at api/main.py lines 577–589 before removing it
  to confirm the correct lines are targeted.
- Do not modify the logic inside response_generator.generate_response(). The internal
  language detection call at line 660 is the one that stays.
- Do not introduce any new logic to replace the removed block. The removal itself is
  the complete change.
- The QueryResponse returned to the caller must be identical in structure before and
  after this change.
- All functions in touched files must retain their existing docstrings unchanged.
- ruff check and ruff format --check must pass on all touched files before committing.

GOAL:
Each call to POST /query triggers exactly one gpt-4.1-nano language detection
call rather than two. This eliminates 1–2 seconds of redundant latency and halves
the language detection API cost on every query.

INVARIANT TARGET:
1. A mock-instrumented POST /query call to the test client triggers
   language_processor.detect_and_process_query() exactly once, not twice.
2. The QueryResponse returned from POST /query contains the correct
   language_info fields populated from the single detection call.
3. api/main.py no longer contains a call to
   language_processor.detect_and_process_query() at or around line 578.
4. The variables english_query and language_result used downstream in the
   process_query() function reference the values produced by the response
   generator's internal detection, not a locally produced copy.
5. A POST /query with a Swahili input still returns a translated English
   response, confirming the detection path still functions correctly after
   the removal.

EXACT FILES TO TOUCH:
- api/main.py                          Remove the duplicate detection block.
- tests/test_api_query.py              Add test_single_language_detection_per_query.

ONLY TOUCH THESE IF STRICTLY REQUIRED BY THE IMPLEMENTATION:
- None.

PLAN (QUALITY-LOCKED):
1. Read api/main.py lines 560–660 and services/response_generator.py lines
   646–690 to understand exactly which variables the removed block populates
   and how those variables are used after line 589.
2. Confirm that every variable produced by the block at lines 578–589
   (english_query, language_result) is also produced inside
   response_generator.generate_response() and returned in the response data.
3. Delete the duplicate block at api/main.py lines 577–589.
4. Adjust any variable references in process_query() that pointed to the
   now-removed block, replacing them with values from the response_data dict.
5. Write test_single_language_detection_per_query in tests/test_api_query.py
   using a mocked language_processor to assert call count equals 1.
6. Run ruff check and ruff format --check on both touched files.
7. Run pytest -q tests/test_api_query.py::test_single_language_detection_per_query.
8. Stage only the two intended files and create one scoped commit.

IMPLEMENTATION REQUIREMENTS:

PART A — REMOVAL
- The block starting at api/main.py line 577 (the `if request.language_detection:`
  branch that calls language_processor.detect_and_process_query()) must be deleted
  in its entirety, including the `else` branch that constructs a fallback dict.
- The variables english_query and language_result must no longer be set in
  process_query(). They must not be referenced anywhere in process_query() after
  the removal. All information they carried is available from response_data returned
  by generate_response().

PART B — DOWNSTREAM ADJUSTMENT
- The LanguageInfo construction at api/main.py lines 612–619 currently reads from
  language_result. After removal, it must read from the language_info dict inside
  response_data, which response_generator.generate_response() already returns.
- The QueryResponse construction must produce identical output to the current
  implementation. No field value may change.

PART C — TEST
- test_single_language_detection_per_query must patch
  services.language_processor.LanguageProcessor.detect_and_process_query with
  a MagicMock and assert mock.call_count == 1 after one POST /query request.
- The test must use a valid English query so the off-topic path is not triggered.
- The test must assert the response status code is 200 and the response body
  contains the response field.

OUT OF SCOPE FOR THIS PROMPT:
- Rate limiting (Milestone 2)
- Query length validation (Milestone 2)
- Grounding rule addition (Milestone 1, separate prompt)
- Evaluator method name fix (Milestone 1, separate prompt)
- API key removal from config (Milestone 1, separate prompt)
- Any changes to services/language_processor.py or services/response_generator.py

TESTS / VALIDATION (REQUIRED):

Positive:
- test_single_language_detection_per_query: one POST /query call triggers
  detect_and_process_query exactly once.
- POST /query with a valid English query returns HTTP 200 with language_info.detected_language
  set to "english".
- POST /query with a Swahili query returns HTTP 200 with language_info.translation_needed
  set to True.

Negative:
- POST /query without the Authorization header returns HTTP 401 (unchanged from before).
- POST /query with an off-topic query returns the standard off-topic response (unchanged).

Determinism:
- Running the test suite twice produces the same pass/fail result for
  test_single_language_detection_per_query.

Regression:
- All previously passing tests in tests/test_api_query.py continue to pass.
- The QueryResponse schema returned by POST /query is structurally unchanged.
- services/response_generator.py is unmodified.

VALIDATION (TOKEN-EFFICIENT):
- git diff --stat
- git status --short
- ruff check api/main.py tests/test_api_query.py
- ruff format --check api/main.py tests/test_api_query.py
- pytest -q tests/test_api_query.py
- git status --short

GIT WORKFLOW:
- Stage only api/main.py and tests/test_api_query.py.
- Run: git diff --cached --name-only
- Verify only these two files appear.
- Commit using the format defined in docs/technical-specification.md Section 7.2.

Commit message:

fix(query-pipeline): remove duplicate language detection call per request

Language detection was executing twice per query: once in api/main.py
lines 577–589 and again inside response_generator.generate_response().
The first call was redundant — its output was discarded because the
response generator reran the same detection internally. Removing the
duplicate saves 1–2 seconds of latency and eliminates one gpt-4.1-nano
API call per query.

addresses SB-MILE-2026-001 Milestone 1

SUCCESS CRITERIA:
- api/main.py contains no call to detect_and_process_query() outside of
  the response generator.
- test_single_language_detection_per_query passes, confirming exactly one
  detection call per request.
- QueryResponse structure and field values are unchanged from before.
- Focused validation passes with no unrelated file changes and one scoped
  commit only.
```

---

### Sample B — Multi-file service change (moderate complexity)

This sample shows the pattern for a fix that touches multiple service files and introduces a new utility module. Use this for Milestone 4-type grounding and accuracy work.

```
HEADER:
SettleBot continuation — add grounding rule to system prompt and implement
post-generation phone number audit (quality locked, production rigor first).

MILESTONE:
Milestone: 4 — Factual Grounding and Hallucination Prevention

SCOPE:
Implement only the grounding rule and phone number audit for response generation.
- Add the grounding rule as the first block of every system prompt produced by
  _get_comprehensive_system_prompt() in services/response_generator.py
- Remove the three hallucination-instructing prompt phrases that tell the LLM
  to provide contact numbers regardless of context
- Create utilities/factcheck.py with extract_phones_from_context() and
  normalise_phone()
- Add the post-generation phone audit that strips unverified phone numbers
  from every response before it is returned

This prompt must not implement the LocaleFactStore, the /admin/update-facts
endpoint, source citation in chunks, the ## SOURCES section, or the
phone_hallucination_rate evaluator metric.

NON-NEGOTIABLE RULES:
- First read and align with:
  - services/response_generator.py
  - config/constants.py
  - docs/technical-specification.md Section 3.2.12
- Start with `git status --short` and stop if unrelated workspace changes are present.
- The grounding rule text must be reproduced verbatim from docs/technical-specification.md
  Section 3.2.12. Do not paraphrase it.
- The grounding rule must be the very first content in every system prompt string
  returned by _get_comprehensive_system_prompt(). No intent-specific or
  locale-specific content may precede it.
- Do not modify any method signature in response_generator.py. The public API
  of ResponseGenerator must be unchanged.
- The post-generation audit must run on every non-off-topic response before it
  is returned from generate_response(). It must not run on off-topic responses.
- Do not add print() statements. All instrumentation must use logger.debug().
- No inline string literals for the grounding rule text. Define it as
  GROUNDING_RULE in config/constants.py and import it.
- ruff check and ruff format --check must pass on all touched files before committing.

GOAL:
The LLM is instructed at the top of every prompt that it may not state a phone
number, address, fee, or operating hours unless it appears verbatim in the
provided context. Any phone number that the LLM fabricates and that slips through
the instruction is caught by the post-generation audit and replaced with a safe
fallback string before the response reaches the user.

INVARIANT TARGET:
1. The string returned by _get_comprehensive_system_prompt() for any intent
   starts with the GROUNDING_RULE constant from config/constants.py.
2. The three phrases "Include specific ... contact numbers", "Include specific
   contacts, websites, and resources", and "Provide specific contacts, websites,
   and locations" no longer appear anywhere in _get_comprehensive_system_prompt().
3. utilities/factcheck.py exists and exports extract_phones_from_context() and
   normalise_phone().
4. extract_phones_from_context() returns the set of normalised phone numbers
   present in the fact store essential_info dict and in the retrieved chunk texts.
5. When generate_response() returns a response containing a phone number not in
   extract_phones_from_context(), that number is replaced with
   "[contact details unavailable — verify at official source]".
6. When generate_response() returns a response containing only verified phone
   numbers, those numbers are preserved unchanged.
7. test_grounding_rule_in_system_prompt passes: the system prompt for every
   IntentType starts with the GROUNDING_RULE text.
8. test_unverified_phone_replaced passes: a mock LLM response containing
   "+254 700 000 000" (not in essential_info) has that number replaced.
9. test_verified_phone_preserved passes: a mock LLM response containing
   "+254 20 2845000" (The Nairobi Hospital, present in essential_info) retains
   that number unchanged.

EXACT FILES TO TOUCH:
- config/constants.py                  Add GROUNDING_RULE and PHONE_RE constants.
- services/response_generator.py       Add grounding rule to system prompt,
                                       remove hallucination-instructing phrases,
                                       add post-generation audit call.
- utilities/factcheck.py               New module: extract_phones_from_context
                                       and normalise_phone.
- tests/test_response_generator.py     Add three targeted tests.

ONLY TOUCH THESE IF STRICTLY REQUIRED BY THE IMPLEMENTATION:
- None.

PLAN (QUALITY-LOCKED):
1. Read services/response_generator.py lines 1120–1230 to locate every
   occurrence of the hallucination-instructing phrases and map the full
   structure of _get_comprehensive_system_prompt().
2. Read config/constants.py to understand existing constant definitions
   and formatting conventions.
3. Add GROUNDING_RULE and PHONE_RE to config/constants.py.
4. Create utilities/factcheck.py with extract_phones_from_context() and
   normalise_phone().
5. Modify _get_comprehensive_system_prompt() to prepend GROUNDING_RULE
   and remove the three hallucination-instructing phrases.
6. Add the post-generation phone audit in generate_response() immediately
   after _apply_final_validation_and_safety() and before translation.
7. Write the three tests in tests/test_response_generator.py.
8. Run ruff check and ruff format --check on all four touched files.
9. Run pytest -q tests/test_response_generator.py -k "grounding or phone".
10. Stage only the four intended files and create one scoped commit.

IMPLEMENTATION REQUIREMENTS:

PART A — CONSTANTS
- GROUNDING_RULE must be a module-level string constant in config/constants.py
  containing the exact grounding rule text from docs/technical-specification.md
  Section 3.2.12, reproduced verbatim.
- PHONE_RE must be a compiled re.Pattern constant:
  re.compile(r'(\+254[\s\-]?\d[\d\s\-]{7,}|\b0[17]\d{2}[\s\-]?\d{3}[\s\-]?\d{3}\b)').

PART B — FACTCHECK UTILITY
- utilities/factcheck.py must export extract_phones_from_context(context_text: str,
  essential_info: dict) -> set[str] and normalise_phone(phone: str) -> str.
- extract_phones_from_context must apply PHONE_RE to both context_text and to
  the JSON-serialised emergency_numbers sub-dict from essential_info to build the
  set of verified phone strings.
- normalise_phone must strip spaces, hyphens, parentheses, and leading zeros to
  produce a canonical form used for set membership comparison.
- Both functions must have complete docstrings per docs/technical-specification.md
  Section 3.2.4.

PART C — SYSTEM PROMPT MODIFICATION
- _get_comprehensive_system_prompt() must prepend GROUNDING_RULE + "\n\n" to the
  base_prompt string before any other content is appended.
- The following three phrases must be removed from base_prompt and from all
  intent_specific entries in the method:
  1. "Include specific Nairobi details - locations, costs in KSh, contact numbers"
  2. "Include specific contacts, websites, and resources"
  3. "Provide specific contacts, websites, and locations"
- These phrases must be replaced with: "Include only specific details and contact
  information that appear verbatim in the RETRIEVED SETTLEMENT INFORMATION or
  ESSENTIAL SETTLEMENT INFORMATION sections provided."

PART D — POST-GENERATION AUDIT
- In generate_response(), immediately after the call to
  _apply_final_validation_and_safety() returns validated_response and before the
  translation block, add a call to the phone audit.
- The audit must: apply PHONE_RE to validated_response to extract all phone
  numbers present, call extract_phones_from_context() with the enhanced_context
  string and self.essential_info, normalise both sets, compute the difference
  (numbers in response but not in verified set), and replace each unverified
  number with "[contact details unavailable — verify at official source]".
- The audit must not modify responses for off-topic queries. The off-topic path
  returns before this point and must not be affected.
- If no unverified numbers are found, the response must be returned unchanged.

OUT OF SCOPE FOR THIS PROMPT:
- LocaleFactStore and locale JSON files (Milestone 8)
- /admin/update-facts endpoint (Milestone 4, separate prompt)
- ## SOURCES section and doc_id citation in chunks (Milestone 4, separate prompt)
- phone_hallucination_rate evaluator metric (Milestone 9)
- Removing hardcoded essential_info from response_generator.py (depends on
  LocaleFactStore from Milestone 8)
- Any changes to the Flask frontend

TESTS / VALIDATION (REQUIRED):

Positive:
- test_grounding_rule_in_system_prompt: for every value in IntentType, the
  string returned by _get_comprehensive_system_prompt() starts with GROUNDING_RULE.
- test_verified_phone_preserved: a generated response containing "+254 20 2845000"
  (Nairobi Hospital, present in essential_info) is returned with that number intact.
- extract_phones_from_context returns correct phone numbers from a fixture context.
- normalise_phone strips spaces and hyphens to produce a canonical form.

Negative:
- test_unverified_phone_replaced: a mock LLM response containing "+254 700 000 000"
  (not in essential_info) has that number replaced with the safe fallback string.
- A response with no phone numbers passes through the audit unchanged.

Determinism:
- Running the audit on the same response text twice produces the same result.

Regression:
- All previously passing tests in tests/test_response_generator.py continue to pass.
- The off-topic response path is unchanged and the audit does not run on it.
- ResponseGenerator's public method signatures are unchanged.

VALIDATION (TOKEN-EFFICIENT):
- git diff --stat
- git status --short
- ruff check config/constants.py services/response_generator.py
  utilities/factcheck.py tests/test_response_generator.py
- ruff format --check config/constants.py services/response_generator.py
  utilities/factcheck.py tests/test_response_generator.py
- pytest -q tests/test_response_generator.py
- git status --short

GIT WORKFLOW:
- Stage only config/constants.py, services/response_generator.py,
  utilities/factcheck.py, and tests/test_response_generator.py.
- Run: git diff --cached --name-only
- Verify only these four files appear.

Commit message:

fix(response-gen): add grounding rule to system prompt and audit fabricated phones

Adds GROUNDING_RULE constant to config/constants.py and prepends it to every
system prompt produced by _get_comprehensive_system_prompt(). Removes the three
prompt phrases that instructed the LLM to produce contact numbers regardless of
context. Adds utilities/factcheck.py with extract_phones_from_context() and
normalise_phone(). A post-generation audit now replaces any phone number in the
response that cannot be verified against the essential_info dict or retrieved
chunks. The fabricated +254 700 000 000 numbers seen in responses.txt will no
longer reach users.

addresses SB-MILE-2026-001 Milestone 4

SUCCESS CRITERIA:
- Every system prompt starts with the verbatim GROUNDING_RULE text.
- The three hallucination-instructing phrases are gone from all system prompts.
- Fabricated phone numbers in LLM output are replaced before the response is returned.
- Verified phone numbers in LLM output are preserved unchanged.
- Focused validation passes with no unrelated file changes and one scoped commit only.
```

---

### Sample C — Flask frontend fix

This sample shows the pattern for Flask frontend work. The structure is identical. Validation commands differ from the RAG backend.

```
HEADER:
SettleBot continuation — wire chat interface to RAG API with correct error
handling and Markdown rendering (quality locked, production rigor first).

MILESTONE:
Milestone: 10 — Frontend Chat Interface Completion

SCOPE:
Implement only the RAG API integration in the anonymous chat view.
- Read RAG_API_URL and RAG_API_KEY from Flask app config in the view that
  handles chat submissions
- Replace any hardcoded localhost URL with the config variable
- Render the three-section response structure with correct visual hierarchy
- Display a user-facing error message for HTTP 429, 401, 422, and network errors

This prompt must not implement the session conversation context, the language
detection badge, the ## SOURCES collapsible section, or the Flask-Login
authentication changes.

NON-NEGOTIABLE RULES:
- First read and align with:
  - front-end/app/anonymous/views.py
  - front-end/app/templates/anonymous/chat.html
  - front-end/config.py
  - front-end/app/__init__.py
- Start with `git status --short` and stop if unrelated workspace changes are present.
- The RAG API URL must be read from flask.current_app.config["RAG_API_URL"].
  It must not be hardcoded anywhere in a view function or template.
- The RAG API key must be read from flask.current_app.config["RAG_API_KEY"].
  It must not appear in any template or JavaScript file.
- No business logic may appear in view functions. Logic belongs in helper
  functions called from the view.
- Every inline style attribute in the template must be replaced with a CSS
  class from the project stylesheet unless the value is dynamically computed.
- ruff check and ruff format --check must pass on all Python files touched.

GOAL:
A user submitting a query through the chat interface receives a response from
the RAG API rendered with correct section formatting. If the API is unavailable
or the request is rejected, the user sees a plain-language message rather than
a blank page or raw JSON.

INVARIANT TARGET:
1. The view function reads RAG_API_URL from app config and constructs the
   request URL from it. No string literal URL appears in views.py.
2. The Authorization header sent to the RAG API uses the value of
   RAG_API_KEY from app config, never a hardcoded string.
3. The rendered chat.html template displays ## DIRECT ANSWER,
   ## ADDITIONAL INFORMATION, and ## NEXT STEPS as visually distinct
   section headers, not raw Markdown.
4. An HTTP 429 response from the RAG API renders the message: "Too many
   requests — please wait a moment before asking again."
5. An HTTP 401 response from the RAG API renders the message: "The
   assistant is temporarily unavailable."
6. An HTTP 422 response from the RAG API renders the message: "Please
   enter a question between 3 and 2000 characters."
7. A network error (connection refused or timeout) renders the message:
   "Could not reach the assistant — please check your connection."
8. front-end/config.py reads RAG_API_URL and RAG_API_KEY from environment
   variables with no hardcoded fallback values for either.

EXACT FILES TO TOUCH:
- front-end/app/anonymous/views.py        Add RAG API call with error handling.
- front-end/app/templates/anonymous/chat.html   Add section rendering and error display.
- front-end/config.py                     Add RAG_API_URL and RAG_API_KEY config fields.
- tests/test_chat_view.py                 Add targeted tests for API integration.

ONLY TOUCH THESE IF STRICTLY REQUIRED BY THE IMPLEMENTATION:
- front-end/app/__init__.py    Only if a config initialisation step is missing.

PLAN (QUALITY-LOCKED):
1. Read all four files listed in Non-Negotiable Rules fully before making
   any changes.
2. Add RAG_API_URL and RAG_API_KEY to front-end/config.py, reading both
   from os.environ with no fallback.
3. Implement a helper function in views.py that calls the RAG API, handles
   error responses, and returns a structured result dict.
4. Update the chat view function to call the helper and pass the result to
   the template.
5. Update chat.html to render the three sections with correct HTML structure
   and the error message when present.
6. Write tests in tests/test_chat_view.py covering the positive and error cases.
7. Run ruff check and ruff format --check on all Python files touched.
8. Run pytest -q tests/test_chat_view.py.
9. Stage only the four intended files and create one scoped commit.

IMPLEMENTATION REQUIREMENTS:

PART A — CONFIG
- front-end/config.py must add RAG_API_URL = os.environ.get("RAG_API_URL") and
  RAG_API_KEY = os.environ.get("RAG_API_KEY") to the Config base class.
- Neither field may have a default fallback value. A missing environment variable
  returns None, and the view must handle None by rendering the unavailability message.

PART B — VIEW HELPER
- A helper function call_rag_api(query: str) -> dict must be added to views.py.
- It must construct the request URL from flask.current_app.config["RAG_API_URL"].
- It must set the Authorization header to f"Bearer {flask.current_app.config['RAG_API_KEY']}".
- It must POST to the /query endpoint with {"query": query, "include_context": False}.
- It must handle HTTPError for status codes 429, 401, and 422, returning a dict
  with keys: success (bool), response (str or None), error_message (str or None).
- It must handle requests.exceptions.ConnectionError and requests.exceptions.Timeout,
  returning success=False with the network error message.
- It must have a timeout of 30 seconds on the requests.post() call.
- It must have a complete docstring per docs/technical-specification.md Section 3.2.4.

PART C — TEMPLATE
- chat.html must check for an error_message in the template context and render
  it in a visible error div if present.
- chat.html must render the response text with section headers converted to
  visual HTML: ## DIRECT ANSWER renders as an <h3> element, ## ADDITIONAL
  INFORMATION renders as an <h3> element, ## NEXT STEPS renders as an <h3>
  element. Bold text wrapped in ** renders as <strong>. This conversion must
  be performed in the view before passing text to the template, not with
  JavaScript in the template.

OUT OF SCOPE FOR THIS PROMPT:
- Session conversation context (Milestone 10, separate prompt)
- Language detection badge (Milestone 10, separate prompt)
- ## SOURCES collapsible section (depends on Milestone 4 doc_id attribution)
- Flask-Login auth changes
- SECRET_KEY production assertion (Milestone 10, separate prompt)
- Any changes to the RAG API backend

TESTS / VALIDATION (REQUIRED):

Positive:
- call_rag_api with a mocked 200 response returns success=True and the
  response text.
- The chat template renders ## DIRECT ANSWER as an <h3> tag when response
  text is passed.

Negative:
- call_rag_api with a mocked 429 response returns success=False with the
  correct error message.
- call_rag_api with a mocked 401 response returns success=False with the
  correct error message.
- call_rag_api with a ConnectionError returns success=False with the network
  error message.
- The chat template renders the error div when error_message is present in context.

Determinism:
- call_rag_api with the same mock response produces the same result dict on
  repeated calls.

Regression:
- All previously passing tests in tests/ continue to pass.
- front-end/config.py base class fields added before this prompt are unchanged.

VALIDATION (TOKEN-EFFICIENT):
- git diff --stat
- git status --short
- ruff check front-end/app/anonymous/views.py front-end/config.py
  tests/test_chat_view.py
- ruff format --check front-end/app/anonymous/views.py front-end/config.py
  tests/test_chat_view.py
- pytest -q tests/test_chat_view.py
- git status --short

GIT WORKFLOW:
- Stage only front-end/app/anonymous/views.py,
  front-end/app/templates/anonymous/chat.html, front-end/config.py, and
  tests/test_chat_view.py.
- Run: git diff --cached --name-only
- Verify only these four files appear.

Commit message:

feat(frontend): wire chat interface to RAG API with section rendering and error handling

Adds RAG_API_URL and RAG_API_KEY config fields read from environment variables.
Implements call_rag_api() helper that handles HTTP 429, 401, 422, and network
errors with plain-language user messages. The chat template now renders the
three-section response structure as HTML headings. Hardcoded localhost URL is
removed. API key no longer appears in any template or JavaScript context.

addresses SB-MILE-2026-001 Milestone 10

SUCCESS CRITERIA:
- RAG_API_URL is read from app config in the view. No hardcoded URL remains.
- All four error conditions display a plain-language message to the user.
- Section headers in responses render as HTML headings, not raw Markdown.
- Focused validation passes with no unrelated file changes and one scoped commit only.
```

---

*Document reference: SB-PROMPT-2026-001 · Version 1.0 · SettleBot Project · May 2026*
