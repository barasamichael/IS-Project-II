# SettleBot — Session Starter Prompt

**Usage:** Paste this prompt at the start of every new coding session before any task prompt is given. Do not modify it between sessions. Do not begin implementation work until confirmation is complete and accurate.

---

```
CONTEXT ALIGNMENT — SettleBot RAG Settlement Assistant
Read all four documents before confirming. Do not touch any code.

DOCUMENTS TO READ (in this order):
1. docs/technical-specification.md
2. docs/feedback.md
3. docs/milestones.md
4. docs/prompt-structure.md

RULES FOR THIS SESSION:
- The codebase is the source of truth. Documents describe intent.
  If at any point during a task the code and a document contradict
  each other, stop and report the contradiction before proceeding.
- Do not read any code files until a task prompt is provided.
- Do not write any code until a task prompt is provided.
- Do not suggest features, refactors, or improvements beyond what
  a task prompt defines.
- Do not assume any milestone is complete unless a task prompt
  confirms it or the code you are instructed to read confirms it.
- All coding standards from docs/technical-specification.md Section 3.2
  are in force for every line of code written in this session without
  exception. This includes Section 3.2.2 (no print() statements),
  Section 3.2.10 (no hardcoded locale values), Section 3.2.12
  (grounding rule in every system prompt), and Section 3.2.13
  (embedding model from settings, never hardcoded).
- All commit messages must follow docs/technical-specification.md
  Section 7.2 exactly: a subject line, a blank line, and a body.
  The body must reference the milestone number.
- Do not reference yourself, your name, or AI involvement in any
  commit message, comment, or docstring.

CONFIRM BY RESPONDING WITH:
1. The number of milestones in docs/milestones.md and the names of
   the first and last milestone exactly as written.
2. The six P0 items from the action plan in docs/feedback.md in the
   order they appear in the table.
3. The three import groups and the ordering rule within each group
   from docs/technical-specification.md Section 3.2.3.
4. The names of all fifteen mandatory prompt sections in the order
   they must appear from docs/prompt-structure.md Section 1.
5. The full grounding rule text from docs/technical-specification.md
   Section 3.2.12 exactly as written, not paraphrased.

Do not proceed until this confirmation is complete and accurate.
```
