# SettleBot — Frontend Session Starter Prompt

**Usage:** Paste this prompt at the start of every new frontend coding session before any task prompt is given. Do not modify it between sessions. Do not begin implementation work until confirmation is complete and accurate.

---

```
CONTEXT ALIGNMENT — SettleBot Frontend (Flask/Jinja2)
Read the document below before confirming. Do not touch any code.

DOCUMENT TO READ:
1. docs/ux-ui-analysis.md  (read in full — every section)

RULES FOR THIS SESSION:

DESIGN SYSTEM
- The Astra Pro theme specification in docs/ux-ui-analysis.md is the
  single source of truth for all visual decisions. No color, spacing,
  font size, radius, shadow, or transition value may be defined outside
  the CSS token system established in that specification.
- No raw hex values, pixel values, or numeric literals for visual
  properties are permitted anywhere in CSS or inline style attributes.
  Every such value must reference a --variable token from the Astra Pro
  token block in the specification.
- The minimum font size for any content a user reads is 17px (--font-size-base).
  14px (--font-size-sm) is permitted only for timestamps, captions,
  and system metadata labels. No readable content goes below 17px.
- html { font-size: 17px } must be set in base.css. All rem values
  compute from this. Do not change this value.
- Body and chat message line-height must be 1.8 (--line-height-loose).
  No body content line-height may be set below 1.6.

TYPOGRAPHY
- The font family for all text is Plus Jakarta Sans loaded from Google
  Fonts as a single variable-weight request. Do not load Inter, Roboto,
  or any other font family unless a task prompt explicitly requires it
  alongside Plus Jakarta Sans.
- JetBrains Mono is the only permitted monospace font, used exclusively
  for phone numbers, addresses, currency amounts, and code.
- Do not mix more than two typefaces in any template or component.
- Never add letter-spacing to body or paragraph text.

ACCESSIBILITY
- All interactive elements (buttons, links, inputs, toggles) must be
  proper semantic HTML elements, not styled divs.
- Every form input must have an associated <label> — not just a placeholder.
- Do not set user-scalable=no or maximum-scale=1 in any viewport meta tag.
- aria-live="polite" must be present on the chat message container.
  The typing indicator must use aria-live="assertive".
- All color choices must maintain a minimum 4.5:1 contrast ratio for
  normal text (WCAG 2.2 AA). Test before declaring a component complete.
- The prefers-reduced-motion media query must be honoured. No animation
  or transition may run without a zero-duration fallback under this query.

TRUST AND SAFETY
- The crisis banner must use --color-crisis (#1a6b8a). Under no
  circumstances may --color-error or any red value be used for a
  crisis-level UI state. Crisis signals must calm the user, not alarm them.
- Verified contact details (phone numbers, addresses surfaced from the
  RAG grounding rule) must be rendered in the verified-contact chip
  component, visually distinct from all other text. They must never
  appear as plain inline text.
- The anonymous session status indicator must be present in the chat
  header whenever the user is not authenticated.

MOBILE AND PERFORMANCE
- Every layout decision is made mobile-first. Desktop is an enhancement.
- The chat input bar must be fixed at the bottom of the viewport on mobile.
- No hover-dependent interaction is permitted as the sole means of
  accessing a feature — touch screens have no hover state.
- Do not serve images without srcset and WebP format with JPEG fallback.
- Do not add autoplay video, background video, or looping animation to
  any public-facing page.

GENERAL
- The codebase is the source of truth. docs/ux-ui-analysis.md describes
  intent and constraints. If the existing code contradicts the spec,
  report the contradiction before proceeding — do not silently override
  existing code to match the spec without flagging it.
- Do not read any code files until a task prompt is provided.
- Do not write any code until a task prompt is provided.
- Do not suggest features, refactors, or improvements beyond what a
  task prompt defines.
- Do not use inline styles. All styles belong in the appropriate CSS file.
- Do not add JavaScript for interactions that CSS handles natively.
- All new CSS custom properties must be added to the :root token block
  in base.css — never scoped to a single component file.

CONFIRM BY RESPONDING WITH:
1. The name of the adopted theme, the body font family, and the exact
   base font size in pixels as specified in the Astra Pro section.
2. The value of --color-crisis and the one context where it must be used
   and the one context where it must never be used, exactly as stated
   in the specification.
3. The line height value for body text and the line height value for
   display headings (H1), both as numeric values from the token block.
4. The two container width values: one for general page content and one
   for chat and form content, with their exact pixel values and token names.
5. The first and last items from the Astra Pro Compliance Checklist,
   exactly as written.
6. The eight Design Principles from the manifesto — title only,
   in the order they appear.

Do not proceed until this confirmation is complete and accurate.
```
