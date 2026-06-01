# SettleBot UX/UI Analysis
## Human-Centered Design Brief — Production Grade

**Prepared for:** SettleBot Development Team
**Date:** June 2026
**Scope:** Full UX/UI analysis across 10 design dimensions + Astra Pro theme specification + page architecture analysis, priority matrix, and design principles manifesto

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dimension 1 — User Psychology & Emotional Design](#dimension-1--user-psychology--emotional-design)
3. [Dimension 2 — Color Theory & Visual Language](#dimension-2--color-theory--visual-language)
4. [Dimension 3 — Typography & Readability](#dimension-3--typography--readability)
5. [Dimension 4 — Information Architecture & Flow](#dimension-4--information-architecture--flow)
6. [Dimension 5 — Device & Context of Use](#dimension-5--device--context-of-use)
7. [Dimension 6 — Conversational UI Patterns](#dimension-6--conversational-ui-patterns)
8. [Dimension 7 — Accessibility & Inclusion](#dimension-7--accessibility--inclusion)
9. [Dimension 8 — Trust, Transparency & Safety Signals](#dimension-8--trust-transparency--safety-signals)
10. [Dimension 9 — Competitive & Domain Benchmarking](#dimension-9--competitive--domain-benchmarking)
11. [Dimension 10 — Design System Recommendations](#dimension-10--design-system-recommendations)
12. [Theme Specification — Astra Pro](#theme-specification--astra-pro)
13. [Page Architecture & Information Hierarchy](#page-architecture--information-hierarchy)
14. [Priority Matrix](#priority-matrix)
15. [Design Principles Manifesto](#design-principles-manifesto)

---

## Executive Summary

SettleBot's most critical design insight is this: **the user arrives in a state of diminished capacity** — cognitively overloaded, often anxious, frequently operating in their second or third language, on a low-end device with unstable connectivity, in an unfamiliar country. Every design decision must be evaluated against that baseline, not against a comfortable user sitting at a desktop in a familiar environment.

The single biggest risk is treating this as a standard SaaS product and designing for confidence and exploration. SettleBot's users need **clarity, safety, and speed to first answer** — not feature discovery. If the interface demands cognitive effort before delivering value, users in distress will abandon it at the exact moment they need it most.

The grounding rule that prevents hallucinated contact details is not just a backend safety measure — it must be visually honoured in the UI, because a wrong phone number in a crisis is not a UX failure, it is a human failure.

This document defines the design constraints, principles, and standards that ensure SettleBot feels like it was built specifically for a 22-year-old student from Cameroon who just arrived at JKIA with a dead phone battery, a confusing landlord situation, and no one to call.

---

## Dimension 1 — User Psychology & Emotional Design

### Governing Principles

- **Cognitive Load Theory (Sweller, 1988):** Working memory is severely limited under stress. Anxious users cannot parse complex interfaces — every unnecessary element competes for processing bandwidth they do not have.
- **Emotional Contagion in UI (Brave & Nass, 2000):** Interfaces that mirror warm, human language reduce perceived threat. Cold, clinical, or mechanical tone actively increases user anxiety.
- **Fogg Behavior Model:** Behavior = Motivation × Ability × Prompt. Users in distress have high motivation but degraded ability. The interface must reduce friction to near zero — no friction at the moment of need is tolerated.
- **Hierarchy of Needs applied to UX (Maslow → Aarron Walter):** Before delight, the system must deliver functionality, reliability, and usability. For SettleBot, safety and belonging needs are literally on the table — not metaphors.

### Applied to SettleBot

The emotional journey has three distinct phases that must be designed separately:

**Phase 1 — Arrival (0–10 seconds)**
The user lands and asks: *"Is this for me? Can I trust this? Will it understand my problem?"* The homepage must answer all three questions visually before a word is read. A student from Ethiopia should see something that says "I understand your world" — not a generic SaaS hero banner.

**Phase 2 — First Question (10s–2min)**
The user types their first message. This is the highest-anxiety moment. The system must acknowledge receipt instantly (typing indicator within 300ms), respond with a warm opening line before the answer, and never return an empty or error state silently.

**Phase 3 — Post-Answer (ongoing)**
The user evaluates the answer. Was it helpful? Does it feel honest? The UI must support follow-up questions naturally, show that the conversation is remembered, and never make the user feel they've hit a dead end.

**Crisis State Design**
When the system detects high emotional distress or crisis, the UI must shift — not just the words, but the visual environment. Reduce visual noise, surface the emergency resource block prominently, and slow the pacing. Never gamify or rush a user in crisis.

### State of the Art

- **Crisis Text Line:** Opens with "What's going on?" — four words. No menus, no forms. Immediate conversational entry. The entire UI is stripped to the conversation.
- **Pi.ai (Inflection):** Leads with warmth, never capability. Establishes relationship before function.
- **Wysa (mental health chatbot):** Uses "emotional check-in" as an entry point before any task — legitimizes the emotional state before problem-solving.
- **UNHCR Signpost:** Designed for displaced persons — uses extremely simple language, large tap targets, and zero assumed literacy about digital interfaces.

### Dos

- Open with a warm, human-written welcome message that names the user's situation: *"Navigating a new city is hard. Ask me anything — I'm here to help you find your footing in Nairobi."*
- Use first-person language from the bot: "I found..." not "The system retrieved..."
- Show a visible "thinking" state during AI response generation — never silent loading
- Offer quick-tap starter prompts for users who don't know where to begin: "Find housing", "I need help with my visa", "Is my area safe?"
- When crisis is detected, immediately render a calm, full-width compassion block before the answer
- Let users rate helpfulness after each response (thumbs up/down) — it signals that their opinion matters

### Don'ts

- Don't open with a registration wall — a user in crisis will not create an account to get help
- Don't use jargon: no "RAG pipeline", no "intent classified as...", no "confidence: 0.87"
- Don't show a generic 404 or "no results found" without an alternative path
- Don't place any marketing language near the chat interface
- Don't use modal interruptions during an active conversation
- Don't use a robotic error tone: never "An error occurred. Code: 500" — use "I couldn't reach my knowledge base just now — please try again in a moment."

### Critical Failure Mode

**Designing the chat interface as if the user is calm and browsing.** If the default state assumes a comfortable, exploratory user, the entire information architecture will be wrong — too many options, too much friction, too much reading required before getting an answer. This single assumption error cascades into every other design decision.

---

## Dimension 2 — Color Theory & Visual Language

### Governing Principles

- **Itten's Color Theory + Emotional Color Psychology:** Colors carry emotional valence. Blue → trust, calm, authority. Green → safety, health, progress. Red → urgency, danger, stop. Yellow/Orange → caution, warmth, energy. These are not universal — they intersect with cultural coding.
- **Cultural Color Semiotics (Aslam, 2006):** White means purity in Western contexts but mourning in East Asian and some African cultures. Green carries sacred meaning in many Muslim-majority countries. Red means luck in China, danger in the West, and mourning in South Africa.
- **WCAG 2.2 Contrast Standards:** Minimum 4.5:1 for normal text, 3:1 for large text. Color alone must never be the sole means of conveying information.
- **Simultaneous Contrast (Albers):** Colors appear different depending on their surroundings. A color system must be designed holistically — not color-by-color.

### Applied to SettleBot

SettleBot's user base spans sub-Saharan Africa, East Africa, the Middle East, South/Southeast Asia, and Europe. No single color carries universal positive meaning across all of these. The solution is **anchoring the palette in universally safe, warm-neutral tones** and using accent colors functionally (not decoratively), with careful avoidance of culturally loaded combinations.

**Recommended Palette:**

| Role | Color | Hex | Rationale |
|---|---|---|---|
| Primary | Deep teal-blue | `#1A5276` | Trust, calm, professionalism — avoids pure blue (cold) and pure green (religious sensitivity) |
| Accent | Warm amber | `#E67E22` | Warmth, approachability — used sparingly for CTAs |
| Success | Muted green | `#27AE60` | Always paired with icon, never color alone |
| Warning | Amber | `#F39C12` | Confidence caveats, unverified information |
| Crisis | Deep calm teal | `#1A6B8A` | Reserved exclusively for crisis banners — calm, NOT red |
| Error | Deep crimson | `#922B21` | Form validation only — never decorative |
| Surface | Off-white | `#FAFAF8` | Dominant canvas — less clinical than pure white |
| Surface raised | White | `#FFFFFF` | Bot chat bubbles, cards |
| Border | Warm grey | `#E8E4E0` | Subtle, warm separation |
| Text primary | Near black | `#2C2C2C` | Warm black, not pure #000000 |
| Text secondary | Mid grey | `#6B6B6B` | Metadata, secondary labels |
| Bubble (user) | Warm sand | `#F0EAE0` | Distinct from bot without being harsh |
| Bubble (bot) | White | `#FFFFFF` | Clean, readable, trustworthy |

### State of the Art

- **Duolingo:** Warm green + yellow palette — encouraging and approachable, not clinical
- **WhatsApp:** Green + white chat bubbles with clear sender distinction — pattern already internalized by this user base
- **Calm / Headspace:** Soft gradients, muted tones — calm apps use calm colors
- **Notion:** Warm off-white canvas over pure white — reduces eye strain on long reading sessions

### Dos

- Use color functionally: primary = navigation and trust, accent = action, semantic = system state
- Test every color combination for contrast using WCAG tools before shipping
- Use warm off-white (`#FAFAF8`) as the base canvas, not pure white
- Reserve crisis colors (`#1A6B8A` range) exclusively for crisis-level states — never decorative use
- Design in both light and dark mode from the start
- Use gradient backgrounds only in hero/landing areas — never in functional UI zones

### Don'ts

- Don't use pure white backgrounds throughout — it reads as a hospital interface
- Don't use green as a primary brand color without testing cultural reception across Islamic and East Asian user groups
- Don't use color alone to signal state — always pair with an icon and text
- Don't use highly saturated, vibrant palettes — they read as playful/gamified, wrong for a welfare context
- Don't use more than 3–4 intentional colors in the functional UI
- Don't use dark backgrounds as the default — high proportion of phone-in-daylight users

### Critical Failure Mode

**Using a bright, gamified color palette (high saturation, multiple competing accent colors).** This signals "fun product" which undermines trust in a welfare context and actively conflicts with the emotional seriousness of topics like immigration status, safety, and financial distress.

---

## Dimension 3 — Typography & Readability

### Governing Principles

- **Flesch-Kincaid Readability:** Text aimed at multilingual/ESL audiences should target a reading grade level of 6–8. Short sentences, common words, active voice.
- **WCAG 1.4.4 Resize Text:** Text must scale to 200% without loss of functionality or content clipping.
- **Typographic Rhythm (Bringhurst, "The Elements of Typographic Style"):** Line height 1.5–1.6x for body text is the standard for sustained reading. Below 1.4x causes line-skipping errors under stress.
- **Variable Font Performance:** On low-bandwidth connections, a single variable font file outperforms loading 4 separate weight files. This is a performance requirement, not an aesthetic one.
- **Multi-script Readability:** Latin-script fonts must be chosen for compatibility with fallback rendering of Arabic (RTL), Amharic (Ethiopic script), Chinese (CJK), and Devanagari.

### Applied to SettleBot

**Recommended Type System:**

| Token | Size | Use |
|---|---|---|
| `--text-xs` | 12px | Captions, metadata, timestamps |
| `--text-sm` | 14px | Secondary labels — never for body content |
| `--text-base` | 16px | Body text, chat messages — absolute minimum |
| `--text-md` | 18px | Subheadings within long responses |
| `--text-lg` | 20px | Section headers |
| `--text-xl` | 24px | Page titles |
| `--text-2xl` | 30px | Hero/landing areas only |

**Other Type Rules:**

- **Primary Typeface:** Inter Variable (already in use) — excellent screen readability at small sizes, strong Latin/Greek/Cyrillic coverage. Single variable font file for performance.
- **Monospace:** JetBrains Mono or system monospace — for phone numbers, addresses, cost figures
- **Line Height:** 1.6 for body/chat text, 1.3 for headings
- **Line Length:** 60–75 characters per line (45–55 on mobile) — beyond 80 causes tracking loss under stress
- **Font Weight:** Regular (400) body, Medium (500) UI labels, SemiBold (600) emphasis, Bold (700) section titles only
- **Letter Spacing:** +0.01em for UI labels and buttons, normal for body copy

### State of the Art

- **Linear App:** Inter + clean scale + heavy whitespace — gold standard for readable, information-dense interfaces
- **Notion:** Inter-family with generous line height — comfortable for long reading sessions
- **Telegram / WhatsApp:** 16px base minimum in chat, comfortable for all literacy levels — the direct chat pattern benchmark

### Dos

- Lock the body/chat minimum at 16px — never smaller
- Use `font-display: swap` on web fonts to prevent invisible text during load
- Use Inter variable font (single file) for performance on slow connections
- Apply 1.6 line-height to all chat message text — single highest-impact typography choice for comprehension
- Use bold/strong within responses to highlight actionable information (phone numbers, place names, costs) — never for decoration
- Format numbers consistently: KSh 2,500 — not 2500 or KShs2,500

### Don'ts

- Don't use a display or serif typeface as primary
- Don't mix more than 2 typefaces in the UI
- Don't allow text to span full viewport width on desktop — constrain to max-width: 680px for chat content
- Don't reduce font size to fit more content — break content up instead
- Don't use italic for long passages — it reduces reading speed for non-native readers by ~12%
- Don't use all-caps for anything longer than 3 words

### Critical Failure Mode

**Setting body/chat text at 14px or below.** On a mid-range Android phone in daylight, 14px at 1.4 line height is functionally unreadable for a stressed user reading their second language. This single decision cascades into abandonment.

---

## Dimension 4 — Information Architecture & Flow

### Governing Principles

- **Miller's Law:** The human mind can hold 7 ±2 items in working memory. Navigation structures with more than 5–7 items overwhelm users who are cognitively taxed.
- **Progressive Disclosure (Tog on Interface):** Show only what the user needs right now. Reveal complexity on demand, never upfront.
- **Job-to-be-Done Framework (Christensen):** The JTBD for SettleBot: *"Help me solve this specific problem I have right now, in a city I don't know."* Every architectural choice must serve that job.
- **Hick's Law:** Decision time increases logarithmically with the number of choices. Fewer, clearer options always beat more options.

### Applied to SettleBot

**The Ideal First-Time Anonymous User Journey:**

```
Land on page
  → Single clear value proposition (one sentence)
  → One primary CTA: "Ask a question" or "Start chatting"
  → No login prompt yet
  → Chat opens immediately
  → Starter prompts visible (Housing / Safety / Transport / Visa)
  → User types or taps a starter
  → Bot responds warmly
  → After 2nd exchange: soft prompt "Save your conversation? Create a free account."
  → User continues regardless of choice
```

This flow has **zero forced decisions before value delivery**.

**Response Style Selector — Recommended Label Redesign:**

| Current Label | Recommended Label | Reason |
|---|---|---|
| Direct | Quick Answer | Clearer outcome, less formal |
| Guided | Walk Me Through It | Conversational, outcome-focused |
| Conversational | Let's Talk About It | Signals dialogue and warmth |

- Present as a segmented control at the top of the chat
- Persist the choice per session/user
- One-line description on hover/tap: "Short, factual, to the point"
- Never require style selection before the first message — default to "Quick Answer"

**Information Hierarchy within a Bot Response:**

```
1. Direct answer       (first sentence — the answer, never preamble)
2. Context / explanation (2–4 sentences)
3. Specific details    (costs, addresses, phone numbers — visually distinct)
4. What to do next     (optional action step)
5. Source caveat       (if needed — subtle, not prominent)
```

### State of the Art

- **Perplexity AI:** Answer first, sources below — immediately scannable even before reading
- **Gov.uk:** Gold standard for stressed, low-literacy users — one task per page, plain English, no navigation clutter during task flows
- **Airbnb onboarding:** Progressive disclosure of features — users aren't hit with all options at once

### Dos

- Zero-friction entry to chat — no form, no login, no choice before first message
- Persistent, minimal navigation: Logo + Chat + History (authenticated) — nothing else visible during chat
- Structure bot responses with the answer in the first sentence — never start with "Great question!" or preamble
- Group related follow-up suggestions as tap targets after each response (3 maximum)
- Conversation history in a collapsible sidebar on desktop, full-screen toggle on mobile
- Auto-scroll to latest message; pause auto-scroll if user scrolls up (they're reading)

### Don'ts

- Don't put the chat behind a navigation menu — it is the product
- Don't require style selection before the first message
- Don't show more than 3 suggested follow-ups — choice paralysis
- Don't use breadcrumbs inside the chat interface
- Don't show empty conversation history as a blank area — use it as onboarding
- Don't paginate chat history — use infinite scroll with a "back to top" anchor

### Critical Failure Mode

**Placing the registration gate before value delivery.** If the user must create an account before typing their first question, the majority of distressed first-time users will not proceed. The entire value proposition of the application collapses at the door.

---

## Dimension 5 — Device & Context of Use

### Governing Principles

- **Mobile-First Design (Luke Wroblewski, 2011):** Design constraints of the smallest screen first, then scale up. Mobile-first forces priority decisions that desktop-first defers.
- **Context of Use (ISO 9241-11):** Usability is defined by effectiveness, efficiency, and satisfaction in a specific context. SettleBot's context includes: outdoor use, low battery, intermittent connectivity, emotional distress, one-handed use.
- **Network-Aware Design:** The median mobile internet speed in Kenya is 15–25 Mbps but drops significantly outside Nairobi's CBD. On campus or in budget accommodation, 3G/H+ is common. Every kilobyte matters.
- **Touch Target Standards:** Minimum 44×44px (Apple HIG) / 48×48dp (Google Material). Below this, error rate on mobile keyboards increases sharply.

### Applied to SettleBot

**Primary Device Profile:**

- Android phones 5.5"–6.7" (Tecno, Infinix, Samsung A-series dominate East African markets) — primary
- iOS (iPhone SE to iPhone 14) — secondary
- Desktop (library computers, campus labs) — tertiary
- No tablets assumed as primary

**Connectivity Tiers:**

| Tier | Connection | Design Response |
|---|---|---|
| 1 | Good WiFi | Full experience, animations, images |
| 2 | Mobile data 4G | Full experience, optimized assets |
| 3 | 3G/H+ | Text-first, deferred assets, skeleton screens |
| 4 | 2G / edge | Core function must work — text in, text out, no JS blocking input |

**Context-Specific Constraints:**

- **One-handed use:** Chat input and send button in bottom-right zone — the natural thumb anchor for right-handed users
- **Outdoor use (glare):** High contrast is not optional — required for outdoor legibility
- **Low battery / power save mode:** Dark mode as first-class design — saves battery on OLED screens (dominant in Android mid-range)
- **Interrupted sessions:** State must persist. On return, show last conversation automatically with "Continue where you left off" anchor

**Page Weight Budget:**

| Asset | Budget |
|---|---|
| Initial page load | < 200KB HTML+CSS (no JS blocking render) |
| Chat interface total | < 500KB on first load including fonts |
| Bot response | Text-only, no images within answers |
| Profile pictures | WebP, max 80×80px, lazy loaded |
| Animations | None autoplay on homepage |

### State of the Art

- **M-Pesa (Safaricom):** Designed for exactly this market — fast, text-first, works on low-end Android, bottom-tab navigation
- **WhatsApp:** The benchmark for mobile chat UX in this market — users already understand its patterns
- **BBC News Africa (mobile):** Excellent text-first, image-deferred design for low-bandwidth African mobile users

### Dos

- Design mobile layout first — every element
- Place the chat input bar at the bottom of the screen (fixed), always visible
- Use bottom navigation (not hamburger menus) for authenticated users — 3 items max: Chat, History, Profile
- Implement service worker / offline fallback — queue messages when connectivity drops
- Use `srcset` and WebP for all images with JPEG fallback
- Lazy-load everything below the fold
- Implement skeleton loading screens (not spinners) — they reduce perceived wait time by ~20%

### Don'ts

- Don't use hover-dependent interactions — touch screens have no hover state
- Don't use fixed sidebars on mobile
- Don't auto-open the mobile keyboard on page load
- Don't use custom scroll behaviors that fight native momentum scrolling
- Don't serve desktop assets to mobile — implement responsive asset delivery
- Don't design for landscape orientation as a primary use case

### Critical Failure Mode

**Building and testing exclusively on a laptop browser at full width.** The entire mobile experience becomes an afterthought, chat inputs will be unreachable, text will be too small, and the user's primary device will feel like a degraded version of the "real" product.

---

## Dimension 6 — Conversational UI Patterns

### Governing Principles

- **Grice's Maxims of Cooperative Conversation:** Be truthful, informative (not more than needed), relevant, and clear. These apply to both the bot's language and the interface surrounding it.
- **Turn-Taking Signal Design:** In human conversation, we use prosody and eye contact to signal whose turn it is. In chat UI, visual cues (typing indicator, message alignment, avatar) must replace these signals clearly.
- **Latency Tolerance Research (Nielsen Norman Group):** Users tolerate up to 1 second of wait before feeling the interface is slow. At 3 seconds without feedback, they assume something is broken. For SettleBot's 3–5 second response time, a visible animated thinking indicator is architecturally mandatory.
- **Response Chunking:** Long responses are more tolerable when streamed progressively than when delivered as a sudden text dump. Streaming renders each sentence as it arrives — it reduces perceived latency and mimics natural conversation.

### Applied to SettleBot

**Chat Anatomy:**

```
[Bot Icon]  [Message bubble]                         [Timestamp on hover]
                                   [User bubble]  [Status indicator]

[Bot typing indicator: three animated dots — appears within 300ms]

[Input bar with placeholder text | Send button]
[Starter prompt chips — visible when input is empty]
```

**Response Streaming:**
SettleBot uses GPT-4o-mini which supports streaming via the OpenAI API. The front-end should stream the response token-by-token. This converts a "4-second wait" into "I can see it thinking in real time" — transforming perceived latency from a flaw into a feature.

**Message Bubble Design:**

| Message Type | Alignment | Background | Notes |
|---|---|---|---|
| User | Right | Warm sand `#F0EAE0` | No avatar needed |
| Bot (default) | Left | White `#FFFFFF` | Subtle brand icon, 4px shadow |
| Bot (error) | Left | White + amber left-border | Icon + "I had trouble with that. Try again?" |
| Bot (crisis) | Left | White + teal left-border | Distinct from normal responses |

**Handling Long Responses:**
The AI will sometimes return 500+ word responses (Guided/Conversational styles):
- Render markdown: bold headings, bullet lists, numbered steps
- Section headings visually distinct (slightly larger, medium weight)
- Collapsible sections for responses over 400 words — show first 200 words with inline "Read more" expand

**Source Attribution:**
- Subtle "Based on settlement resources" below the response, expandable on tap
- Never show raw chunk IDs or ChromaDB scores
- When `web_search_used: true`, indicate: "I also checked current web sources for this."

### State of the Art

- **Claude.ai / ChatGPT:** Token streaming as default — users now expect to see the response building. No streaming = feels broken.
- **Perplexity AI:** Sources inline with numbered citations, collapsible source panel below — clean, honest, not intrusive
- **Intercom / Zendesk AI:** Typing indicators that appear within 200ms — the industry standard
- **Pi.ai:** Extremely warm conversational tone with natural pacing — the empathy benchmark for AI chat UI

### Dos

- Implement response streaming — single highest-impact improvement to perceived performance
- Show typing indicator within 300ms of message send, always
- Render markdown in bot responses: headers, bold, bullets, numbered lists
- Show 3 suggested follow-up chips after every response, contextually generated
- Persist conversation context — the bot should reference earlier parts naturally
- Add "Copy response" icon on hover/long-press
- Add "Was this helpful?" micro-feedback (👍 / 👎) per message

### Don'ts

- Don't dump the full response at once after 4 seconds of silence — stream it
- Don't show raw confidence scores, intent names, or technical metadata to the user
- Don't use chat bubbles for system messages — use subtle toast notifications
- Don't make starter prompt chips disappear after the first message
- Don't hard-wrap bot responses without markdown — a wall of unformatted text is unreadable

### Critical Failure Mode

**No typing/thinking indicator + full-response dump after 4 seconds.** The user sends a message, sees nothing for 4 seconds, then 300 words appear at once. This reads as "the system froze and then something happened." It destroys the perception of a live, responsive, intelligent system — which is the core product promise.

---

## Dimension 7 — Accessibility & Inclusion

### Governing Principles

- **WCAG 2.2:** For a welfare application serving a vulnerable population, AA compliance is the minimum; AAA should be targeted for key flows (chat, crisis response).
- **Inclusive Design (Microsoft Toolkit):** Designing for the margins improves the center. A design optimized for a low-vision user on a small phone in bright sunlight is a better design for everyone.
- **Cognitive Accessibility (WCAG 2.1 SC 3.1–3.3):** Plain language, consistent navigation, error identification, and input assistance are accessibility requirements for cognitively stressed or low-literacy users.
- **RTL Layout Support:** Arabic, Persian, Urdu, and Hebrew are RTL scripts. CSS logical properties and `dir="rtl"` must be implemented systematically, not patched.

### Highest Priority WCAG Criteria

| Criterion | Level | Why it matters for SettleBot |
|---|---|---|
| 1.4.3 Contrast (Minimum) | AA | Mobile outdoor use, daylight glare, low-end screens |
| 1.4.4 Resize Text | AA | Multilingual users often increase font size |
| 1.4.10 Reflow | AA | 400% zoom without horizontal scroll — critical for low vision |
| 2.4.3 Focus Order | AA | Keyboard navigation for campus computer users |
| 2.4.7 Focus Visible | AA | Tab navigation must show visible focus ring |
| 3.1.1 Language of Page | A | `lang` attribute on `<html>` — affects screen reader pronunciation |
| 3.1.2 Language of Parts | AA | Mark language of non-English content in responses |
| 3.3.1 Error Identification | A | Errors described in text, not just color |
| 1.3.1 Info and Relationships | A | Chat bubbles must have semantic role — not just visual layout |

### Screen Reader Implementation for Chat

- Message container: `role="log"` and `aria-live="polite"` — announces new messages without interrupting
- Typing indicator: `aria-live="assertive"` — signals the system is responding
- Message timestamps: use `<time datetime="ISO">` element
- Feedback buttons: proper `<button>` elements with `aria-label="Mark as helpful"` / `aria-label="Mark as not helpful"`

### RTL Layout

- Use CSS `dir="auto"` on message bubbles — browser auto-detects text direction per message
- Icons that imply direction (arrows, back buttons, chat bubble tails) must mirror in RTL context
- Input bar alignment flips in RTL: text right-to-left, send button on left side
- Use CSS logical properties (`margin-inline-start` not `margin-left`) throughout

### Cognitive Accessibility

- Plain language in all UI copy — no legal jargon, no technical vocabulary in user-facing text
- Consistent navigation: same items in same positions across all pages
- No time limits on any interaction — a user needing extra time must never be timed out mid-entry
- Error messages state what happened AND what to do: "Your message is too long (over 2,000 characters). Please shorten it and try again."

### State of the Art

- **Gov.uk:** The accessibility gold standard for government/civic digital services — every pattern has an accessible implementation documented
- **BBC Accessibility Guidelines:** Comprehensive public standard covering both web and conversational interface patterns
- **Airbnb:** Strong accessibility engineering — open-source component library includes exhaustive ARIA implementations

### Dos

- Run automated accessibility audit (axe, Lighthouse) on every page before shipping any feature
- Test with VoiceOver (iOS) and TalkBack (Android) on the chat interface specifically
- Implement `aria-live="polite"` on the chat message container
- Use `<button>` not `<div onclick>` for all interactive elements
- Ensure all form inputs have associated `<label>` elements (not just placeholders)
- Implement skip navigation: "Skip to chat" as first focusable element

### Don'ts

- Don't rely on placeholder text as the only label for form fields
- Don't use `display:none` for content that should be read by screen readers — use visually-hidden utility class
- Don't disable zoom on mobile (`user-scalable=no`) — this is an accessibility violation and breaks WCAG 1.4.4
- Don't use low-contrast ghost text for secondary information
- Don't add `aria-label` to elements that already have visible text
- Don't use `tabindex > 0` — it breaks natural focus order

### Critical Failure Mode

**Disabling pinch-to-zoom (`user-scalable=no`) on mobile.** This is an extremely common mistake and an outright accessibility violation. It blocks low-vision users from being able to use the application at all. It is a one-line mistake with a catastrophic effect.

---

## Dimension 8 — Trust, Transparency & Safety Signals

### Governing Principles

- **Trust Calibration (Parasuraman & Riley, 1997):** Users must neither over-trust nor under-trust AI systems. Over-trust leads to acting on bad information; under-trust leads to abandonment. The interface must honestly signal both capability and limitation.
- **Verification Heuristic (Norman, "The Design of Everyday Things"):** Users trust systems that give them the means to verify information. A phone number is more trusted when the UI says where it came from.
- **Privacy Paradox (Acquisti, 2009):** For SettleBot's user base, immigration-sensitive data (visa status, financial situation) is genuinely sensitive — privacy signals must be explicit and honest, not buried in a footer.
- **Crisis Response Standards (Mental Health America, Samaritans):** Emergency resources must appear before, not after, the AI-generated response.

### Applied to SettleBot

**The Grounding Rule as a UX Signal:**
SettleBot's backend enforces a strict rule: contact details are only surfaced if they appear verbatim in retrieved documents. This is a trust feature, not just a safety feature. The UI must make this visible:

- Contact details in responses appear in a distinct **"verified detail" chip**: shield icon + information + tooltip: *"Found in verified settlement resources"*
- When contact details cannot be found, the fallback message is styled distinctly (amber, italic, external link icon) — never buried in response text

**Crisis Detection UI Protocol:**
When `crisis_level` = "medium" or "high" in the API response:

1. Render a calm, full-width compassion banner **above** the response — soft teal background, warm icon (not red/alarm)
2. Message: *"It sounds like you may be going through a really tough time. These contacts can help right now:"*
3. Show 2–3 emergency contacts in large, tap-to-call format (Kenya Red Cross, university counseling, police emergency)
4. The AI response renders **below** this banner — never instead of it
5. Do not use red for this banner — calm blue or deep teal only

**Data and Privacy Signals:**

- Anonymous users: persistent subtle badge in chat header — "You're chatting anonymously — your conversation isn't saved"
- Authenticated users: on first login, plain-language data use explanation (not a GDPR wall): *"We save your conversations so you can come back to them. We never share your data without your permission."*
- The `allowDataUsage` field must have a corresponding, accessible toggle in profile settings — reachable in 2 taps

**AI Transparency:**

- Visibly clear that the user is talking to an AI: "I'm SettleBot, an AI assistant for international students in Nairobi."
- The bot says "I don't know" when it doesn't — styled as honest guidance, not an error
- When `web_search_used: true`, indicate: "I also checked current web sources for this."

### State of the Art

- **Woebot:** Exemplary crisis signal design — always surfaces "Talk to a human" button, never buries crisis resources
- **Claude.ai:** Explicit "I'm an AI" framing, consistent acknowledgment of uncertainty
- **Citizen (safety app):** High trust through hyper-local, verified information with clear sourcing

### Dos

- Display verified contact details in a distinct visual chip with a provenance signal
- Surface crisis resources above the AI response, not as a footnote
- Keep anonymous session status permanently visible in the interface
- Let the AI say "I'm not sure" — and make that message look helpful, not like an error
- Include a persistent "Talk to a human" or "Contact your university's international office" link in navigation

### Don'ts

- Don't hide the fact that it's an AI
- Don't bury privacy controls in a nested settings menu — accessible in 2 taps maximum
- Don't style crisis banners in red — calm colors de-escalate, red escalates
- Don't let a verified contact detail look identical to an AI-inferred statement
- Don't show a generic error message when the AI is uncertain

### Critical Failure Mode

**Displaying AI-generated contact information with no visual distinction from verified, retrieved contacts.** If a user in a crisis dials a hallucinated phone number because the interface made it look authoritative, the application has done active harm. The grounding rule exists in the backend — it must be honoured visually in the frontend.

---

## Dimension 9 — Competitive & Domain Benchmarking

### Governing Principles

- **Category Design:** Users evaluate a product relative to the mental model of every similar product they've used. SettleBot will be compared to WhatsApp (chat), Google (search), and university web portals (information) — not to other AI assistants they may never have used.
- **Steal Like an Artist (UX version):** Good designers borrow proven patterns from the best in adjacent domains. The goal is not copying — it is applying what works in a related context.

### Category 1: AI Chat Assistants

| App | Learn | Avoid |
|---|---|---|
| **ChatGPT** | Clean distraction-free UI; markdown rendering; streaming responses | Too many utilities for a welfare app; assumes tech-savvy user |
| **Claude.ai** | Honest AI framing; "I'm not sure" states; long-form response formatting | Complex system prompt UI irrelevant for SettleBot |
| **Pi.ai** | Warmest AI personality in the market; relationship-first framing; calm visual design | Less task-completion focused — SettleBot needs warmth AND information |
| **Perplexity AI** | Source attribution done well; answer-first structure; follow-up questions | Search-centric UI not suitable for a welfare chatbot |

### Category 2: Crisis & Welfare Applications

| App | Learn | Avoid |
|---|---|---|
| **Crisis Text Line** | Zero friction entry; conversation as the interface | Too stripped-down for SettleBot's topic navigation needs |
| **Woebot** | Emotional check-in before task; crisis resource placement; warm language | Mental health framing too clinical for a settlement assistant |
| **UNHCR Signpost** | Designed for displaced persons; icon-heavy; multiple languages first-class | Overly simplified — SettleBot's users are university students |
| **Wysa** | Excellent crisis escalation design; smooth handoff to human resources | Mental health focused; settlement context absent |

### Category 3: International Student & Newcomer Platforms

| App | Learn | Avoid |
|---|---|---|
| **InterNations** | Community + information hybrid; local guide content by city topic | Social network complexity; assumes settled users |
| **University portals** | Topic-organized information (housing, visa, health) | Static, non-conversational; notoriously terrible UX — SettleBot's opportunity |
| **Expatica** | Practical cost info, neighborhood breakdowns | Article-heavy, not interactive; no personalization |

### Category 4: East African Digital Products

| App | Learn | Avoid |
|---|---|---|
| **M-Pesa (Safaricom)** | Bottom navigation; text-first; fast on low-end devices; trusted by the market | Financial app patterns don't map to chat |
| **WhatsApp** | Chat bubble conventions; read receipts; typing indicators — already internalized by 95%+ of users | E2E encryption feature expectations SettleBot cannot match |
| **Jumia** | Mobile-first, image-light, works on spotty connectivity | Shopping context irrelevant |

### The Biggest Opportunity Gap

University international office portals are universally hated by international students. They are static, poorly organized, written in bureaucratic English, non-multilingual, and not conversational. **SettleBot's direct competition is a PDF and a FAQ page.** This is both the opportunity and the bar — which is low, but the temptation is to only clear it when SettleBot could lap the field entirely.

### Dos

- Adopt WhatsApp's chat bubble conventions — the user already knows them
- Adopt Perplexity's source panel pattern — collapsible, below the response
- Adopt Crisis Text Line's zero-friction entry — the chat IS the landing page
- Adopt Pi.ai's tone calibration — warm, personal, never robotic
- Adopt Gov.uk's plain language standards for all UI copy

### Don'ts

- Don't try to match the visual complexity of InterNations
- Don't replicate university portal IA — it is the anti-pattern
- Don't borrow consumer app gamification (streaks, badges, points) — deeply wrong for a welfare tool
- Don't add social features (following, sharing, community) — they dilute clarity of purpose

### Critical Failure Mode

**Benchmarking against the wrong category.** If the design team looks at Duolingo, Notion, or a SaaS dashboard for inspiration, they will build the wrong product. The relevant comparison set is welfare tools, civic tech, and crisis support — not productivity apps.

---

## Dimension 10 — Design System Recommendations

### Governing Principles

- **Atomic Design (Brad Frost):** Build from atoms (colors, type, spacing) → molecules (input + label + error) → organisms (chat bubble group) → templates (chat page) → pages.
- **Design Token Architecture:** All visual decisions defined as named tokens, not raw values. A token like `--color-crisis` can be changed globally without hunting through CSS files.
- **Component-Driven Development:** Each component is designed once, tested for all states, and never reimplemented ad hoc.

### Recommended Foundation

**Stack:** Tailwind CSS utility-first approach is well-suited for a Flask/Jinja2 stack. It avoids React component library overhead while enabling systematic design tokens.

### Design Token Set

```css
/* ===== COLOR TOKENS ===== */
--color-primary:        #1A5276;
--color-primary-light:  #2E86C1;
--color-accent:         #E67E22;
--color-surface:        #FAFAF8;
--color-surface-raised: #FFFFFF;
--color-border:         #E8E4E0;
--color-text-primary:   #2C2C2C;
--color-text-secondary: #6B6B6B;
--color-text-muted:     #9E9E9E;
--color-success:        #27AE60;
--color-warning:        #F39C12;
--color-crisis:         #1A6B8A;   /* calm teal — NOT red */
--color-error:          #C0392B;   /* form validation only */
--color-bubble-user:    #F0EAE0;
--color-bubble-bot:     #FFFFFF;

/* ===== TYPOGRAPHY TOKENS ===== */
--font-primary:         'Inter Variable', system-ui, sans-serif;
--font-mono:            'JetBrains Mono', monospace;
--text-xs:              0.75rem;   /* 12px — captions only */
--text-sm:              0.875rem;  /* 14px — metadata only */
--text-base:            1rem;      /* 16px — minimum for body */
--text-md:              1.125rem;  /* 18px — subheadings */
--text-lg:              1.25rem;   /* 20px — section headers */
--text-xl:              1.5rem;    /* 24px — page titles */
--text-2xl:             1.875rem;  /* 30px — hero only */
--line-height-body:     1.6;
--line-height-heading:  1.3;

/* ===== SPACING SCALE (4px base) ===== */
--space-1:  4px;
--space-2:  8px;
--space-3:  12px;
--space-4:  16px;
--space-6:  24px;
--space-8:  32px;
--space-12: 48px;

/* ===== RADIUS ===== */
--radius-sm:          6px;
--radius-md:          12px;
--radius-lg:          20px;
--radius-bubble-bot:  18px 18px 18px 4px;
--radius-bubble-user: 18px 18px 4px 18px;

/* ===== SHADOW ===== */
--shadow-bubble: 0 2px 4px rgba(0,0,0,0.06);
--shadow-card:   0 4px 12px rgba(0,0,0,0.08);
```

### Non-Negotiable Components

Every component below must be designed in all listed states before implementation:

| Component | Required States |
|---|---|
| **Chat Bubble (Bot)** | Default, verified-contact, warning-caveat, crisis-variant, error-variant, streaming |
| **Chat Bubble (User)** | Default, sending, failed |
| **Typing Indicator** | Animated 3-dot pulse, `aria-live` accessible |
| **Message Input Bar** | Empty (with starter chips), focused, filled, sending, disabled, char-limit warning |
| **Response Style Selector** | 3 options, selected, hover, mobile segmented control |
| **Crisis Banner** | Appears above response, calm teal color, tap-to-call emergency contacts |
| **Verified Contact Chip** | Shield icon + data + provenance tooltip |
| **Unverified Info Indicator** | Amber, italic, external link icon |
| **Starter Prompt Chips** | Default, hover, tap, dismissed state |
| **Skeleton Loader** | Chat bubble shape, animated shimmer |
| **Empty State** | Conversation list empty, first-time user, no results |
| **Toast Notification** | Success, error, info — auto-dismiss 4s |
| **Bottom Navigation Bar** | 3 tabs, active state, notification badge |

### Animation & Motion Principles

- Use `prefers-reduced-motion` media query — all animations must have a no-motion fallback
- **Typing indicator:** ease-in-out, 600ms cycle, 3 dots staggered by 100ms
- **Message appear:** `opacity 0→1` + `translateY 8px→0`, 200ms ease-out — subtle, not bouncy
- **No slide-in animations for bot responses** — they feel slow when the user is waiting for information
- **Page transitions:** none — instant navigation, no page-level animations

### Icon & Illustration System

- **Icons:** Phosphor Icons (MIT license) — excellent coverage including culturally neutral iconography; avoid FontAwesome's more western-coded human figure icons
- **Illustrations (empty states, onboarding):** Geometric and abstract, multicultural — no illustrated human faces (skin tone assumptions), no illustrations that culturally code as Western

### State of the Art

- **Radix UI + Tailwind:** Current gold standard for accessible, themeable component systems without framework lock-in
- **Material Design 3 (Google):** Most mature accessible component system — useful as reference even if not used directly

### Dos

- Define every color, spacing, and radius as a named token — no magic numbers in CSS
- Build the typing indicator first — it is the most important UI component in the application
- Design every component in all states before implementation
- Implement `prefers-reduced-motion` and `prefers-color-scheme` (dark mode) from day one
- Document every component with accessibility requirements alongside visual spec

### Don'ts

- Don't use a heavy React-based component library if the frontend stays as Flask/Jinja2
- Don't define colors as hex values inline in templates — always through tokens
- Don't design desktop and mobile as separate products
- Don't skip loading/skeleton states — they are components, not afterthoughts
- Don't use `border-radius: 50%` for chat bubbles — it creates pill shapes that look like buttons

### Critical Failure Mode

**Building without a design token system.** Without tokens, every color change requires searching through hundreds of CSS rules, dark mode is nearly impossible to implement correctly, and the crisis color ends up hardcoded in five different places with five slightly different values. Technical debt in the design layer directly translates to an inconsistent, untrustworthy-looking interface.

---

## Theme Specification — Astra Pro

Astra Pro is the adopted visual theme for SettleBot's front-end. It is one of the most widely used, performance-optimised, and accessibility-conscious WordPress themes — its design language translates directly into a clean CSS design system for any server-rendered stack. The specification below defines every aspect of the Astra Pro aesthetic as it applies to SettleBot: typography, spacing, color role, buttons, forms, layout, and motion. This section is the implementation contract — every element built on the front-end must conform to it.

---

### Why Astra Pro

- **Performance-first:** Astra Pro generates minimal CSS and loads no unnecessary JavaScript. Its philosophy directly aligns with SettleBot's page weight budget requirements (< 200KB HTML+CSS on initial load).
- **Readability-centred:** Astra Pro's default type scale is deliberately generous — larger base sizes, relaxed line heights, strong heading contrast. This maps exactly onto SettleBot's multilingual, mobile-first readability requirements.
- **Whitespace as content:** Astra Pro treats spacing as a design element, not padding between things. Sections breathe. Content is never cramped. This reduces cognitive load for stressed users.
- **Neutral, professional, trustworthy aesthetic:** Astra Pro does not look like a startup landing page or a gamified app. It looks like a reliable, professional service — which is exactly the trust signal SettleBot's users need.

---

### Typography System

#### Font Families

Astra Pro's default and most widely used font pairing is adopted here:

| Role | Font | Weights | Fallback |
|---|---|---|---|
| **Body & UI** | Plus Jakarta Sans | 400, 500, 600 | system-ui, -apple-system, sans-serif |
| **Headings** | Plus Jakarta Sans | 600, 700, 800 | system-ui, -apple-system, sans-serif |
| **Monospace** (phone numbers, costs, codes) | JetBrains Mono | 400, 500 | Consolas, monospace |

**Google Fonts import (single optimised request):**
```
https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap
```

**Why Plus Jakarta Sans over Inter:**
Plus Jakarta Sans is Astra Pro's flagship font for modern templates. It has slightly more personality than Inter at display sizes (warmer curves, stronger heading presence) while remaining equally readable at body sizes. It also carries a subtle warmth that Inter's strict geometric structure lacks — appropriate for a welfare-focused application.

---

#### Font Size Scale

Astra Pro uses a modular scale with a **17px body base** — larger than most frameworks default to, deliberately chosen for readability on varied devices and for audiences that include non-native readers. SettleBot adopts this exactly.

| Token | Size | rem | Use |
|---|---|---|---|
| `--font-size-xs` | 12px | 0.706rem | Timestamps, metadata, captions |
| `--font-size-sm` | 14px | 0.824rem | Secondary labels, helper text — never body content |
| `--font-size-base` | **17px** | **1rem** | **Body text, chat messages, paragraphs — the floor** |
| `--font-size-md` | 19px | 1.118rem | Lead text, intro paragraphs |
| `--font-size-lg` | 22px | 1.294rem | Card titles, subheadings within content |
| `--font-size-xl` | 28px | 1.647rem | Section headings (H3 level) |
| `--font-size-2xl` | 36px | 2.118rem | Page-level headings (H2 level) |
| `--font-size-3xl` | 46px | 2.706rem | Hero headings (H1 level) |
| `--font-size-4xl` | 56px | 3.294rem | Landing hero only — large viewport |

> **Rule:** 17px is the absolute minimum for any content a user needs to read. 14px is permitted only for timestamps, metadata, and system-level labels. Nothing a user reads for information goes below 17px.

---

#### Line Height

Astra Pro uses relaxed line heights — the defining characteristic that makes its typography feel spacious and readable compared to tighter alternatives.

| Context | Line Height | Reason |
|---|---|---|
| Body text / chat messages | **1.8** | Maximum comfort for sustained reading, ESL users, mobile |
| Lead / intro paragraphs | 1.75 | Slightly tighter but still generous |
| Subheadings (H3–H4) | 1.4 | Headings need less space between lines |
| Display headings (H1–H2) | 1.2 | Large type is readable at tighter leading |
| UI labels / buttons | 1.3 | Compact enough for controls |
| Form helper text | 1.6 | Readable but compact |

---

#### Letter Spacing

| Context | Spacing | Reason |
|---|---|---|
| Body text | `0` (normal) | Never add letter spacing to body — it breaks word-shape recognition |
| UI labels, nav links | `+0.01em` | Subtle openness for short label strings |
| H3–H4 headings | `-0.01em` | Slight tightening for medium headings feels more solid |
| H1–H2 display headings | `-0.02em` | Tight tracking at large sizes is a hallmark of Astra Pro's heading style |
| Buttons | `+0.01em` | Adds visual clarity to button labels |
| ALL CAPS text (badges, tags) | `+0.08em` | Required when text is all-caps — prevents cramped appearance |

---

#### Font Weight Usage

| Weight | Name | Use |
|---|---|---|
| 400 | Regular | Body text, paragraphs, chat messages |
| 500 | Medium | Nav links, form labels, secondary UI text |
| 600 | SemiBold | Card titles, subheadings, emphasis within body, button labels |
| 700 | Bold | H2–H3 headings, important callouts |
| 800 | ExtraBold | H1 / hero headings only |

---

#### Heading Scale (Astra Pro standard)

```css
h1 { font-size: 46px;  font-weight: 800; line-height: 1.2;  letter-spacing: -0.02em; }
h2 { font-size: 36px;  font-weight: 700; line-height: 1.25; letter-spacing: -0.02em; }
h3 { font-size: 28px;  font-weight: 700; line-height: 1.3;  letter-spacing: -0.01em; }
h4 { font-size: 22px;  font-weight: 600; line-height: 1.35; letter-spacing: -0.01em; }
h5 { font-size: 19px;  font-weight: 600; line-height: 1.4;  letter-spacing: 0; }
h6 { font-size: 17px;  font-weight: 600; line-height: 1.4;  letter-spacing: 0; }
```

---

### Spacing System

Astra Pro uses a **4px base grid** with a generous application of whitespace. Sections breathe — internal element spacing is tighter, section-level spacing is expansive.

#### Spacing Scale

```css
--space-1:   4px;    /* Micro: icon gaps, tight inline spacing */
--space-2:   8px;    /* XS: badge padding, tight component internal */
--space-3:   12px;   /* SM: form field internal padding, small gaps */
--space-4:   16px;   /* Base: standard component padding */
--space-5:   20px;   /* MD: card internal spacing */
--space-6:   24px;   /* LG: between related elements */
--space-8:   32px;   /* XL: between distinct components */
--space-10:  40px;   /* 2XL: section sub-divisions */
--space-12:  48px;   /* 3XL: major section padding (mobile) */
--space-16:  64px;   /* 4XL: section padding (tablet) */
--space-20:  80px;   /* 5XL: section padding (desktop) — Astra Pro's signature spacious sections */
--space-24:  96px;   /* 6XL: hero section padding (desktop) */
```

#### Applied Spacing Rules

| Element | Top/Bottom Padding | Left/Right Padding |
|---|---|---|
| Page sections (desktop) | `--space-20` (80px) | Container margin |
| Page sections (tablet) | `--space-16` (64px) | Container margin |
| Page sections (mobile) | `--space-12` (48px) | `--space-4` (16px) |
| Cards | `--space-6` (24px) | `--space-6` (24px) |
| Form fields | `--space-3` (12px) | `--space-4` (16px) |
| Buttons (standard) | `--space-3` (12px) | `--space-6` (24px) |
| Buttons (large) | `--space-4` (16px) | `--space-8` (32px) |
| Nav items | `--space-3` (12px) | `--space-4` (16px) |
| Chat message bubbles | `--space-3` (12px) | `--space-4` (16px) |
| Between chat messages | `--space-3` (12px) | — |
| Footer sections | `--space-12` (48px) | `--space-6` (24px) |

---

### Layout & Grid

#### Container

Astra Pro uses a centred container with generous max-width and side padding:

```css
--container-max:    1200px;   /* Astra Pro default content width */
--container-wide:   1400px;   /* For full-bleed sections with inset content */
--container-narrow: 760px;    /* For single-column reading content (chat, forms) */
--container-px:     24px;     /* Side padding on mobile/tablet */
```

All page content lives within `.container` (max-width: 1200px, centred, padded). Chat content lives within `.container-narrow` (max-width: 760px) — Astra Pro's single-column reading width, chosen because line lengths beyond 75 characters impair readability.

#### Column Grid

Astra Pro uses a 12-column grid. Common page layouts:

| Layout | Columns |
|---|---|
| Full content | 12 / 12 |
| Content + sidebar | 8 / 4 |
| Three columns | 4 / 4 / 4 |
| Two columns (cards) | 6 / 6 |
| Narrow centred (forms, chat) | 8 centred (offset 2) |

---

### Color Role (Astra Pro applied to SettleBot palette)

Astra Pro separates colors by role, not by hue name. The existing SettleBot color palette is retained but remapped to Astra Pro's role system:

| Role Token | Value | Astra Pro Role | Use |
|---|---|---|---|
| `--color-primary` | `#6b4423` | Primary brand | CTAs, active nav, links, focus rings |
| `--color-primary-light` | `#8b5f47` | Primary light | Hover states on primary |
| `--color-primary-subtle` | `#f4e7d1` | Primary tint | Backgrounds of primary-adjacent elements |
| `--color-accent` | `#f59e0b` | Accent / highlight | Underlines, badges, icon backgrounds, small highlights |
| `--color-surface` | `#fafaf8` | Site background | Page canvas — Astra Pro off-white, not pure white |
| `--color-surface-card` | `#ffffff` | Card background | Elevated surfaces: cards, chat bubbles (bot), modals |
| `--color-surface-input` | `#ffffff` | Input background | Form fields |
| `--color-border` | `#e4e4e7` | Border | Card borders, input borders, dividers |
| `--color-border-strong` | `#d4d4d8` | Strong border | Active input borders, table borders |
| `--color-text-primary` | `#18181b` | Body text | Main readable content |
| `--color-text-secondary` | `#3f3f46` | Secondary text | Subtext, descriptions |
| `--color-text-muted` | `#71717a` | Muted text | Timestamps, captions, metadata |
| `--color-text-inverse` | `#ffffff` | Inverse text | Text on dark backgrounds |
| `--color-success` | `#10b981` | Success state | Confirmations — always paired with icon |
| `--color-warning` | `#f59e0b` | Warning state | Caution — unverified info indicator |
| `--color-error` | `#ef4444` | Error state | Form validation only — never for crisis |
| `--color-crisis` | `#1a6b8a` | Crisis (calm) | Crisis banners — calm teal, never red |
| `--color-bubble-user` | `#f0eae0` | User message | Warm sand — distinct from bot without harshness |
| `--color-bubble-bot` | `#ffffff` | Bot message | Clean white, readable |

---

### Button Styles (Astra Pro standard)

Astra Pro buttons are clean, slightly rounded, and rely on weight and colour — not shadows or gradients — to communicate action hierarchy.

#### Variants

| Variant | Background | Text | Border | Use |
|---|---|---|---|---|
| **Primary** | `--color-primary` | White | None | Main CTA — one per section max |
| **Primary Outline** | Transparent | `--color-primary` | 2px solid primary | Secondary CTA alongside primary |
| **Ghost** | Transparent | `--color-text-secondary` | None | Tertiary actions, nav-adjacent links |
| **Danger** | `#ef4444` | White | None | Destructive actions (delete, leave) |
| **Danger Outline** | Transparent | `#ef4444` | 2px solid error | Soft destructive, e.g. cancel |

#### Sizing

| Size | Height | Padding (H/V) | Font Size | Border Radius |
|---|---|---|---|---|
| **SM** | 36px | 8px / 16px | 14px | 6px |
| **MD** (default) | 44px | 12px / 24px | 15px | 6px |
| **LG** | 52px | 16px / 32px | 17px | 8px |
| **XL** | 60px | 20px / 40px | 18px | 8px |

> **Astra Pro button radius:** 6–8px — not pill-shaped (too playful), not sharp (too harsh). Clean, purposeful rounding.

#### States

- **Hover:** `translateY(-2px)` + slight background darkening (10%) — Astra Pro's signature subtle lift
- **Active / Pressed:** `translateY(0)` — returns to baseline, removes lift
- **Focus:** 2px offset outline in `--color-primary` at 50% opacity — WCAG 2.4.7 compliant
- **Disabled:** 50% opacity, `cursor: not-allowed` — no hover effect
- **Loading:** Button text replaced by a 16px spinner, button width locked to prevent layout shift

---

### Form Elements (Astra Pro standard)

Astra Pro forms are clean and spacious — no cramped fields, no visual noise.

#### Input Fields

```
Height:           48px (standard), 52px (large forms)
Padding:          12px top/bottom, 16px left/right
Border:           1px solid --color-border
Border radius:    6px
Background:       --color-surface-input (#ffffff)
Font size:        17px (base body size — never smaller)
Font weight:      400
Line height:      1.5
Color:            --color-text-primary

Focus state:
  Border:         2px solid --color-primary
  Box shadow:     0 0 0 3px rgba(107, 68, 35, 0.12)
  Outline:        none (replaced by box shadow)

Error state:
  Border:         2px solid --color-error
  Box shadow:     0 0 0 3px rgba(239, 68, 68, 0.12)

Placeholder:
  Color:          --color-text-muted (#71717a)
  Font weight:    400
```

#### Labels

```
Font size:        15px
Font weight:      500
Color:            --color-text-secondary
Margin bottom:    6px
```

#### Helper / Error Text

```
Font size:        14px
Line height:      1.5
Margin top:       6px
Error color:      --color-error
Helper color:     --color-text-muted
```

#### Textarea (chat input specifically)

```
Min height:       52px (single line equivalent)
Max height:       180px (before internal scroll)
Resize:           vertical only
Font:             inherit body (Plus Jakarta Sans, 17px)
Line height:      1.75
Padding:          14px 16px
```

---

### Card Style (Astra Pro standard)

Astra Pro cards are defined by subtle borders and soft shadows — no heavy drop shadows, no coloured backgrounds on content cards.

```
Background:       --color-surface-card (#ffffff)
Border:           1px solid --color-border (#e4e4e7)
Border radius:    12px
Padding:          24px
Box shadow:       0 2px 8px rgba(0, 0, 0, 0.06)

Hover (interactive cards):
  Box shadow:     0 8px 24px rgba(0, 0, 0, 0.10)
  Transform:      translateY(-2px)
  Transition:     all 0.2s ease
```

---

### Navigation (Astra Pro standard)

Astra Pro's header is clean, transparent-on-scroll, sticky, and minimal.

```
Header height:          72px (desktop), 60px (mobile)
Background:             rgba(255, 255, 255, 0.95) with backdrop-filter: blur(10px)
Border bottom:          1px solid --color-border on scroll only
Logo font size:         22px, font-weight: 700
Nav link font size:     15px, font-weight: 500
Nav link color:         --color-text-secondary
Nav link hover:         --color-primary
Nav link active:        --color-primary, underline accent
Gap between nav items:  32px
CTA button in nav:      Primary MD size
```

---

### Footer (Astra Pro standard)

```
Background:             --color-primary (#6b4423) — dark branded footer
Padding top/bottom:     64px / 40px (desktop), 48px / 32px (mobile)
Column grid:            4 columns (desktop), 2 columns (tablet), 1 column (mobile)
Column gap:             48px (desktop), 32px (tablet)
Heading color:          #ffffff, font-weight: 700, font-size: 16px
Link color:             rgba(255,255,255,0.7)
Link hover:             #ffffff
Footer bottom bar:      rgba(0,0,0,0.2) background, 14px text, centred
```

---

### Radius Scale

Astra Pro uses consistent, purposeful border radius — not uniform pills, not sharp squares.

```css
--radius-xs:    4px;    /* Tags, badges, inline chips */
--radius-sm:    6px;    /* Buttons, inputs, small cards */
--radius-md:    10px;   /* Standard cards, dropdowns */
--radius-lg:    14px;   /* Large cards, modals, panels */
--radius-xl:    20px;   /* Feature cards, hero elements */
--radius-bubble-bot:   18px 18px 18px 4px;   /* Bot chat bubble */
--radius-bubble-user:  18px 18px 4px 18px;   /* User chat bubble */
```

---

### Shadow Scale

Astra Pro shadows are subtle — they establish depth without competing with content.

```css
--shadow-xs:   0 1px 2px rgba(0, 0, 0, 0.05);             /* Subtle lift */
--shadow-sm:   0 2px 8px rgba(0, 0, 0, 0.06);             /* Cards at rest */
--shadow-md:   0 4px 16px rgba(0, 0, 0, 0.08);            /* Cards on hover */
--shadow-lg:   0 8px 32px rgba(0, 0, 0, 0.10);            /* Dropdowns, popovers */
--shadow-xl:   0 16px 48px rgba(0, 0, 0, 0.14);           /* Modals */
--shadow-2xl:  0 24px 64px rgba(0, 0, 0, 0.18);           /* Full-page overlays */
```

---

### Motion & Transitions (Astra Pro standard)

Astra Pro uses fast, purposeful transitions — not decorative animation.

```css
--transition-fast:    0.15s ease;     /* Hover color changes, opacity toggles */
--transition-base:    0.2s ease;      /* Buttons, links, icon transforms */
--transition-smooth:  0.3s ease;      /* Menus, dropdowns, panels */
--transition-slow:    0.4s ease;      /* Page-level elements, hero animations */
```

**Astra Pro motion rules:**
- Hover lifts: `translateY(-2px)`, `transition: 0.2s ease` — not more than 2px
- No bounce, spring, or elastic easing — these feel playful, not professional
- No entrance animations on content — only on interactive overlays (modals, dropdowns)
- All transitions respect `prefers-reduced-motion: reduce`

```css
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
```

---

### Complete CSS Token Reference

The following block is the complete Astra Pro design token set for SettleBot. This is the single source of truth for all CSS custom properties — no values appear as raw hex or numbers anywhere in the codebase outside this block.

```css
:root {
    /* ── BRAND COLORS ── */
    --color-primary:         #6b4423;
    --color-primary-hover:   #573820;
    --color-primary-light:   #8b5f47;
    --color-primary-subtle:  #f4e7d1;
    --color-accent:          #f59e0b;
    --color-accent-hover:    #dc6803;

    /* ── SEMANTIC COLORS ── */
    --color-success:         #10b981;
    --color-success-bg:      #f0fdf4;
    --color-warning:         #f59e0b;
    --color-warning-bg:      #fffbeb;
    --color-error:           #ef4444;
    --color-error-bg:        #fef2f2;
    --color-crisis:          #1a6b8a;
    --color-crisis-bg:       #eff8ff;

    /* ── SURFACE COLORS ── */
    --color-surface:         #fafaf8;
    --color-surface-card:    #ffffff;
    --color-surface-input:   #ffffff;
    --color-surface-muted:   #f4f4f5;

    /* ── TEXT COLORS ── */
    --color-text-primary:    #18181b;
    --color-text-secondary:  #3f3f46;
    --color-text-muted:      #71717a;
    --color-text-disabled:   #a1a1aa;
    --color-text-inverse:    #ffffff;

    /* ── BORDER COLORS ── */
    --color-border:          #e4e4e7;
    --color-border-strong:   #d4d4d8;
    --color-border-focus:    #6b4423;

    /* ── CHAT COLORS ── */
    --color-bubble-user:     #f0eae0;
    --color-bubble-bot:      #ffffff;

    /* ── TYPOGRAPHY ── */
    --font-primary:          'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    --font-mono:             'JetBrains Mono', Consolas, monospace;

    --font-size-xs:          0.706rem;   /* 12px */
    --font-size-sm:          0.824rem;   /* 14px */
    --font-size-base:        1rem;       /* 17px  ← html font-size: 17px */
    --font-size-md:          1.118rem;   /* 19px */
    --font-size-lg:          1.294rem;   /* 22px */
    --font-size-xl:          1.647rem;   /* 28px */
    --font-size-2xl:         2.118rem;   /* 36px */
    --font-size-3xl:         2.706rem;   /* 46px */
    --font-size-4xl:         3.294rem;   /* 56px */

    --line-height-tight:     1.2;
    --line-height-snug:      1.35;
    --line-height-normal:    1.5;
    --line-height-relaxed:   1.7;
    --line-height-loose:     1.8;        /* body text default */

    --font-weight-regular:   400;
    --font-weight-medium:    500;
    --font-weight-semibold:  600;
    --font-weight-bold:      700;
    --font-weight-extrabold: 800;

    /* ── SPACING ── */
    --space-1:    4px;
    --space-2:    8px;
    --space-3:    12px;
    --space-4:    16px;
    --space-5:    20px;
    --space-6:    24px;
    --space-8:    32px;
    --space-10:   40px;
    --space-12:   48px;
    --space-16:   64px;
    --space-20:   80px;
    --space-24:   96px;

    /* ── LAYOUT ── */
    --container-max:    1200px;
    --container-wide:   1400px;
    --container-narrow: 760px;
    --container-px:     24px;

    /* ── BORDER RADIUS ── */
    --radius-xs:            4px;
    --radius-sm:            6px;
    --radius-md:            10px;
    --radius-lg:            14px;
    --radius-xl:            20px;
    --radius-bubble-bot:    18px 18px 18px 4px;
    --radius-bubble-user:   18px 18px 4px 18px;

    /* ── SHADOWS ── */
    --shadow-xs:    0 1px 2px rgba(0, 0, 0, 0.05);
    --shadow-sm:    0 2px 8px rgba(0, 0, 0, 0.06);
    --shadow-md:    0 4px 16px rgba(0, 0, 0, 0.08);
    --shadow-lg:    0 8px 32px rgba(0, 0, 0, 0.10);
    --shadow-xl:    0 16px 48px rgba(0, 0, 0, 0.14);
    --shadow-2xl:   0 24px 64px rgba(0, 0, 0, 0.18);

    /* ── TRANSITIONS ── */
    --transition-fast:    0.15s ease;
    --transition-base:    0.2s ease;
    --transition-smooth:  0.3s ease;
    --transition-slow:    0.4s ease;

    /* ── Z-INDEX ── */
    --z-base:             1;
    --z-dropdown:         1000;
    --z-sticky:           1020;
    --z-fixed:            1030;
    --z-modal-backdrop:   1040;
    --z-modal:            1050;
    --z-popover:          1060;
    --z-tooltip:          1070;
    --z-toast:            1080;
    --z-maximum:          99999;
}

/* ── BASE RESET ── */
html {
    font-size: 17px;          /* Astra Pro base — all rem values compute from this */
    scroll-behavior: smooth;
    -webkit-text-size-adjust: 100%;
}

body {
    font-family:    var(--font-primary);
    font-size:      var(--font-size-base);
    font-weight:    var(--font-weight-regular);
    line-height:    var(--line-height-loose);
    color:          var(--color-text-primary);
    background:     var(--color-surface);
    overflow-x:     hidden;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── HEADINGS ── */
h1 { font-size: var(--font-size-3xl); font-weight: var(--font-weight-extrabold); line-height: var(--line-height-tight);  letter-spacing: -0.02em; }
h2 { font-size: var(--font-size-2xl); font-weight: var(--font-weight-bold);      line-height: 1.25;                       letter-spacing: -0.02em; }
h3 { font-size: var(--font-size-xl);  font-weight: var(--font-weight-bold);      line-height: 1.3;                        letter-spacing: -0.01em; }
h4 { font-size: var(--font-size-lg);  font-weight: var(--font-weight-semibold);  line-height: var(--line-height-snug);    letter-spacing: -0.01em; }
h5 { font-size: var(--font-size-md);  font-weight: var(--font-weight-semibold);  line-height: 1.4;                        letter-spacing: 0; }
h6 { font-size: var(--font-size-base);font-weight: var(--font-weight-semibold);  line-height: 1.4;                        letter-spacing: 0; }

/* ── PARAGRAPHS ── */
p {
    line-height: var(--line-height-loose);
    color: var(--color-text-secondary);
    margin-bottom: var(--space-4);
}

/* ── LINKS ── */
a {
    color: var(--color-primary);
    text-decoration: none;
    transition: color var(--transition-fast);
}
a:hover { color: var(--color-primary-hover); text-decoration: underline; }

/* ── REDUCED MOTION ── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
```

---

### Astra Pro Compliance Checklist

Before any page or component is considered complete, verify the following:

- [ ] Font family is Plus Jakarta Sans loaded from Google Fonts (variable weight, single request)
- [ ] `html` font-size is set to `17px` — all rem values compute from this
- [ ] No body text, chat messages, or readable content is set below `17px`
- [ ] No font sizes are hardcoded — all use `--font-size-*` tokens
- [ ] Line height on all body/chat text is `1.8` (`--line-height-loose`)
- [ ] All colors reference `--color-*` tokens — no raw hex values in component CSS
- [ ] All spacing uses `--space-*` tokens — no raw pixel values
- [ ] Button hover state uses `translateY(-2px)` lift — no larger transforms
- [ ] Card shadows use `--shadow-sm` at rest, `--shadow-md` on hover
- [ ] Crisis banner uses `--color-crisis` (`#1a6b8a`) — never `--color-error` or red
- [ ] All transitions reference `--transition-*` tokens
- [ ] `prefers-reduced-motion` media query is present in base CSS
- [ ] Container max-width is `--container-max` (1200px) for general content, `--container-narrow` (760px) for chat/forms
- [ ] Border radius on buttons and inputs is `--radius-sm` (6px) — no pill shapes, no sharp corners

---

## Page Architecture & Information Hierarchy

### Governing Principles

- **Progressive Trust Model (BJ Fogg):** Users give more of themselves — attention, data, personal questions — as trust accumulates. The page sequence must be designed to build trust progressively, not extract commitment before it has been earned.
- **Information Scent (Pirolli & Card, 1999):** Users navigate by following cues that signal they are getting warmer — closer to what they need. Page labels, nav order, and link text must carry strong information scent toward the user's goal.
- **Anxiety Reduction Sequencing:** For welfare applications, the first pages a user encounters must reduce anxiety, not require effort. The sequence must answer: *What is this? → Is it for me? → Can I try it safely? → Can I trust it?* — in that order, before asking anything of the user.
- **Wayfinding Theory (Lynch, "The Image of the City"):** Users build a mental map of a space through landmarks, nodes, and paths. A well-ordered page architecture gives users clear landmarks so they always know where they are and how to return to safety — in this case, the chat.
- **Fitts's Law applied to navigation:** The most important destination (the chat) must be the most accessible — fewest clicks, largest target, most prominent position. Everything else is secondary.

---

### The Root Page Decision

**`/` is the chat interface — anonymous, no login required, open immediately.**

This is the single most important architectural decision in the entire application. The product IS the chat. Therefore the root IS the chat. No hero banner, no feature list, no marketing copy, no navigation wall stands between the user and the thing they came for.

The moment a user lands on SettleBot, they are already in the right place. No clicking. No deciding. No reading. The input is there. The starter prompts are visible. They can type immediately.

This is what Crisis Text Line does. What ChatGPT does. What Claude.ai does. What Pi.ai does. It is the correct pattern for any application where the core value is a conversation.

**Authenticated vs. anonymous rendering at `/`:**
The root renders the same URL for both user types — not a redirect. The template checks session state:
- Anonymous user → anonymous chat interface, session created automatically via cookie
- Authenticated user → full chat interface with conversation history accessible in a collapsible side panel

**The soft registration prompt:**
After 2–3 exchanges in the anonymous chat, a non-blocking, dismissible prompt appears within the chat — not a modal, not a page interruption: *"Save this conversation? Create a free account →"* The user can ignore it and continue. The conversation never pauses.

---

### Current Page Inventory

**Public (no authentication):**
`/` · `/about` · `/faq` · `/features` · `/terms-and-conditions` · `/privacy-policy` · `/anonymous/chat`

**Authentication:**
`/authentication/register` · `/authentication/sign-in` · `/authentication/password-reset-request` · `/authentication/password-reset/<token>`

**Authenticated:**
`/account/settlement-assistant` · `/account/conversations` · `/account/conversation/<id>` · `/account/create-conversation` · `/account/profile` · `/account/update-profile` · `/account/change-password`

**Error pages:** 403 · 404 · 500

**Administration:** Dashboard and management pages

---

### The Core Problem With The Current Inventory

The existing page set was built to support the application's features. It was not designed around the user's journey. There are seven critical gaps and a navigation order problem that together create a friction-heavy, trust-undermining experience — particularly for the anxious, first-time user who matters most.

The current implied navigation order is a **company website pattern applied to a welfare tool** — it prioritises information about the product over access to the product. It must be restructured.

---

### Missing Pages — Prioritised

#### 1. Emergency Resources — `/emergency` — CRITICAL

**This is the most serious missing page in the entire application.**

SettleBot's backend detects crisis states and responds empathetically. But there is no standalone, always-accessible, zero-friction page that lists verified emergency contacts for Nairobi. A user in acute distress — who cannot type a coherent question, whose hands are shaking, who arrived at a hospital they don't recognise — cannot rely on the chat. They need a page reachable in one tap from anywhere that shows police emergency, Kenya Red Cross, the nearest hospital, and their university's welfare line.

**Specification:**
- Publicly accessible — no login, no chat required
- Linked from the **primary navigation on every single page** — not the footer
- The nav link is visually distinct: calm teal badge, never red
- Content is verified, static contacts only — not AI-generated, not RAG-retrieved. Hard-coded by the team from official sources, reviewed quarterly
- Opens instantly with zero JS dependency — pure HTML + CSS fallback if scripts fail
- All phone numbers are `tel:` links on mobile — one tap to call
- Must function even when the server is degraded — cacheable by a service worker

**State of the art:** The Samaritans website shows its emergency number on every single page without exception. UNHCR Signpost pins emergency resources permanently to the top of the interface. Crisis Text Line's persistent "if you are in immediate danger, call 911" banner is visible before any interaction begins.

---

#### 2. How It Works — `/how-it-works` — HIGH

A first-time user does not know: Will it understand Swahili? Will it judge me for my question? Will my conversation be shared? These are real anxiety points — especially for users from cultures with lower baseline trust in digital systems. The About and Features pages exist but neither answers *"What will happen when I type my question?"* in a simple, honest, step-by-step way.

**Specification:**
- Three steps maximum: **Ask → Get an Answer → Save It (Optional)**
- Each step in plain language, illustrated simply — no complex diagrams
- Honest about AI limitations: *"I use a knowledge base of settlement guides. For the very latest information, I'll also check current web sources."*
- Addresses the biggest fear directly: *"Your conversations are private. Anonymous users are not tracked."*
- CTA at the bottom: *"Ready? Ask your first question →"* — links directly to `/`

**State of the art:** Intercom's "How it works" flows — 3 steps, visual, no jargon. Notion's onboarding explains what the product does before showing how to use it.

---

#### 3. Topic Landing Pages — `/topics/<topic>` (×8) — HIGH

SettleBot's most important users will not arrive at the homepage. They will arrive from a Google search: *"how to find student housing in Nairobi"*, *"is Westlands safe for international students"*, *"how to open a bank account in Kenya as a foreigner."* Without topic-specific landing pages, these users never find SettleBot at all.

**The 8 topic pages:**
- `/topics/housing` — Housing in Nairobi for International Students
- `/topics/safety` — Safety Guide: Nairobi Neighbourhoods
- `/topics/banking` — Banking & Finance for International Students
- `/topics/transport` — Getting Around Nairobi
- `/topics/immigration` — Visa & Immigration Support
- `/topics/healthcare` — Healthcare & Medical Services
- `/topics/university` — University Processes & Administration
- `/topics/cultural-adaptation` — Cultural Adaptation & Life in Kenya

**What each topic page is — and is not:**
These are **thin, focused entry points into the chat**, not encyclopedias. Each page contains:
- A heading specific to the topic
- 2–3 short paragraphs of human-written introduction (not AI-generated) — enough to signal deep knowledge of that topic
- 3–5 example questions students commonly ask
- One prominent CTA: *"Ask SettleBot about [topic] →"* — opens the chat at `/` with topic context pre-loaded

That is the entire page. It is not a blog post, not a full guide, not a knowledge base article. It is a door into the chat — its purpose is both SEO discoverability and trust-building before the user asks their first question.

**State of the art:** Expatica's country-specific guides. Gov.uk topic pages. Intercom's help centre — organised by topic, each with a direct "find your answer" CTA.

---

#### 4. Post-Registration Onboarding — `/account/welcome` — HIGH

When a user registers, they are currently dropped into the application with no orientation. The transition from anonymous to registered user is a significant moment — the user has committed trust. The application must honour that.

**Specification:**
- Seen only once — flagged in the user model after completion
- 3 steps maximum, skippable at any point, no data required
- Step 1: *"Your conversations are now saved. Here's what that means."*
- Step 2: One optional preference — language or Nairobi neighbourhood — framed as *"Help me help you better"*, not as a form
- Step 3: Introduce the 3 response styles with one plain-English sentence each
- Final CTA: *"Start your first conversation →"*

**State of the art:** Duolingo's language selector — one question, warm framing, always skippable. Headspace's first-session question: *"What brings you here?"* — single question that personalises immediately.

---

#### 5. Unified Settings — `/account/settings` — MEDIUM

Currently `/account/update-profile` and `/account/change-password` are separate, isolated routes. As the application grows, additional settings will accumulate: language preference, notification settings, data export, account deletion, and the existing `allowDataUsage` toggle. Scattering these creates a fragmented experience.

**Specification — tabbed sections:**
- **Profile** — name, username, avatar, language preference
- **Security** — change password, active sessions
- **Privacy** — `allowDataUsage` toggle, data export (GDPR right to portability), account deletion
- **Preferences** — default response style, neighbourhood/location context

Existing routes remain functional but are reached through the unified settings page, not as standalone navigation destinations.

**State of the art:** GitHub Settings. Linear Settings. Single URL, tabbed structure, each section a focused form.

---

#### 6. Accessibility Statement — `/accessibility` — MEDIUM

SettleBot commits to WCAG 2.2 AA compliance. An application making that commitment is expected to publish an Accessibility Statement:
- The compliance level targeted (WCAG 2.2 AA)
- Any known limitations stated honestly
- A contact mechanism for users who encounter barriers
- The date of the last accessibility audit

Linked from the footer. Short, plain language — not legal text. This is a trust signal for disabled users deciding whether the application is usable for them before they invest effort in it.

---

#### 7. Contact / Report an Issue — `/contact` — MEDIUM

There is currently no mechanism for a user to report an incorrect answer, a broken experience, or a concern about their data. This is a trust gap — it signals that feedback has no destination.

**Specification:**
- Simple form: name (optional), email (optional), category (incorrect info / accessibility / data concern / other), message
- No login required
- Confirmation: *"Thank you. We review every report."*
- Linked from the footer and as a subtle *"Report this response"* link on each bot message

---

### Navigation Architecture

#### Primary Navigation — Unauthenticated

| Position | Item | Destination | Notes |
|---|---|---|---|
| Left | SettleBot logo | `/` | Returns to the chat |
| Centre | How It Works | `/how-it-works` | For hesitant first-timers |
| Right | Emergency | `/emergency` | Visually distinct — calm teal badge, always visible |
| Right | Sign In | `/authentication/sign-in` | Ghost button |
| Right | Register | `/authentication/register` | Primary CTA button |

**Removed from primary nav:** About, FAQ, Features, Terms, Privacy — reference material, not navigation destinations for a user in need. These move to the footer only.

#### Primary Navigation — Authenticated

| Position | Item | Destination | Notes |
|---|---|---|---|
| Left | SettleBot logo | `/` | Returns to the chat |
| Right | Emergency | `/emergency` | Always present — every page, every state |
| Right | My Conversations | `/account/conversations` | |
| Right | [Avatar] | Dropdown | Profile · Settings · Sign out |

#### Footer — Four Columns

| Column | Links |
|---|---|
| **SettleBot** | About · How It Works · Languages Supported · Accessibility |
| **Help Topics** | Housing · Safety · Banking · Transport · Healthcare · Immigration · University · Cultural Adaptation |
| **Account** | Sign In · Create Account · My Conversations · Settings |
| **Legal & Support** | Privacy Policy · Terms of Use · Contact / Report an Issue · Emergency Resources |

---

### Error Pages — Content Strategy

For a welfare application, a user hitting a 404 may be a distressed user who followed a broken link while urgently seeking help. Error pages must not be cold or clinical.

**404 — Page Not Found:**
- Heading: *"We couldn't find that page."*
- One line: *"The link may have changed or expired."*
- Two options: Back button + *"Ask SettleBot a question →"* (→ `/`)

**403 — Access Denied:**
- Heading: *"You need to be signed in to see this."*
- Two options: *"Sign in"* + *"Continue as guest"* (→ `/`)

**500 — Server Error:**
- Heading: *"Something went wrong on our end."*
- Reassurance: *"Your conversation was not lost."*
- Action: *"Try again"* reload button
- **The Emergency nav link must remain visible** — a server error must never block access to emergency information

---

### The User Journey Mapped to Pages

| Journey Stage | User State | Entry Point | Key Requirement |
|---|---|---|---|
| Discovery via search | Purposeful, informed | `/topics/<topic>` → `/` | Topic page speaks their specific need; CTA drops them into chat |
| Discovery via referral | Curious, unsure | `/` | Root is already the chat — no intermediate step needed |
| First visit, urgent | Anxious, distressed | `/` directly | Input visible immediately, no decision required |
| Crisis, can't type | Acute distress | `/emergency` | One tap from nav, tap-to-call contacts, no chat required |
| First visit, hesitant | Uncertain | `/how-it-works` → `/` | Clarity before commitment, then straight to chat |
| Anonymous → registered | Trusting, committing | Register → `/account/welcome` → `/` | Honour the trust moment with warm onboarding |
| Returning authenticated | Habitual | `/` | History accessible, new chat ready — no friction |
| Account management | Administrative | `/account/settings` | Single destination, all settings in one place |
| Legal/policy reference | Deliberate | Footer links | Never in primary nav |

---

### Complete Recommended Page List

| Page | Route | Status | Priority |
|---|---|---|---|
| Chat (root) | `/` | Exists as `/anonymous/chat` — **must become the root** | CRITICAL |
| Emergency Resources | `/emergency` | **Missing** | CRITICAL |
| How It Works | `/how-it-works` | **Missing** | HIGH |
| Topic: Housing | `/topics/housing` | **Missing** | HIGH |
| Topic: Safety | `/topics/safety` | **Missing** | HIGH |
| Topic: Banking | `/topics/banking` | **Missing** | HIGH |
| Topic: Transport | `/topics/transport` | **Missing** | HIGH |
| Topic: Immigration | `/topics/immigration` | **Missing** | HIGH |
| Topic: Healthcare | `/topics/healthcare` | **Missing** | HIGH |
| Topic: University | `/topics/university` | **Missing** | HIGH |
| Topic: Cultural Adaptation | `/topics/cultural-adaptation` | **Missing** | HIGH |
| Post-registration Onboarding | `/account/welcome` | **Missing** | HIGH |
| Unified Settings | `/account/settings` | Fragmented — needs consolidation | MEDIUM |
| Accessibility Statement | `/accessibility` | **Missing** | MEDIUM |
| Contact / Report Issue | `/contact` | **Missing** | MEDIUM |
| About | `/about` | Exists — footer only | LOW |
| FAQ | `/faq` | Exists — footer only | LOW |
| Features | `/features` | Exists — footer only | LOW |
| Privacy Policy | `/privacy-policy` | Exists | — |
| Terms of Use | `/terms-and-conditions` | Exists | — |
| Sign In | `/authentication/sign-in` | Exists | — |
| Register | `/authentication/register` | Exists | — |
| Password Reset | `/authentication/password-reset-*` | Exists | — |
| Authenticated Chat | `/account/settlement-assistant` | Exists | — |
| Conversation History | `/account/conversations` | Exists | — |
| Individual Conversation | `/account/conversation/<id>` | Exists | — |
| Profile | `/account/profile` | Exists | — |
| 404 / 403 / 500 | Error routes | Exists — **needs content strategy** | HIGH |

---

### Critical Failure Mode

**Treating the homepage as a marketing page and the chat as a feature.** In a welfare application, the chat is not a feature — it is the entire service. When the first thing a user encounters is information about the product rather than the product itself, users in distress face a decision wall before they can get help. The root page is the chat. The chat is the root page. If the first action on arrival is not to type a question, the architecture is wrong.

---

## Priority Matrix

Ranked by impact on **user trust and task success** for SettleBot specifically.

| Rank | Dimension | Priority | Justification |
|---|---|---|---|
| 1 | Information Architecture & Flow | **CRITICAL** | If the user can't reach the chat instantly, nothing else matters — the entire product is inaccessible |
| 2 | Conversational UI Patterns | **CRITICAL** | The chat interface IS the product — a broken chat experience means a broken product, full stop |
| 3 | Trust, Transparency & Safety Signals | **CRITICAL** | One wrong contact detail acted on in a crisis is a human failure, not a bug — this has real-world consequences |
| 4 | User Psychology & Emotional Design | **HIGH** | The user's diminished cognitive and emotional state at point of use defines the entire design envelope |
| 5 | Device & Context of Use | **HIGH** | The majority of users will be on mid-range Android phones on mobile data — a desktop-optimized product is the wrong product |
| 6 | Typography & Readability | **HIGH** | A stressed, multilingual user reading dense AI responses on a small screen — typography is comprehension, and comprehension is the service |
| 7 | Accessibility & Inclusion | **HIGH** | Disabling zoom or missing ARIA live regions breaks the experience for a significant portion of the user base and violates international standards |
| 8 | Color Theory & Visual Language | **MEDIUM-HIGH** | Color errors erode trust and can cause harm — but are recoverable with iteration |
| 9 | Design System Recommendations | **MEDIUM** | High leverage for maintainability and consistency, but impact is on team velocity and future quality, not immediate user harm |
| 10 | Competitive & Domain Benchmarking | **MEDIUM** | Informs strategic positioning and pattern borrowing — a reference layer, not a build layer |

---

## Design Principles Manifesto

These 8 principles are non-negotiable. Every design decision is evaluated against them before it ships.

---

### 1. Help First, Everything Else Second
The user gets an answer before they are asked for anything — no login, no form, no choice. Value is delivered immediately. Trust is earned before it is asked for.

---

### 2. Design for the Hardest Moment
Our user is not browsing. They are stuck. They may be scared. They may be reading in their second language on a low-end phone in the sun. Every design decision is evaluated against that baseline — not a comfortable user at a desktop.

---

### 3. Calm is a Design Choice
Every color, every animation, every word of copy must reduce stress, not add to it. If a design element could increase anxiety in a distressed user, it does not belong in this interface.

---

### 4. Honesty Over Impressiveness
The system must say "I don't know" when it doesn't know. Unverified contact details must look different from verified ones. The user must always be able to tell what the AI is confident about and what it is not. We never pretend.

---

### 5. The Phone is the Computer
Every layout, every interaction, every tap target, every page weight decision is made for a 6-inch Android screen on mobile data. Desktop is an enhancement. Mobile is the product.

---

### 6. Visible Thinking, Honest Waiting
The AI takes 3–5 seconds to respond. That time must never feel like silence. A typing indicator within 300ms. A streaming response that shows words arriving. The user must always know the system is working for them.

---

### 7. One Language is Not Enough
We serve 24+ languages and dozens of cultures. No color means the same thing to everyone. No icon is universal. No phrase translates perfectly. Every design decision is made with the awareness that the user may be experiencing this interface through a completely different cultural lens.

---

### 8. A Wrong Number in a Crisis is Not a UX Failure — It is a Human Failure
The grounding rule is not a backend concern. It is a design contract. Verified information looks verified. Unconfirmed information looks unconfirmed. Crisis resources are prominent, calm, and real. This principle has no exceptions.

---

*SettleBot UX/UI Analysis — Human-Centered Design Brief*
*Version 1.2 | June 2026 — Astra Pro theme specification and page architecture analysis added*
