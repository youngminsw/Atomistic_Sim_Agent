# ASA TUI Design System

## 1. Product Identity

ASA is a laboratory control room for atomistic simulation work. The interface should feel like an instrument panel for MD, MDN, feature-scale profile evolution, GraphDB memory, and QA gates, not a marketing page or generic chatbot.

## 2. Visual Signature

Signature: graphite control-room panels with an instrument-grade status rail. Cyan marks live routing and interactive affordances; amber marks gates, warnings, and required next actions; green marks connected or passed states; red marks blockers.

Avoid one-note purple/blue gradients, decorative blobs, large hero treatments, and playful illustration. The first screen must communicate "ready to operate a scientific runtime."

## 3. Palette Tokens

- `surface`: `#0b0f0e`
- `panel`: `#111614`
- `panel_raised`: `#18201d`
- `line`: `#33413b`
- `text`: `#e5ebe7`
- `muted`: `#9aa8a0`
- `accent_cyan`: `#63c7b2`
- `accent_amber`: `#d8a657`
- `success`: `#84c88a`
- `danger`: `#e06c75`
- `info`: `#7aa2f7`

ANSI color is allowed only when stdout is a TTY and color is not disabled. Non-TTY output remains plain text so tests, logs, and ledgers stay parseable.

## 4. Typography

Use the terminal monospace face. Favor short labels, tabular-looking status text, and compact scientific wording. Do not scale terminal text by viewport width or use negative spacing.

## 5. Layout

Primary TUI frames stay 92 terminal cells wide. Components must preserve CJK display-width correctness. Panels are dense, scan-friendly, and un-nested: welcome deck, HUD, workboard, chat deck, selectors, and command palette.

## 6. Components

- Welcome deck: model rail, evidence rail, agent rail, session/workspace rail.
- HUD: model/auth/session/chat/ledger/gate summary.
- Workboard: native agent roster and bounded responsibilities.
- Chat deck: direct Orchestrator chat and `@agent` routing without requiring `/chat`.
- Prompt toolbar: `/` for commands, `@` for agents, then the highest-frequency controller actions.
- Selectors: stable focus marker, no cursor-artifact corruption, escape/cancel path visible.

## 7. Interaction Rules

Plain text submits to the Orchestrator. `@agent` routes through the Orchestrator and summons the native specialist. `/login` handles provider login only; `/model` handles model selection. The UI should guide but not narrate feature manuals inside visible panels.
