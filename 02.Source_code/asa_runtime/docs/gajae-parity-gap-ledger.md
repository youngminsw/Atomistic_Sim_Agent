# Gajae-Style ASA Runtime Parity Gap Ledger

Date: 2026-06-19

This ledger is the acceptance anchor for the ASA TUI/runtime rebuild. Do not close a parity task because one local symptom is fixed. Close it only when the relevant row is implemented, tested, and visually checked through the interactive TUI path.

## Source Evidence

- Gajae model profiles: `/tmp/gajae-code/packages/coding-agent/src/config/model-profiles.ts`
- Gajae thinking levels: `/tmp/gajae-code/packages/coding-agent/src/thinking.ts`
- Gajae command validation thinking values: `/tmp/gajae-code/packages/coding-agent/src/modes/shared/agent-wire/command-validation.ts`
- Gajae input controller: `/tmp/gajae-code/packages/coding-agent/src/modes/controllers/input-controller.ts`
- Gajae selector controller: `/tmp/gajae-code/packages/coding-agent/src/modes/controllers/selector-controller.ts`
- Gajae autocomplete: `/tmp/gajae-code/packages/coding-agent/src/modes/prompt-action-autocomplete.ts`
- Gajae model selector: `/tmp/gajae-code/packages/coding-agent/src/modes/components/model-selector.ts`
- Gajae status/agent UI: `/tmp/gajae-code/packages/coding-agent/src/modes/components/status-line.ts`, `/tmp/gajae-code/packages/coding-agent/src/modes/components/agent-dashboard.ts`

## Parity Matrix

| Area | ASA status | Gap to close |
| --- | --- | --- |
| Slash and mention input | Partial: prompt-toolkit suggestions and `@agent` targeted chat exist. | Fuzzy ranked action autocomplete, richer skill/plugin suggestions, and equal fallback behavior outside prompt-toolkit. |
| Selector controller | Partial: arrow-key menus exist and now clear nested menus. | One controller surface for provider onboarding, model profile/model selection, settings, agent dashboard, plugins, sessions, and post-login recommendation. |
| Model profiles | Partial: `codex-eco`, `codex-medium`, and `codex-pro` now save default model plus agent overrides. | Grouped provider tabs, profile recommendation after login, profile editing, saved active profile display in HUD/status. |
| Thinking levels | Partial: ASA recognizes `inherit`, `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`. | Provider-specific mapping/clamping before real model calls, plus UI explanation for unsupported provider levels. |
| Provider login | Partial: browser OAuth URL/callback and provider credential store exist. | Clean user-facing success screen, no debug key dump, copy-open fallback when browser is invisible, provider-specific login flows beyond OpenAI path. |
| Model/provider catalog | Partial: grouped company catalog exists. | Gajae-like compact provider tabs, model search/filter, authenticated/provider availability badges, and separate login vs model configuration surface. |
| Agent dashboard | Partial: static workboard and `/model agents` rows exist. | Interactive dashboard, per-agent model editing, live activity, subagent run receipts, and custom agent creation. |
| Status/HUD | Partial: welcome panel, `/hud`, `/status`, and workboard exist. | Two-layer status line with active profile, token/context usage, background jobs, blockers, and active workflow state. |
| Skills/plugins | Partial: `/skills` lists simulation skills. | `/` skill action invocation, plugin discovery/install surfacing, and skill-specific workflow cards. |
| Agent session/loop | Partial: Python runtime skeleton and dry-run/tool-gateway paths exist. | Real provider-backed tool-calling loop, bounded native subagent sessions, durable transcript, interrupt/resume, and evidence-gated workflow progression. |
| Workflow enforcement | Partial: guards and workflow command smoke paths exist. | System-enforced stage transitions with evidence ledger gates, not only prompt text and user discipline. |
| Visual QA | Partial: pty tests cover selector clearing. | Routine terminal capture checks for every TUI batch at representative widths with CJK/wide-char inspection. |

## Current Closed Increment

- Model profile commands: `/model profiles`, `/model profile <name>`, and interactive `/model set` profile landing.
- Profile presets: `codex-eco`, `codex-medium`, `codex-pro`.
- Role assignments: `md_agent`, `ml_agent`, `feature_scale_agent`, `research_agent`, `qa_agent`.
- Thinking vocabulary: `inherit`, `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`.

## Next Priority

1. Clean OAuth/login success UI and remove debug-style key-value spam from normal TUI output.
2. Add post-login provider/model recommendation that opens the profile/model selector without requiring typed commands.
3. Add interactive agent dashboard for `@agent` summon, per-agent model assignment, and live status.
4. Add real provider-backed AgentLoop smoke where the model chooses tools and records evidence.
