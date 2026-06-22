# ASA Runtime Spine Contract

Todo 1 freezes the runtime contract before deeper implementation. The current
status is a deliberate gap contract: each spine names the required assertion,
the known gap, and the later acceptance probe that must turn the spine green.

## Model/Provider/Transport

Provider calls must be selected by `api_protocol` and provider metadata. The
current blocker is the fixed `/v1/responses` path used by the provider tool
choice model.

## AgentSession

One durable session must own role prompt, provider state, history, events, tools,
skills, workflow state, and resume state. The current blocker is the frozen
`AsaAgentSession` DTO used for one goal.

## AgentLoop

The runtime loop must stream model events, execute tool calls, append tool
results, and continue until terminal assistant output. The current blocker is
the one-shot `choose_tools` flow.

## Prompt/Skill/Workflow Assembly

System prompt, role prompt, active skills, workflow gates, compaction summary,
transcript tail, tool schemas, provider metadata, and ledger facts must be
assembled through one path. The current blocker is ad hoc provider input using
only the current goal.

## Subagent/Task Runtime

Persistent agents must spawn bounded subagent jobs with list, inspect, await,
cancel, pause, resume, steer, progress, and output references. The current
blocker is the absence of a detached controllable job runtime.

## Context/Compaction/Resume

Compaction checkpoints and transcript tails must feed the next provider call
after resume. The current blocker is that live turn construction ignores stored
history and summaries.

## Tool Registry/Tool Runtime

Provider-visible tools must come from registry metadata with schema, provenance,
side-effect class, approval policy, load mode, owner, and gating. The current
blocker is minimal schemas plus placeholder domain capabilities.

## TUI/UX/Observability

TUI palettes, direct agent routing, model state, runtime timeline, and HUD must
render from live runtime state. The current blocker is static or semantic-line
surfaces that can drift from provider payloads and ledgers.
