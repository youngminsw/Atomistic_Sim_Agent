# Model Registry

| Model Key | Type | Intelligence | Speed | Description | Use Case |
|-----------|------|--------------|-------|-------------|----------|
| `direct/local-network-5090` | Local | Low | Medium | GLM-4.7 Flash on RTX 5090 (10.24.12.85). Privacy & Unlimited. | Drafts, experimental runs. |
| `direct/local-network-4090` | Local | Low | Medium | Qwen 3 (32B) on RTX 4090 (10.24.12.81). Extra local capacity. | Parallel local tasks. |
| `direct/gemini-3-pro-preview` | Cloud | Very High | Medium | Gemini 3 Pro (Preview). Highest reasoning. | Complex architecture, deep reasoning. |
| `direct/gemini-3-flash-preview` | Cloud | High | Very Fast | Gemini 3 Flash (Preview). Latest fast model. | High volume, quick responses. |
| `google/antigravity-gemini-3-flash` | Cloud | High | Very Fast | Baseline Gemini 3 Flash via Opencode CLI. | General coding, planning. |
| `google/antigravity-gemini-3-pro-high` | Cloud | Very High | Medium | Gemini 3 Pro via Opencode CLI. High rate limits. | Production grade tasks. |
| `google/gemini-3-flash-preview` | Cloud | High | Very Fast | Gemini 3 Flash Preview via Opencode CLI. | Inspection Agent default. |
| `direct/glm-4.7` | Cloud | High | Fast | GLM 4.7 Cloud API. | Alternative to Gemini. |
| `zai-coding-plan/glm-4.7` | Cloud | High | Fast | GLM 4.7 specialized for Coding Plan via Opencode CLI. | Planning tasks. |
