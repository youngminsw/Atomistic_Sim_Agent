from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


LoginTokenMode = Literal["oauth", "api_key"]
LoginFlowKind = Literal["browser_oauth", "device_oauth", "api_key", "local", "manual_oauth"]

GATEWAY_BASE_URL: Final = "https://model-gateway.local/v1"
LOCAL_GATEWAY_BASE_URL: Final = "http://localhost:8787/v1"
MODEL_GATEWAY_TOKEN_ENV: Final = "MODEL_GATEWAY_TOKEN"


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    provider_id: str
    company: str
    label: str
    default_base_url: str
    default_auth_mode: str
    default_api_key_env: str
    summary: str


@dataclass(frozen=True, slots=True)
class LoginProfileSpec:
    profile_id: str
    provider_id: str
    company: str
    label: str
    summary: str
    token_mode: LoginTokenMode
    flow_kind: LoginFlowKind
    dashboard_url: str = ""


def _provider(
    provider_id: str,
    company: str,
    label: str,
    summary: str,
    *,
    base_url: str = GATEWAY_BASE_URL,
    auth_mode: str = "gateway",
    api_key_env: str = MODEL_GATEWAY_TOKEN_ENV,
) -> ProviderSpec:
    return ProviderSpec(provider_id, company, label, base_url, auth_mode, api_key_env, summary)


def _login(
    profile_id: str,
    provider_id: str,
    company: str,
    label: str,
    summary: str,
    token_mode: LoginTokenMode,
    flow_kind: LoginFlowKind,
    dashboard_url: str = "",
) -> LoginProfileSpec:
    return LoginProfileSpec(profile_id, provider_id, company, label, summary, token_mode, flow_kind, dashboard_url)


PROVIDER_SPECS: Final[tuple[ProviderSpec, ...]] = (
    _provider("openai-codex", "OpenAI", "ChatGPT Plus/Pro", "ChatGPT subscription OAuth account"),
    _provider("openai", "OpenAI", "OpenAI API", "Direct OpenAI API key", base_url="https://api.openai.com/v1", auth_mode="api_key", api_key_env="OPENAI_API_KEY"),
    _provider("anthropic", "Anthropic", "Claude", "Claude OAuth or direct Anthropic API", base_url="https://api.anthropic.com/v1", auth_mode="api_key", api_key_env="ANTHROPIC_API_KEY"),
    _provider("google-gemini-cli", "Google", "Google Cloud Code Assist", "Gemini CLI OAuth account"),
    _provider("google-antigravity", "Google", "Antigravity", "Google Antigravity subscription account"),
    _provider("github-copilot", "Developer Tools", "GitHub Copilot", "Copilot subscription account"),
    _provider("cursor", "Developer Tools", "Cursor", "Cursor subscription account"),
    _provider("xai", "xAI", "xAI", "Grok API or subscription gateway", base_url="https://api.x.ai/v1", auth_mode="api_key", api_key_env="XAI_API_KEY"),
    _provider("deepseek", "DeepSeek", "DeepSeek", "DeepSeek API key", base_url="https://api.deepseek.com/v1", auth_mode="api_key", api_key_env="DEEPSEEK_API_KEY"),
    _provider("kimi-code", "Moonshot / Kimi", "Kimi Code", "Kimi coding plan account"),
    _provider("moonshot", "Moonshot / Kimi", "Moonshot API", "Moonshot API key", base_url="https://api.moonshot.ai/v1", auth_mode="api_key", api_key_env="MOONSHOT_API_KEY"),
    _provider("qwen-portal", "Alibaba / Qwen", "Qwen Portal", "Qwen portal account"),
    _provider("alibaba-coding-plan", "Alibaba / Qwen", "Alibaba Coding Plan", "Alibaba coding subscription"),
    _provider("zai", "Z.AI", "Z.AI", "GLM coding plan account"),
    _provider("minimax-code", "MiniMax", "MiniMax Code", "MiniMax coding plan account"),
    _provider("perplexity", "Perplexity", "Perplexity", "Perplexity Pro/Max account or API", base_url="https://api.perplexity.ai", auth_mode="api_key", api_key_env="PERPLEXITY_API_KEY"),
    _provider("fireworks", "Fireworks", "Fireworks", "Fireworks API key", base_url="https://api.fireworks.ai/inference/v1", auth_mode="api_key", api_key_env="FIREWORKS_API_KEY"),
    _provider("firepass", "Fireworks", "Fire Pass", "Fireworks subscription gateway"),
    _provider("together", "Together", "Together", "Together API key", base_url="https://api.together.xyz/v1", auth_mode="api_key", api_key_env="TOGETHER_API_KEY"),
    _provider("cerebras", "Cerebras", "Cerebras", "Cerebras API key", base_url="https://api.cerebras.ai/v1", auth_mode="api_key", api_key_env="CEREBRAS_API_KEY"),
    _provider("huggingface", "Hugging Face", "Hugging Face Inference", "Hugging Face token", base_url="https://router.huggingface.co/v1", auth_mode="api_key", api_key_env="HF_TOKEN"),
    _provider("nvidia", "NVIDIA", "NVIDIA", "NVIDIA NIM API key", base_url="https://integrate.api.nvidia.com/v1", auth_mode="api_key", api_key_env="NVIDIA_API_KEY"),
    _provider("nanogpt", "NanoGPT", "NanoGPT", "NanoGPT API key", base_url="https://nano-gpt.com/api/v1", auth_mode="api_key", api_key_env="NANOGPT_API_KEY"),
    _provider("venice", "Venice", "Venice", "Venice API key", base_url="https://api.venice.ai/api/v1", auth_mode="api_key", api_key_env="VENICE_API_KEY"),
    _provider("ollama", "Local / Custom", "Ollama", "Local OpenAI-compatible Ollama", base_url="http://localhost:11434/v1", auth_mode="none", api_key_env="OLLAMA_API_KEY"),
    _provider("lm-studio", "Local / Custom", "LM Studio", "Local OpenAI-compatible LM Studio", base_url="http://localhost:1234/v1", auth_mode="none", api_key_env="LM_STUDIO_API_KEY"),
    _provider("vllm", "Local / Custom", "vLLM", "Local OpenAI-compatible vLLM", base_url="http://localhost:8000/v1", auth_mode="none", api_key_env="VLLM_API_KEY"),
    _provider("local_gateway", "Local / Custom", "Local ASA gateway", "Project-owned local gateway", base_url=LOCAL_GATEWAY_BASE_URL, auth_mode="none", api_key_env="RUNTIME_GATEWAY_TOKEN"),
)


LOGIN_PROFILE_SPECS: Final[tuple[LoginProfileSpec, ...]] = (
    _login("chatgpt_codex", "openai-codex", "OpenAI", "ChatGPT Plus/Pro", "ChatGPT subscription browser login", "oauth", "browser_oauth"),
    _login("chatgpt_codex_device", "openai-codex", "OpenAI", "ChatGPT Device Login", "headless/device ChatGPT login", "oauth", "device_oauth"),
    _login("openai_api", "openai", "OpenAI", "OpenAI API", "direct OpenAI API key", "api_key", "api_key", "https://platform.openai.com/api-keys"),
    _login("anthropic_pro", "anthropic", "Anthropic", "Anthropic Claude Pro/Max", "Claude subscription OAuth account", "oauth", "manual_oauth", "https://claude.ai/settings/keys"),
    _login("anthropic_api", "anthropic", "Anthropic", "Anthropic Claude API", "direct Claude API key", "api_key", "api_key", "https://console.anthropic.com/settings/keys"),
    _login("google_gemini_cli", "google-gemini-cli", "Google", "Google Cloud Code Assist", "Gemini CLI OAuth account", "oauth", "manual_oauth", "https://cloud.google.com/gemini/docs/codeassist"),
    _login("google_antigravity", "google-antigravity", "Google", "Antigravity", "Gemini/Claude/GPT-OSS subscription account", "oauth", "manual_oauth", "https://antigravity.google"),
    _login("github_copilot", "github-copilot", "Developer Tools", "GitHub Copilot", "Copilot subscription account", "oauth", "manual_oauth", "https://github.com/settings/copilot"),
    _login("cursor", "cursor", "Developer Tools", "Cursor", "Cursor subscription account", "oauth", "manual_oauth", "https://cursor.com/settings"),
    _login("xai", "xai", "xAI", "xAI", "Grok API key", "api_key", "api_key", "https://console.x.ai"),
    _login("deepseek", "deepseek", "DeepSeek", "DeepSeek", "DeepSeek API key", "api_key", "api_key", "https://platform.deepseek.com/api_keys"),
    _login("kimi_code", "kimi-code", "Moonshot / Kimi", "Kimi Code", "Kimi coding plan account", "oauth", "device_oauth", "https://auth.kimi.com"),
    _login("moonshot", "moonshot", "Moonshot / Kimi", "Moonshot API", "Moonshot/Kimi API key", "api_key", "api_key", "https://platform.moonshot.ai/console/api-keys"),
    _login("qwen_portal", "qwen-portal", "Alibaba / Qwen", "Qwen Portal", "Qwen portal account", "oauth", "manual_oauth", "https://chat.qwen.ai"),
    _login("alibaba_coding_plan", "alibaba-coding-plan", "Alibaba / Qwen", "Alibaba Coding Plan", "Alibaba coding plan account", "oauth", "manual_oauth"),
    _login("zai", "zai", "Z.AI", "Z.AI", "GLM coding plan account", "oauth", "manual_oauth", "https://chat.z.ai"),
    _login("minimax_code", "minimax-code", "MiniMax", "MiniMax Code", "MiniMax coding plan account", "oauth", "manual_oauth"),
    _login("perplexity", "perplexity", "Perplexity", "Perplexity", "Perplexity API key", "api_key", "api_key", "https://www.perplexity.ai/settings/api"),
    _login("fireworks", "fireworks", "Fireworks", "Fireworks", "Fireworks API key", "api_key", "api_key", "https://fireworks.ai/account/api-keys"),
    _login("firepass", "firepass", "Fireworks", "Fire Pass", "Fireworks subscription gateway", "oauth", "manual_oauth"),
    _login("together", "together", "Together", "Together", "Together API key", "api_key", "api_key", "https://api.together.xyz/settings/api-keys"),
    _login("cerebras", "cerebras", "Cerebras", "Cerebras", "Cerebras API key", "api_key", "api_key", "https://cloud.cerebras.ai/platform"),
    _login("huggingface", "huggingface", "Hugging Face", "Hugging Face Inference", "Hugging Face token", "api_key", "api_key", "https://huggingface.co/settings/tokens"),
    _login("nvidia", "nvidia", "NVIDIA", "NVIDIA", "NVIDIA API key", "api_key", "api_key", "https://build.nvidia.com"),
    _login("nanogpt", "nanogpt", "NanoGPT", "NanoGPT", "NanoGPT API key", "api_key", "api_key", "https://nano-gpt.com/api"),
    _login("venice", "venice", "Venice", "Venice", "Venice API key", "api_key", "api_key", "https://venice.ai/settings/api"),
    _login("ollama", "ollama", "Local / Custom", "Ollama", "local provider, no login required", "api_key", "local"),
    _login("lm_studio", "lm-studio", "Local / Custom", "LM Studio", "local provider, no login required", "api_key", "local"),
    _login("vllm", "vllm", "Local / Custom", "vLLM", "local provider, optional token", "api_key", "local"),
    _login("local_gateway", "local_gateway", "Local / Custom", "Local ASA gateway", "project-owned gateway", "api_key", "local"),
)


def provider_specs() -> tuple[ProviderSpec, ...]:
    return PROVIDER_SPECS


def login_profile_specs() -> tuple[LoginProfileSpec, ...]:
    return LOGIN_PROFILE_SPECS


def provider_ids(*, include_legacy: bool = False) -> tuple[str, ...]:
    ids = tuple(spec.provider_id for spec in PROVIDER_SPECS)
    if not include_legacy:
        return ids
    return (*ids, "oauth_gateway", "anthropic_gateway", "openclaw")


def provider_by_id(provider_id: str) -> ProviderSpec | None:
    normalized = provider_id.strip().lower()
    for spec in PROVIDER_SPECS:
        if spec.provider_id == normalized:
            return spec
    return None


def login_profile_by_id(profile_id: str) -> LoginProfileSpec | None:
    normalized = profile_id.strip().lower()
    for spec in LOGIN_PROFILE_SPECS:
        if spec.profile_id == normalized:
            return spec
    return None


def login_profile_by_provider(provider_id: str) -> LoginProfileSpec | None:
    normalized = provider_id.strip().lower()
    for spec in LOGIN_PROFILE_SPECS:
        if spec.provider_id == normalized:
            return spec
    return None


def login_companies() -> tuple[str, ...]:
    companies: list[str] = []
    for profile in LOGIN_PROFILE_SPECS:
        if profile.company not in companies:
            companies.append(profile.company)
    return tuple(companies)


def default_api_key_env(provider_id: str) -> str:
    spec = provider_by_id(provider_id)
    if spec is None:
        return MODEL_GATEWAY_TOKEN_ENV
    return spec.default_api_key_env


def default_auth_mode(provider_id: str) -> str:
    spec = provider_by_id(provider_id)
    if spec is None:
        return "gateway"
    return spec.default_auth_mode
