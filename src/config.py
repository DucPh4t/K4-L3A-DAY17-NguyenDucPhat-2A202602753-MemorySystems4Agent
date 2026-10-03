from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the Day 17 Memory Systems lab.

    Attributes:
        base_dir: Root directory of the repository.
        data_dir: Path to benchmark datasets (conversations.json, etc.).
        state_dir: Path to runtime state (User.md profiles, logs).
        compact_threshold_tokens: Token count threshold to trigger compaction.
        compact_keep_messages: Number of most recent messages to keep uncompressed.
        model: ProviderConfig for the primary agent LLM.
        judge_model: ProviderConfig for the evaluation judge LLM.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load configuration from environment variables and default paths."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env if present
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Directories
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    # Provider & model settings
    raw_provider = os.getenv("LLM_PROVIDER", "openai")
    provider = normalize_provider(raw_provider)
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Resolve API key based on provider
    api_key = None
    base_url = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")

    agent_model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    judge_model = ProviderConfig(
        provider=os.getenv("JUDGE_PROVIDER", provider),
        model_name=os.getenv("JUDGE_MODEL", model_name),
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "120"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=agent_model,
        judge_model=judge_model,
    )
