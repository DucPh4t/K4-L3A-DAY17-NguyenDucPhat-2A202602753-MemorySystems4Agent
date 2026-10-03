from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline Agent (Agent A).

    Characteristics:
    - Short-term, within-session memory only (keyed strictly by thread_id).
    - No persistent User.md profile storage across threads.
    - No compaction mechanism: as the thread grows, the entire history is sent
      in the prompt, causing prompt tokens to scale linearly or quadratically.
    - Completely forgets cross-session facts when a new thread_id is used.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live agent or deterministic offline mode."""
        if self.langchain_agent and not self.force_offline:
            try:
                # Live LangChain execution
                session = self._get_session(thread_id)
                prompt_tokens = self._estimate_prompt_tokens(thread_id, message)
                session.prompt_tokens_processed += prompt_tokens

                resp = self.langchain_agent.invoke(
                    {"messages": [("user", message)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                text = resp["messages"][-1].content
                out_tokens = estimate_tokens(text)
                session.token_usage += out_tokens
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": text})
                return {"reply": text, "tokens": out_tokens, "prompt_tokens": prompt_tokens}
            except Exception:
                # Fallback to offline on network/quota failure
                pass

        return self._reply_offline(thread_id, message)

    def _estimate_prompt_tokens(self, thread_id: str, current_message: str) -> int:
        session = self._get_session(thread_id)
        # Baseline concatenates full session history into each turn prompt
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in session.messages)
        full_prompt = f"System: Bạn là trợ lý AI hữu ích.\n{history_text}\nUser: {current_message}\nAssistant:"
        return estimate_tokens(full_prompt)

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._get_session(thread_id)

        # 1. Prompt token accounting (full uncompacted history + current message)
        prompt_tokens = self._estimate_prompt_tokens(thread_id, message)
        session.prompt_tokens_processed += prompt_tokens

        # 2. Baseline logic: only aware of current thread's messages
        # When asked recall questions in a new session (fresh thread_id),
        # session.messages is empty, so baseline cannot recall past facts.
        lower_msg = message.lower()
        if not session.messages:
            # First turn of a fresh thread
            if any(kw in lower_msg for kw in ["tên gì", "ở đâu", "nghề gì", "nhắc lại", "đồ uống", "món ăn"]):
                reply_text = "Xin chào! Mình chưa có thông tin trước đó trong phiên này nên không rõ thông tin cá nhân của bạn."
            else:
                reply_text = f"Chào bạn! Mình đã nhận được tin nhắn trong phiên mới: '{message[:35]}...'."
        else:
            # Ongoing thread
            reply_text = f"Đã ghi nhận trong phiên hiện tại: '{message[:40]}'."

        # 3. Output token accounting
        out_tokens = estimate_tokens(reply_text)
        session.token_usage += out_tokens

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": out_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative generated response tokens for thread_id."""
        return self._get_session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt tokens processed for thread_id."""
        return self._get_session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline agent has no compact memory."""
        return 0

    def _maybe_build_langchain_agent(self):
        """Optionally build LangChain runnable with in-memory checkpointer."""
        if not self.config.model.api_key and self.config.model.provider != "ollama":
            return None
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            memory = MemorySaver()
            return create_react_agent(llm, tools=[], checkpointer=memory)
        except Exception:
            return None
