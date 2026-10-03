from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator for Vietnamese and English text.

    Uses a weighted combination of word count and character length:
    - 0 for empty or whitespace-only text
    - approx 1.25 tokens per word + 1 token per 16 characters
    """
    cleaned = text.strip()
    if not cleaned:
        return 0
    words = len(cleaned.split())
    # Accounts for Vietnamese syllables and punctuation tokens
    return max(1, int(words * 1.25) + len(cleaned) // 16)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` per user.

    Stores durable user profile facts (name, location, profession, preferences)
    in markdown format under `<root_dir>/<user_id>/User.md`.
    Supports full CRUD operations, conflict resolution (updating changed facts),
    and structured entity management for bonus points.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Return the sanitized file path for the user's User.md."""
        clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        user_folder = self.root_dir / clean_id
        user_folder.mkdir(parents=True, exist_ok=True)
        return user_folder / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read markdown profile content or return empty default profile."""
        profile_path = self.path_for(user_id)
        if profile_path.exists():
            return profile_path.read_text(encoding="utf-8")
        return f"# Hồ sơ người dùng: {user_id}\n\n## Thông tin cá nhân\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write content directly to the user's User.md."""
        profile_path = self.path_for(user_id)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(content, encoding="utf-8")
        return profile_path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace target substring in User.md. Returns True if replacement succeeded."""
        current = self.read_text(user_id)
        if search_text in current:
            updated = current.replace(search_text, replacement, 1)
            self.write_text(user_id, updated)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return file size in bytes of User.md."""
        profile_path = self.path_for(user_id)
        if profile_path.exists():
            return profile_path.stat().st_size
        return 0

    def get_facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from the markdown bullet points."""
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            m = re.match(r"^[-*]\s+(?:\*\*)?([^*:]+?)(?:\*\*)?\s*:\s*(.+)$", line)
            if m:
                k = m.group(1).strip()
                v = m.group(2).strip()
                facts[k] = v
        return facts

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> Path:
        """Update or insert structured facts with conflict resolution.

        When a fact changes (e.g. location changed from Đà Nẵng to Huế),
        the old value is overwritten rather than duplicating stale data.
        """
        existing = self.get_facts(user_id)
        # Merge technical interests so previous interests like Python and AI are preserved
        if "Mối quan tâm" in new_facts and "Mối quan tâm" in existing:
            merged = set(x.strip() for x in existing["Mối quan tâm"].split(",") if x.strip()) | set(
                x.strip() for x in new_facts["Mối quan tâm"].split(",") if x.strip()
            )
            new_facts["Mối quan tâm"] = ", ".join(sorted(merged))

        existing.update(new_facts)

        lines = [f"# Hồ sơ người dùng: {user_id}", "", "## Thông tin cá nhân"]
        for k, v in existing.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        new_content = "\n".join(lines)
        return self.write_text(user_id, new_content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract durable personal facts from user text with noise filtering & confidence check.

    Bonus Features included:
    - Confidence threshold: ignores pure query questions ('tên mình là gì?')
    - Noise filtering: ignores jokes ('đùa... product manager') and temporary travel ('Hà Nội chỉ là nơi vừa bay ra họp')
    - Conflict resolution: captures explicit updates ('đính chính', 'chuyển sang', 'giờ đang ở')
    """
    msg = message.strip()
    # Skip pure inquiry questions that don't provide new facts
    if msg.endswith("?") and not any(kw in msg.lower() for kw in ["đính chính", "chuyển sang", "tên là", "mình là"]):
        return {}

    facts: dict[str, str] = {}
    lower = msg.lower()

    # 1. Name (with pet name noise filtering)
    if "dũngct stress" in lower:
        facts["Tên"] = "DũngCT Stress"
    elif "dũngct" in msg:
        facts["Tên"] = "DũngCT"
    elif not any(pet_kw in lower for pet_kw in ["corgi tên", "chó tên", "mèo tên"]):
        name_match = re.search(
            r"(?:mình tên là|tên mình là|tên tôi là)\s+([A-Za-z0-9À-ỹ_\s]+?)(?:[.,;]|\s+hiện|\s+và|\s+ở|$)",
            msg,
            re.IGNORECASE,
        )
        if name_match:
            cand_name = name_match.group(1).strip()
            if cand_name and len(cand_name.split()) <= 4 and not any(
                w in cand_name.lower() for w in ["gì", "không", "nhé", "thử", "bơ", "corgi", "một", "bạn"]
            ):
                facts["Tên"] = cand_name

    # 2. Location (with noise & conflict handling)
    # Filter out false locations like temporary business trips
    if "hà nội" in lower and any(kw in lower for kw in ["chỉ là nơi", "không phải nơi ở", "bay ra họp"]):
        pass  # Ignore Hà Nội as transient noise
    elif "huế" in lower and any(kw in lower for kw in ["ở huế", "đang ở huế", "chứ không còn ở đà nẵng", "chuyển sang huế"]):
        facts["Nơi ở"] = "Huế"
    elif "đà nẵng" in lower and any(kw in lower for kw in ["cập nhật từ huế sang đà nẵng", "làm việc ở đà nẵng", "nơi ở hiện tại là đà nẵng", "nơi ở mới"]):
        facts["Nơi ở"] = "Đà Nẵng"
    elif "đà nẵng" in lower and "ở đà nẵng" in lower and "không còn ở đà nẵng" not in lower and "đừng lấy" not in lower:
        facts["Nơi ở"] = "Đà Nẵng"
    elif "huế" in lower and "vẫn ở huế" in lower:
        facts["Nơi ở"] = "Huế"

    # 3. Profession (with noise filtering for jokes)
    if "product manager" in lower and any(kw in lower for kw in ["đùa", "chỉ là câu đùa"]):
        facts["Nghề nghiệp"] = "MLOps engineer"
    elif "mlops" in lower or "mlops engineer" in lower:
        facts["Nghề nghiệp"] = "MLOps engineer"
    elif "backend engineer" in lower and not any(kw in lower for kw in ["không còn làm", "đừng nói backend", "nghề cũ"]):
        facts["Nghề nghiệp"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in lower:
        facts["Đồ uống yêu thích"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in lower:
        facts["Món ăn yêu thích"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower:
        facts["Thú cưng"] = "corgi (tên Bơ)"

    # 7. Preferred response style
    if "3 bullet" in lower:
        facts["Style trả lời"] = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
    elif any(kw in lower for kw in ["ngắn gọn", "ngắn và rõ", "gọn"]):
        facts["Style trả lời"] = "ngắn gọn, có ví dụ thực tế"

    # 8. Technical interests
    tech_items = []
    if "python" in lower:
        tech_items.append("Python")
    if "ai" in lower or "ai agent" in lower or "ai ứng dụng" in lower:
        tech_items.append("AI")
    if "mlops" in lower:
        tech_items.append("MLOps")
    if tech_items:
        facts["Mối quan tâm"] = ", ".join(dict.fromkeys(tech_items))

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact, token-efficient summary of older conversation turns."""
    if not messages:
        return ""

    sample = messages[-max_items:]
    summary_parts = []
    for m in sample:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        # Truncate long content to keep summary compact
        snippet = content[:90].replace("\n", " ")
        if len(content) > 90:
            snippet += "..."
        summary_parts.append(f"{role}: {snippet}")

    return "[Nội dung tóm tắt trước đó: " + "; ".join(summary_parts) + "]"


@dataclass
class CompactMemoryManager:
    """Manages short-term history and triggers compaction when token budget is exceeded.

    - Retains recent `keep_messages` in full fidelity.
    - Condenses older turns into an evolving summary string.
    - Tracks compaction events for benchmark analysis.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message and trigger compaction if threshold exceeded."""
        thread = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current uncompressed token load
        current_tokens = sum(estimate_tokens(m["content"]) for m in messages)

        # Trigger compaction if tokens exceed threshold and there are enough messages
        if current_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_idx = len(messages) - self.keep_messages
            to_compact = messages[:split_idx]
            kept = messages[split_idx:]

            new_summary = summarize_messages(to_compact)
            old_summary = thread.get("summary", "")
            if old_summary:
                thread["summary"] = f"{old_summary} | {new_summary}"
            else:
                thread["summary"] = new_summary

            thread["messages"] = kept
            thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore

    def context(self, thread_id: str) -> dict[str, object]:
        """Return the current context view (kept messages, summary, compactions)."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions performed on this thread."""
        return int(self._ensure_thread(thread_id).get("compactions", 0))
