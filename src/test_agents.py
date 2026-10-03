from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated configuration for deterministic unit tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,  # Lower threshold for immediate compaction in tests
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify that User.md can be created, read, edited, and queried for size."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user_01"

    # Default read when not created yet
    initial_text = store.read_text(user_id)
    assert user_id in initial_text

    # Write initial profile
    content = "# Hồ sơ: test_user_01\n- **Tên**: Nguyễn Văn A\n- **Nơi ở**: Đà Nẵng\n"
    store.write_text(user_id, content)
    assert store.read_text(user_id) == content
    assert store.file_size(user_id) > 0

    # Edit profile (change location to Huế)
    success = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert success is True
    updated = store.read_text(user_id)
    assert "Huế" in updated
    assert "Đà Nẵng" not in updated

    # Edit non-existent text returns False
    failed = store.edit_text(user_id, "NonExistentKey", "Value")
    assert failed is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify that CompactMemoryManager compacts when tokens exceed threshold."""
    manager = CompactMemoryManager(threshold_tokens=30, keep_messages=2)
    thread_id = "thread_compact_test"

    # Turn 1
    manager.append(thread_id, "user", "Tin nhắn số một có độ dài tương đối để tích lũy token.")
    manager.append(thread_id, "assistant", "Phản hồi số một từ trợ lý.")
    assert manager.compaction_count(thread_id) == 0

    # Turn 2
    manager.append(thread_id, "user", "Tin nhắn số hai dài hơn tiếp tục đẩy tổng token lên cao hơn ngưỡng.")
    manager.append(thread_id, "assistant", "Phản hồi số hai ghi nhận thông tin.")

    # Threshold (30 tokens) exceeded and message count > keep_messages (2)
    assert manager.compaction_count(thread_id) >= 1

    ctx = manager.context(thread_id)
    # Kept messages should not exceed keep_messages
    assert len(ctx["messages"]) <= 2
    # Summary should contain compacted older context
    assert len(str(ctx["summary"])) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced Agent remembers facts across threads while Baseline forgets."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "user_cross_test"
    thread_1 = "session_01"
    thread_2 = "session_02"  # New independent session

    # Session 1: User introduces themselves
    intro = "Chào bạn, mình tên là DũngCT, ở Huế và đồ uống yêu thích là cà phê sữa đá."
    baseline.reply(user_id=user_id, thread_id=thread_1, message=intro)
    advanced.reply(user_id=user_id, thread_id=thread_1, message=intro)

    # Session 2: User asks recall question in a completely new thread
    recall_q = "Mình tên gì và đồ uống yêu thích là gì?"
    base_res = baseline.reply(user_id=user_id, thread_id=thread_2, message=recall_q)
    adv_res = advanced.reply(user_id=user_id, thread_id=thread_2, message=recall_q)

    # Baseline has no cross-session memory
    assert "DũngCT" not in base_res["reply"]
    assert "cà phê sữa đá" not in base_res["reply"]

    # Advanced recalls perfectly from persistent User.md
    assert "DũngCT" in adv_res["reply"]
    assert "cà phê sữa đá" in adv_res["reply"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Verify compact memory caps prompt token explosion compared to uncompacted baseline."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    thread_id = "thread_stress_load"
    user_id = "stress_user"

    long_paragraph = (
        "Chi tiết kỹ thuật về hệ thống phân tích Artemis III, X-59 siêu thanh và báo cáo biến đổi khí hậu WMO. "
        "Đây là phần kiểm thử tải ngữ cảnh dài nhằm mô phỏng các hội thoại sâu với nhiều tham số và phân tích logic. "
    )
    turns = [f"Lượt hội thoại thứ {i}: {long_paragraph}" for i in range(12)]

    for turn in turns:
        baseline.reply(user_id=user_id, thread_id=thread_id, message=turn)
        advanced.reply(user_id=user_id, thread_id=thread_id, message=turn)

    assert advanced.compaction_count(thread_id) > 0

    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Over long sessions, compaction keeps context bounded
    assert adv_prompt_tokens < base_prompt_tokens


def test_bonus_conflict_handling_and_noise_filtering(tmp_path: Path) -> None:
    """Verify bonus features: conflict resolution and joke/transient noise rejection."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "bonus_user"

    # Step 1: Initial fact
    up1 = extract_profile_updates("Chào bạn, mình tên là DũngCT, đang ở Đà Nẵng và làm backend engineer.")
    store.upsert_facts(user_id, up1)
    facts1 = store.get_facts(user_id)
    assert facts1["Nơi ở"] == "Đà Nẵng"
    assert facts1["Nghề nghiệp"] == "backend engineer"

    # Step 2: Conflict resolution / Correction
    up2 = extract_profile_updates("Mình đính chính nhé: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa, và chuyển sang MLOps engineer.")
    store.upsert_facts(user_id, up2)
    facts2 = store.get_facts(user_id)
    assert facts2["Nơi ở"] == "Huế"
    assert facts2["Nghề nghiệp"] == "MLOps engineer"

    # Step 3: Noise filtering (joke about product manager)
    up3 = extract_profile_updates("Mình đùa với bạn là hay chuyển sang product manager, nhưng đó chỉ là câu đùa. Nghề nghiệp hiện tại vẫn là MLOps engineer.")
    store.upsert_facts(user_id, up3)
    facts3 = store.get_facts(user_id)
    assert facts3["Nghề nghiệp"] == "MLOps engineer"

    # Step 4: Transient travel noise (business trip to Hà Nội)
    up4 = extract_profile_updates("Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại.")
    store.upsert_facts(user_id, up4)
    facts4 = store.get_facts(user_id)
    assert facts4["Nơi ở"] == "Huế"
