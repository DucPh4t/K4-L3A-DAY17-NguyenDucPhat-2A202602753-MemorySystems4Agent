from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from tabulate import tabulate


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversation dataset from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return proportion of expected keywords that appear in answer (0.0 to 1.0)."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Score response quality on a 0.0 - 1.0 scale.

    Factors:
    - Factual accuracy (recall ratio): 70% weight
    - Fluency & structure (non-empty, bullet/sentence structure): 30% weight
    """
    ans = answer.strip()
    if not ans:
        return 0.0

    acc = recall_points(ans, expected)
    struct_score = 0.5
    if any(marker in ans for marker in ["- ", "• ", "\n"]):
        struct_score += 0.3
    if len(ans) >= 30:
        struct_score += 0.2

    struct_score = min(1.0, struct_score)
    return round(0.7 * acc + 0.3 * struct_score, 2)


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations and recall questions."""
    all_threads: list[str] = []
    user_ids: set[str] = set()

    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    # 1. Feed regular dialogue turns into the agent
    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        user_ids.add(user_id)
        all_threads.append(conv_id)

        for turn in conv.get("turns", []):
            res = agent.reply(user_id=user_id, thread_id=conv_id, message=turn)
            total_agent_tokens += res.get("tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)

        # 2. Ask recall questions in FRESH independent threads
        for q_idx, q in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv_id}_recall_{q_idx}"
            all_threads.append(recall_thread)
            question = q["question"]
            expected = q["expected_contains"]

            res = agent.reply(user_id=user_id, thread_id=recall_thread, message=question)
            total_agent_tokens += res.get("tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)

            score = recall_points(res["reply"], expected)
            qual = heuristic_quality(res["reply"], expected)
            recall_scores.append(score)
            quality_scores.append(qual)

    # 3. Calculate metrics
    avg_recall = (sum(recall_scores) / len(recall_scores)) * 100.0 if recall_scores else 0.0
    avg_quality = (sum(quality_scores) / len(quality_scores)) * 100.0 if quality_scores else 0.0

    # Memory growth
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        for uid in user_ids:
            memory_growth += agent.memory_file_size(uid)

    # Compactions
    total_compactions = sum(agent.compaction_count(tid) for tid in all_threads)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 1),
        response_quality=round(avg_quality, 1),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean Markdown table with the 6 mandatory columns."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                f"{r.agent_tokens_only:,}",
                f"{r.prompt_tokens_processed:,}",
                f"{r.recall_score}%",
                f"{r.response_quality}%",
                f"{r.memory_growth_bytes:,} B",
                r.compactions,
            ]
        )
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both Standard Benchmark and Long-Context Stress Benchmark."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    print("================================================================================")
    print("PHASE 2 - TRACK 3 - DAY 17: MEMORY SYSTEMS FOR AI AGENT BENCHMARK")
    print("================================================================================\n")

    # -------------------------------------------------------------------------
    # Suite 1: Standard Benchmark (Cross-session recall across 10 conversations)
    # -------------------------------------------------------------------------
    print(">>> 1. RUNNING STANDARD BENCHMARK (data/conversations.json)...")
    std_convs = load_conversations(std_data_path)

    baseline_std = BaselineAgent(config, force_offline=True)
    row_baseline_std = run_agent_benchmark("Baseline Agent", baseline_std, std_convs, config)

    advanced_std = AdvancedAgent(config, force_offline=True)
    row_advanced_std = run_agent_benchmark("Advanced Agent", advanced_std, std_convs, config)

    print("\n### Standard Benchmark Results:")
    print(format_rows([row_baseline_std, row_advanced_std]))
    print()

    # -------------------------------------------------------------------------
    # Suite 2: Long-Context Stress Benchmark (Stress test with 16 heavy turns)
    # -------------------------------------------------------------------------
    print(">>> 2. RUNNING LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)...")
    stress_convs = load_conversations(stress_data_path)

    baseline_stress = BaselineAgent(config, force_offline=True)
    row_baseline_stress = run_agent_benchmark("Baseline Agent", baseline_stress, stress_convs, config)

    advanced_stress = AdvancedAgent(config, force_offline=True)
    row_advanced_stress = run_agent_benchmark("Advanced Agent", advanced_stress, stress_convs, config)

    print("\n### Long-Context Stress Benchmark Results:")
    print(format_rows([row_baseline_stress, row_advanced_stress]))
    print()

    # -------------------------------------------------------------------------
    # Analysis & Insights
    # -------------------------------------------------------------------------
    print("================================================================================")
    print("KEY INSIGHTS & MEMORY ARCHITECTURE ANALYSIS:")
    print("================================================================================")
    print(
        "1. Cross-Session Recall:\n"
        f"   - Baseline Agent đạt {row_baseline_std.recall_score}% vì chỉ lưu trong phiên (thread_id).\n"
        f"     Khi sang phiên mới, Baseline quên hoàn toàn thông tin cá nhân.\n"
        f"   - Advanced Agent đạt {row_advanced_std.recall_score}% nhờ User.md lưu bền vững trên đĩa.\n"
    )
    print(
        "2. Long-Context Prompt Token Efficiency (Stress Benchmark):\n"
        f"   - Baseline Agent tốn {row_baseline_stress.prompt_tokens_processed:,} prompt tokens do mang toàn bộ lịch sử thô.\n"
        f"   - Advanced Agent chỉ tốn {row_advanced_stress.prompt_tokens_processed:,} prompt tokens (giảm tải rõ rệt)\n"
        f"     nhờ CompactMemoryManager kích hoạt {row_advanced_stress.compactions} lần compaction.\n"
    )
    print(
        "3. Trade-offs:\n"
        "   - Ở các hội thoại ngắn, Advanced Agent có overhead ban đầu do inject User.md vào prompt.\n"
        "   - Nhưng ở hội thoại dài, Compact Memory kéo chi phí prompt context xuống theo thời gian,\n"
        "     giúp kiểm soát ngân sách LLM mà vẫn bảo toàn facts dài hạn.\n"
    )


if __name__ == "__main__":
    main()
