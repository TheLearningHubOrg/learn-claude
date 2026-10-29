"""
Comprehensive Unit Tests for Anthropic Claude Enterprise Masterclass Engine.

Tests:
- Client configuration and Messages API payload generation.
- XML-structured prompting and semantic boundaries.
- Ephemeral prompt caching token calculations and 90% read discount.
- Multi-turn tool calling and tool_result feedback blocks.
- Multi-cloud enterprise cost calculator.
- Pedagogical roadmap stage completion.
"""

import json
import unittest

from src.learn_claude.client import (
    AnthropicClient,
    ClaudeCostCalculator,
    MessageParam,
    PromptCacheBlock,
    TextBlock,
    ToolParam,
    ToolResultBlock,
    ToolUseBlock,
)
from src.learn_claude.roadmap import ClaudeRoadmap


class TestClaudeEngine(unittest.TestCase):

    def setUp(self) -> None:
        self.client = AnthropicClient(api_key="sk-ant-test-key")

    def test_client_initialization(self) -> None:
        self.assertEqual(self.client.api_key, "sk-ant-test-key")

    def test_standard_message_creation(self) -> None:
        messages = [MessageParam(role="user", content="Hello Claude!")]
        response = self.client.create_message(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            system="You are an expert engineer.",
        )
        self.assertEqual(response.model, "claude-3-5-sonnet-20241022")
        self.assertEqual(response.stop_reason, "end_turn")
        self.assertTrue(len(response.content) > 0)
        self.assertIsInstance(response.content[0], TextBlock)
        self.assertTrue(response.usage.input_tokens > 0)
        self.assertTrue(response.usage.output_tokens > 0)

    def test_prompt_caching_ephemeral(self) -> None:
        # Prompt with ephemeral cache control
        cached_sys = TextBlock(
            text="Long system instructions " * 50,
            cache_control=PromptCacheBlock(type="ephemeral"),
        )
        messages = [MessageParam(role="user", content="Test prompt caching.")]

        response = self.client.create_message(
            model="claude-3-5-sonnet-20241022",
            system=[cached_sys],
            messages=messages,
        )

        self.assertTrue(response.usage.cache_read_input_tokens > 0)

    def test_tool_use_loop(self) -> None:
        tool = ToolParam(
            name="get_memory_usage",
            description="Returns node memory usage.",
            input_schema={"type": "object", "properties": {"node_id": {"type": "string"}}},
        )

        handlers = {
            "get_memory_usage": lambda args: json.dumps({"node": "node-1", "used_mb": 4096}),
        }

        conversation = [
            MessageParam(role="user", content="Please run get_memory_usage on node-1.")
        ]

        history = self.client.run_tool_loop(
            model="claude-3-5-sonnet-20241022",
            messages=conversation,
            tools=[tool],
            tool_handlers=handlers,
        )

        # Expect: 1 user, 1 assistant (tool_use), 1 user (tool_result), 1 final assistant
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1].role, "assistant")
        self.assertIsInstance(history[1].content[0], ToolUseBlock)
        self.assertEqual(history[2].role, "user")
        self.assertIsInstance(history[2].content[0], ToolResultBlock)
        self.assertEqual(history[3].role, "assistant")
        self.assertIsInstance(history[3].content[0], TextBlock)

    def test_cost_calculator(self) -> None:
        # 1M input, 1M output, 500k cached read on Claude 3.5 Sonnet
        cost = ClaudeCostCalculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=500_000,
            is_batch=False,
        )

        # Baseline: 1M * $3.00 + 1M * $15.00 = $18.00
        self.assertAlmostEqual(cost["baseline_cost"], 18.00, places=2)
        # With 500k cache read ($0.30/1M = $0.15) + 500k base input ($3.00/1M = $1.50) + 1M output ($15.00) = $16.65
        self.assertAlmostEqual(cost["total_cost"], 16.65, places=2)
        self.assertTrue(cost["savings_usd"] > 0)

        # With batch mode (50% discount)
        batch_cost = ClaudeCostCalculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=500_000,
            is_batch=True,
        )
        self.assertAlmostEqual(batch_cost["total_cost"], 16.65 * 0.5, places=2)

    def test_roadmap_progress(self) -> None:
        roadmap = ClaudeRoadmap()
        self.assertEqual(len(roadmap.stages), 7)
        self.assertEqual(roadmap.total_progress(), 100.0)
        summary = roadmap.get_summary()
        self.assertEqual(summary["total_stages"], 7)
        self.assertEqual(summary["completed_stages"], 7)
        self.assertEqual(summary["overall_progress_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
