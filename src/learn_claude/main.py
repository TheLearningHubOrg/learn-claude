"""
Anthropic Claude Enterprise Masterclass: CLI & Demonstrator.

Demonstrates:
1. 7-Stage Curriculum progress tracking.
2. Messages API with XML structured tags and prompt boundaries.
3. Ephemeral Prompt Caching with 90% token read discount.
4. Tool Use loop with automatic tool execution and tool_result feedback blocks.
5. Computer Use action parsing and execution simulation.
6. Multi-model cost economics across Sonnet, Haiku, and Opus.
"""

import json
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


def run_roadmap_showcase() -> None:
    roadmap = ClaudeRoadmap()
    summary = roadmap.get_summary()

    print("=" * 80)
    print("ANTHROPIC CLAUDE ENTERPRISE MASTERCLASS: 7-STAGE CURRICULUM")
    print("=" * 80)
    print(f"Curriculum Completion: {summary['overall_progress_pct']}% | Stages: {summary['completed_stages']}/{summary['total_stages']} Completed\n")

    for s in summary["stages"]:
        status_symbol = "[x]" if s["status"] == "COMPLETED" else "[ ]"
        print(f"  {status_symbol} Stage {s['stage']}: {s['title']} ({s['topics_count']} topics, {s['progress_pct']}%)")
    print("=" * 80 + "\n")


def run_xml_prompting_demo(client: AnthropicClient) -> None:
    print("-" * 80)
    print("STAGE 2: XML STRUCTURAL PROMPT FORMATTING & SEMANTIC BOUNDARIES")
    print("-" * 80)

    system_prompt = (
        "<instructions>\n"
        "You are a principal staff engineer conducting an architectural audit.\n"
        "Analyze the provided architecture specification within <architecture_spec> tags.\n"
        "Structure your output using <analysis>, <vulnerabilities>, and <recommendations> tags.\n"
        "</instructions>"
    )

    user_prompt = (
        "<architecture_spec>\n"
        "API Gateway routes to 12 stateless microservices on EKS.\n"
        "Primary database is Aurora PostgreSQL Multi-AZ with Redis write-through cache.\n"
        "Cross-region disaster recovery uses async WAL replication.\n"
        "</architecture_spec>\n\n"
        "Please conduct the audit."
    )

    response = client.create_message(
        model="claude-3-5-sonnet-20241022",
        system=system_prompt,
        messages=[MessageParam(role="user", content=user_prompt)],
    )

    print("Request dispatched with XML semantic tags.")
    print(f"Model: {response.model} | Stop Reason: {response.stop_reason}")
    print(f"Output: {response.content[0].text if response.content else ''}")
    print(f"Token Usage: Input={response.usage.input_tokens}, Output={response.usage.output_tokens}\n")


def run_prompt_caching_demo(client: AnthropicClient) -> None:
    print("-" * 80)
    print("STAGE 3: EPHEMERAL PROMPT CACHING (90% COST REDUCTION)")
    print("-" * 80)

    # Static system knowledge base cached with cache_control: {"type": "ephemeral"}
    cached_system_block = TextBlock(
        text=(
            "<company_policy>\n"
            "All enterprise microservices must implement mTLS with SPIFFE/SPIRE identity.\n"
            "Data at rest must be encrypted with AWS KMS customer-managed keys.\n"
            "All API endpoints require JWT authentication verified against Okta OIDC.\n"
            "</company_policy>"
        ),
        cache_control=PromptCacheBlock(type="ephemeral"),
    )

    response = client.create_message(
        model="claude-3-5-sonnet-20241022",
        system=[cached_system_block],
        messages=[
            MessageParam(role="user", content="Verify if public endpoints can bypass mTLS under any circumstance.")
        ],
    )

    print("Dispatched message with ephemeral cache breakpoint on system instructions.")
    print(f"  Input Tokens:        {response.usage.input_tokens}")
    print(f"  Cache Read Tokens:   {response.usage.cache_read_input_tokens} (Billed at 90% discount: $0.30/1M)")
    print(f"  Output Tokens:       {response.usage.output_tokens}\n")


def run_tool_use_demo(client: AnthropicClient) -> None:
    print("-" * 80)
    print("STAGE 4: TOOL USE & TOOL_RESULT FEEDBACK LOOP")
    print("-" * 80)

    vpc_tool = ToolParam(
        name="describe_vpc_routes",
        description="Retrieves active routing table entries for an AWS VPC ID.",
        input_schema={
            "type": "object",
            "properties": {
                "vpc_id": {"type": "string"},
            },
            "required": ["vpc_id"],
        },
    )

    handlers = {
        "describe_vpc_routes": lambda args: json.dumps({
            "vpc_id": args.get("vpc_id", "vpc-0a1b2c3d4e"),
            "routes": [
                {"destination": "10.0.0.0/16", "target": "local"},
                {"destination": "0.0.0.0/0", "target": "nat-0987654321fedcba"},
            ],
            "nat_gateway_state": "available",
        })
    }

    conversation = [
        MessageParam(role="user", content="Please call describe_vpc_routes for vpc-production-01.")
    ]

    history = client.run_tool_loop(
        model="claude-3-5-sonnet-20241022",
        messages=conversation,
        tools=[vpc_tool],
        tool_handlers=handlers,
    )

    print(f"Multi-turn tool loop completed across {len(history)} messages:")
    for idx, msg in enumerate(history, 1):
        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    print(f"  Turn {idx} [ASSISTANT]: Tool Use -> {block.name}({block.input})")
                elif isinstance(block, ToolResultBlock):
                    print(f"  Turn {idx} [USER]: Tool Result -> {block.content}")
                elif isinstance(block, TextBlock):
                    print(f"  Turn {idx} [ASSISTANT]: {block.text}")
        else:
            print(f"  Turn {idx} [{msg.role.upper()}]: {msg.content}")
    print()


def run_cost_economics_demo() -> None:
    print("-" * 80)
    print("STAGE 3 & 6: ENTERPRISE CLAUDE TOKEN ECONOMICS (100,000 REQUESTS)")
    print("-" * 80)

    # 100k requests with 4,000 input tokens (3,000 cached read) and 600 output tokens
    reqs = 100_000
    total_input = reqs * 4000
    cache_read = reqs * 3000
    total_output = reqs * 600

    print(f"Workload: {reqs:,} requests (4,000 input tokens with 75% cache hit rate, 600 output tokens)\n")
    print(f"{'Model':<28} | {'Baseline Cost':<15} | {'With Cache (-90%)':<20} | {'Savings':<15}")
    print("-" * 82)

    for model in ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]:
        std = ClaudeCostCalculator.calculate_cost(model, total_input, total_output, cache_read_tokens=0)
        cached = ClaudeCostCalculator.calculate_cost(model, total_input, total_output, cache_read_tokens=cache_read)

        print(f"{model:<28} | ${std['total_cost']:<14.2f} | ${cached['total_cost']:<19.2f} | ${cached['savings_usd']:<8.2f} ({cached['savings_pct']}%)")
    print()


def main() -> None:
    client = AnthropicClient()
    run_roadmap_showcase()
    run_xml_prompting_demo(client)
    run_prompt_caching_demo(client)
    run_tool_use_demo(client)
    run_cost_economics_demo()


if __name__ == "__main__":
    main()
