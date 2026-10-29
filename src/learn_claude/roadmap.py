"""
Pedagogical Roadmap: 7-Stage Anthropic Claude Enterprise Masterclass.

Tracks curriculum progress across Architecture, Messages API, XML Prompting,
Prompt Caching, Tool Use, Computer Use, and Enterprise Multi-Cloud Deployments.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class StageStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


@dataclass
class Topic:
    id: str
    title: str
    summary: str
    completed: bool = False


@dataclass
class Stage:
    stage_number: int
    title: str
    description: str
    topics: List[Topic]
    status: StageStatus = StageStatus.NOT_STARTED

    @property
    def progress_percentage(self) -> float:
        if not self.topics:
            return 0.0
        completed = sum(1 for t in self.topics if t.completed)
        return (completed / len(self.topics)) * 100.0


class ClaudeRoadmap:
    """Manages progress and milestones for the 7-Stage Anthropic Claude curriculum."""

    def __init__(self) -> None:
        self.stages: Dict[int, Stage] = {
            1: Stage(
                stage_number=1,
                title="Architecture, Model Topology & The Messages API",
                description="Claude 3.5 Sonnet, 3.5 Haiku, 3 Opus topology, separate system prompts, and strict Messages API roles.",
                topics=[
                    Topic("1.1", "Model Topology & Intelligence Frontier", "Evaluating Claude 3.5 Sonnet vs Haiku vs Opus for coding, reasoning, and latency.", True),
                    Topic("1.2", "The Messages API Format", "System prompt top-level parameter, strict user/assistant alternating turns, and content block arrays.", True),
                    Topic("1.3", "Streaming & Event Parsing", "MessageStream, content_block_start, content_block_delta, and message_delta event cycles.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            2: Stage(
                stage_number=2,
                title="Advanced Prompt Engineering & XML Structural Formatting",
                description="Using XML tags (<instructions>, <context>, <scratchpad>) for semantic grounding, few-shot conditioning, and chain-of-thought.",
                topics=[
                    Topic("2.1", "XML Tag Semantic Disambiguation", "Structuring prompts with semantic boundaries (<documents>, <guidelines>, <rules>) to prevent prompt injection.", True),
                    Topic("2.2", "Prefilling Assistant Responses", "Seeding partial assistant messages ({'role': 'assistant', 'content': '```json'}) for deterministic formatting.", True),
                    Topic("2.3", "Extended Thinking & Scratchpads", "Guiding Claude to utilize <thinking> tags before generating production responses.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            3: Stage(
                stage_number=3,
                title="90% Cost Reduction with Prompt Caching",
                description="Ephemeral cache control blocks, 5-minute TTL, 4-breakpoint limit, and token cost economics (90% read discount).",
                topics=[
                    Topic("3.1", "Prompt Caching Architecture", "How KV cache state is preserved across requests using cache_control: {'type': 'ephemeral'}.", True),
                    Topic("3.2", "Cache Breakpoint Placement", "Strategic placement on static system instructions, reference documents, and tool definitions.", True),
                    Topic("3.3", "Token Cost Modeling", "Analyzing write premium (+25%) vs read discount (-90%) and break-even thresholds (>=2 requests).", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            4: Stage(
                stage_number=4,
                title="Tool Use (Function Calling) & Structured Extraction",
                description="Declaring tools, handling tool_use blocks, executing tool code, and returning tool_result content blocks.",
                topics=[
                    Topic("4.1", "Tool Definition & Schemas", "Defining input_schema with strict JSON schemas and descriptive property guidelines.", True),
                    Topic("4.2", "Tool Choice Policies", "Configuring auto, any, and forced tool choice ({'type': 'tool', 'name': '...'}) modes.", True),
                    Topic("4.3", "Tool Execution & Feedback Loop", "Multi-turn dispatch pipeline submitting tool_result blocks with error handling.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            5: Stage(
                stage_number=5,
                title="The Computer Use API & Multimodal Vision",
                description="Anthropic's beta Computer Use API (mouse, keyboard, bash, screenshot) and multimodal document/image inspection.",
                topics=[
                    Topic("5.1", "Computer Use Architecture", "Enabling anthropic-beta header, coordinate space normalization, and screen capture.", True),
                    Topic("5.2", "Action Dispatch Loop", "Handling mouse_move, left_click, type, key, and bash execution requests safely.", True),
                    Topic("5.3", "Multimodal Vision Reasoning", "Processing high-resolution screenshots, PDFs, charts, and technical diagrams.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            6: Stage(
                stage_number=6,
                title="Enterprise Deployments: AWS Bedrock, GCP Vertex & Message Batches",
                description="Multi-cloud enterprise hosting (Bedrock Converse API, Vertex AI Claude) and async Message Batches at 50% discount.",
                topics=[
                    Topic("6.1", "AWS Bedrock & GCP Vertex Integrations", "IAM authentication, enterprise VPC boundaries, and cross-cloud failover routing.", True),
                    Topic("6.2", "Message Batches API", "High-throughput asynchronous 24-hour batch processing at 50% token cost reduction.", True),
                    Topic("6.3", "Rate Limits & Resilience", "Handling 429 throttling, jittered exponential backoff, and tenant quota management.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            7: Stage(
                stage_number=7,
                title="Staff-Level Interview Prep, Cheatsheet & Appendix",
                description="Architectural decision frameworks, incident runbooks, and staff interview technical blueprints.",
                topics=[
                    Topic("7.1", "Architectural Comparisons", "Claude vs OpenAI vs Open-Source trade-offs in reasoning, cost, and context limits.", True),
                    Topic("7.2", "Production Incident Runbooks", "Mitigating model degradation, token spikes, and upstream provider outages.", True),
                    Topic("7.3", "Master Interview Q&A", "Deep-dive answers to 40+ staff-level prompt engineering, caching, and systems questions.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
        }

    def get_stage(self, stage_number: int) -> Optional[Stage]:
        return self.stages.get(stage_number)

    def total_progress(self) -> float:
        total_topics = sum(len(s.topics) for s in self.stages.values())
        if total_topics == 0:
            return 0.0
        completed = sum(sum(1 for t in s.topics if t.completed) for s in self.stages.values())
        return (completed / total_topics) * 100.0

    def get_summary(self) -> Dict[str, object]:
        return {
            "total_stages": len(self.stages),
            "completed_stages": sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED),
            "overall_progress_pct": round(self.total_progress(), 2),
            "stages": [
                {
                    "stage": s.stage_number,
                    "title": s.title,
                    "status": s.status.value,
                    "progress_pct": round(s.progress_percentage, 1),
                    "topics_count": len(s.topics),
                }
                for s in self.stages.values()
            ],
        }
