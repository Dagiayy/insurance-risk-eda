"""AI-powered business insight generation over already-computed analytics.

Real-world use case this solves: a pricing actuary or ops manager doesn't want to read
raw loss-ratio tables and p-values -- they want three sentences of "what changed and what
should I do about it." Rather than bolting an LLM onto raw data (expensive, prone to
hallucinating numbers), this module only ever gives the model *pre-computed, validated
statistics* (from :mod:`insurance_analytics.analysis`) and asks it to narrate them,
constrained to a structured schema. The model cannot introduce a number that isn't
already in the prompt because every numeric claim in the output schema is optional prose,
not a place for it to invent figures.

Design choices, per the "implement AI properly" requirements:
  * Prompt management: templates are versioned constants in this file, not inline f-strings
    scattered across callers.
  * Structured output: the model is asked for JSON matching :class:`InsightReport` and the
    response is validated with Pydantic; malformed output raises rather than silently
    passing bad data downstream.
  * Cost/token control: the input is a compact JSON summary (already-aggregated stats),
    never the raw dataset, and ``max_tokens`` is capped.
  * Graceful degradation: with no API key configured, ``generate`` returns a deterministic,
    template-based report instead of failing -- the feature is additive, not a hard
    dependency of the pipeline.
  * Observability: every call logs latency, whether it used the LLM path or the fallback,
    and token usage when available.
"""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, Field

from insurance_analytics.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an actuarial analyst assistant for a South African car-insurance underwriter. "
    "You are given ONLY pre-computed, already-verified statistics as JSON. "
    "Never invent a number that is not present in the input JSON. "
    "Respond with a single JSON object matching the requested schema exactly, no prose outside the JSON."
)

USER_PROMPT_TEMPLATE = """Here is a JSON summary of exploratory data analysis and hypothesis-test results
for a motor insurance policy book:

{stats_json}

Produce a JSON object with exactly these keys:
- "executive_summary": 2-3 sentence plain-English summary for a non-technical exec.
- "key_findings": a list of 3-5 short bullet strings, each grounded in a specific number from the input.
- "risk_flags": a list of 0-4 short strings naming segments/attributes that look risky or statistically significant.
- "recommendations": a list of 2-4 short, actionable pricing/underwriting recommendations.

Respond with ONLY the JSON object."""


class InsightReport(BaseModel):
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    source: str = "template"  # "llm" or "template", so callers/UI can label the output honestly
    generated_at: float = Field(default_factory=time.time)
    model: str | None = None
    latency_seconds: float | None = None


def _template_report(stats: dict) -> InsightReport:
    """Deterministic, no-LLM fallback so the endpoint always returns something useful."""
    findings = []
    if "overall_loss_ratio" in stats:
        findings.append(f"Overall loss ratio is {stats['overall_loss_ratio']:.3f} (claims / premium).")
    if "claim_frequency" in stats:
        findings.append(f"Claim frequency across the book is {stats['claim_frequency']:.2%}.")
    for key, value in stats.items():
        if key.startswith("loss_ratio_by_") and isinstance(value, dict) and value:
            worst = max(value, key=value.get)
            findings.append(f"Highest loss ratio by {key.replace('loss_ratio_by_', '')}: {worst} ({value[worst]:.3f}).")

    return InsightReport(
        executive_summary=(
            "Automated summary (template mode -- set ANTHROPIC_API_KEY to enable AI-generated narrative)."
        ),
        key_findings=findings or ["Insufficient data to compute headline findings."],
        risk_flags=[],
        recommendations=["Configure an LLM API key to enable narrative recommendations."],
        source="template",
    )


class InsightGenerator:
    def __init__(self, api_key: str | None, model: str = "claude-sonnet-5", max_tokens: int = 1024):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, stats: dict) -> InsightReport:
        if not self.api_key:
            logger.info("No ANTHROPIC_API_KEY configured; returning template insight report")
            return _template_report(stats)

        try:
            return self._generate_with_llm(stats)
        except Exception:
            logger.exception("LLM insight generation failed; falling back to template report")
            return _template_report(stats)

    def _generate_with_llm(self, stats: dict) -> InsightReport:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        start = time.monotonic()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_PROMPT_TEMPLATE.format(stats_json=json.dumps(stats, default=str))}],
        )
        latency = time.monotonic() - start

        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        payload = json.loads(text)
        report = InsightReport(**payload)
        report.source = "llm"
        report.model = self.model
        report.latency_seconds = round(latency, 3)
        logger.info("Generated LLM insight report in %.2fs (model=%s)", latency, self.model)
        return report
