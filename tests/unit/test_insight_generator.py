from unittest.mock import MagicMock, patch

from insurance_analytics.ai.insight_generator import InsightGenerator, InsightReport


def test_generate_without_api_key_returns_template_report():
    generator = InsightGenerator(api_key=None)
    report = generator.generate({"overall_loss_ratio": 0.6, "claim_frequency": 0.05})

    assert report.source == "template"
    assert isinstance(report, InsightReport)
    assert len(report.key_findings) > 0


def test_generate_falls_back_to_template_on_llm_error():
    generator = InsightGenerator(api_key="fake-key")
    with patch.object(InsightGenerator, "_generate_with_llm", side_effect=RuntimeError("boom")):
        report = generator.generate({"overall_loss_ratio": 0.6})
    assert report.source == "template"


def test_generate_with_llm_parses_structured_json_response():
    generator = InsightGenerator(api_key="fake-key", model="claude-sonnet-5")

    fake_text_block = MagicMock()
    fake_text_block.type = "text"
    fake_text_block.text = (
        '{"executive_summary": "Loss ratio is healthy.", "key_findings": ["finding one"], '
        '"risk_flags": [], "recommendations": ["keep monitoring"]}'
    )
    fake_response = MagicMock()
    fake_response.content = [fake_text_block]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    fake_anthropic_module = MagicMock()
    fake_anthropic_module.Anthropic.return_value = fake_client

    with patch.dict("sys.modules", {"anthropic": fake_anthropic_module}):
        report = generator.generate({"overall_loss_ratio": 0.4})

    assert report.source == "llm"
    assert report.executive_summary == "Loss ratio is healthy."
    assert report.recommendations == ["keep monitoring"]
