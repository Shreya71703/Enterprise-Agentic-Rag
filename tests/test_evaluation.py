"""
Enterprise RAG — Evaluation Tests

Tests for the RAGEvaluator metrics calculation.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from src.evaluation.evaluator import RAGEvaluator


class TestRAGEvaluator:
    """Tests for the RAGEvaluator client."""

    def test_json_list_parsing(self):
        text = 'Here is the JSON list:\n\n[\n  "Claim 1",\n  "Claim 2"\n]\n\nHope this helps!'
        claims = RAGEvaluator._parse_json_list(text)
        assert claims == ["Claim 1", "Claim 2"]

    def test_json_dict_parsing(self):
        text = '```json\n{\n  "supported": true,\n  "reason": "Test explanation"\n}\n```'
        data = RAGEvaluator._parse_json_dict(text)
        assert data["supported"] is True
        assert data["reason"] == "Test explanation"

    def test_mock_evaluator(self):
        # When dummy fallback model is used, evaluator returns default scores
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        class DummyChatModel(BaseChatModel):
            def _generate(self, messages, stop=None, **kwargs):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Final Mock Response"))])
            def bind_tools(self, tools, **kwargs):
                return self
            @property
            def _llm_type(self) -> str:
                return "dummy"

        evaluator = RAGEvaluator(llm=DummyChatModel())
        res = evaluator.evaluate("query", ["context"], "answer")
        assert res["faithfulness"] == 1.0
        assert res["context_relevance"] == 1.0
        assert "Mock evaluation" in res["reasoning"]["faithfulness"]
