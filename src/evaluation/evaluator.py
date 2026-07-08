"""
Enterprise RAG — Evaluation Service

Implements RAGAS-like evaluation metrics using LLM-as-a-judge:
    1. Faithfulness (Groundedness): Is the answer supported by the retrieved context?
    2. Context Relevance: Is the retrieved context relevant and noise-free for the query?

This custom implementation provides transparent step-by-step reasoning
and avoids brittle dependencies of the official RAGAS library.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAG Evaluation Service
# ---------------------------------------------------------------------------
class RAGEvaluator:
    """
    Computes Faithfulness and Context Relevance scores.

    Usage:
        evaluator = RAGEvaluator(llm)
        scores = evaluator.evaluate(
            query="What LLMs does Cortex support?",
            contexts=["Cortex supports Gemini and Llama 3."],
            answer="Cortex supports Gemini and Llama 3."
        )
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            from src.router.agent import create_router_llm
            self.llm = create_router_llm()
        else:
            self.llm = llm

        self.is_mock = hasattr(self.llm, "_llm_type") and getattr(self.llm, "_llm_type") == "dummy"

    def evaluate(
        self,
        query: str,
        contexts: list[str],
        answer: str,
    ) -> dict[str, Any]:
        """
        Evaluate a single query-context-answer triad.
        """
        if self.is_mock:
            return {
                "faithfulness": 1.0,
                "context_relevance": 1.0,
                "reasoning": {
                    "faithfulness": "Mock evaluation: perfect score.",
                    "context_relevance": "Mock evaluation: perfect score."
                }
            }

        context_text = "\n\n".join(contexts)

        # Compute metrics
        faith_score, faith_reason = self._compute_faithfulness(answer, context_text)
        relevance_score, relevance_reason = self._compute_context_relevance(query, context_text)

        return {
            "faithfulness": round(faith_score, 2),
            "context_relevance": round(relevance_score, 2),
            "reasoning": {
                "faithfulness": faith_reason,
                "context_relevance": relevance_reason,
            }
        }

    # -------------------------------------------------------------------
    # Faithfulness Calculation
    # -------------------------------------------------------------------
    def _compute_faithfulness(self, answer: str, context: str) -> tuple[float, str]:
        """
        Judge if statements in answer are supported by context.
        """
        # Step 1: Extract individual claims/statements from the answer
        extract_prompt = (
            "Split the following RAG answer into a list of independent, single-sentence factual claims/statements. "
            "Output the results ONLY as a JSON list of strings, with no other conversational text.\n\n"
            f"Answer: {answer}"
        )
        try:
            res = self.llm.invoke([SystemMessage(content="You are a factual claim extractor."), HumanMessage(content=extract_prompt)])
            claims = self._parse_json_list(res.content)
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return 0.0, f"Error extracting claims: {e}"

        if not claims:
            return 1.0, "No claims extracted from answer."

        # Step 2: Verify each claim against the context
        verifications = []
        yes_count = 0

        for claim in claims:
            verify_prompt = (
                "Verify if the following claim is directly supported by the context. "
                "Respond ONLY with a JSON object in this format:\n"
                '{"supported": true/false, "reason": "brief explanation"}\n\n'
                f"Context: {context}\n\n"
                f"Claim: {claim}"
            )
            try:
                res_verify = self.llm.invoke([SystemMessage(content="You are a strict factual verifier."), HumanMessage(content=verify_prompt)])
                ver = self._parse_json_dict(res_verify.content)
                supported = ver.get("supported", False)
                verifications.append({"claim": claim, "supported": supported, "reason": ver.get("reason", "")})
                if supported:
                    yes_count += 1
            except Exception as e:
                verifications.append({"claim": claim, "supported": False, "reason": f"Verification error: {e}"})

        score = yes_count / len(claims)
        reason = f"{yes_count} of {len(claims)} claims supported by context.\nDetails:\n" + "\n".join(
            f"  - [{ '✓' if v['supported'] else '✗' }] '{v['claim']}' (Reason: {v['reason']})"
            for v in verifications
        )

        return score, reason

    # -------------------------------------------------------------------
    # Context Relevance Calculation
    # -------------------------------------------------------------------
    def _compute_context_relevance(self, query: str, context: str) -> tuple[float, str]:
        """
        Judge how much of the context is relevant to the query.
        """
        # Split context into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        if not sentences:
            return 0.0, "Empty context."

        relevance_prompt = (
            "Examine each sentence in the context below and determine if it is directly relevant "
            "to answering the user query. "
            "Respond ONLY with a JSON object where keys are 0-indexed sentence indices (0, 1, 2...) "
            "and values are booleans indicating relevance. Format:\n"
            '{"0": true, "1": false, "2": true}\n\n'
            f"Query: {query}\n\n"
            "Context Sentences:\n" + "\n".join(f"{idx}: {sentence}" for idx, sentence in enumerate(sentences))
        )

        try:
            res = self.llm.invoke([SystemMessage(content="You are a search relevance grader."), HumanMessage(content=relevance_prompt)])
            grades = self._parse_json_dict(res.content)
        except Exception as e:
            logger.error(f"Error grading relevance: {e}")
            return 0.0, f"Error grading relevance: {e}"

        relevant_count = 0
        details = []
        for idx, sentence in enumerate(sentences):
            # Check string or int key in dict
            is_relevant = grades.get(str(idx), grades.get(idx, False))
            details.append({"sentence": sentence, "relevant": is_relevant})
            if is_relevant:
                relevant_count += 1

        score = relevant_count / len(sentences)
        reason = f"{relevant_count} of {len(sentences)} sentences judged relevant to query.\nDetails:\n" + "\n".join(
            f"  - [{ '✓' if d['relevant'] else '✗' }] '{d['sentence']}'"
            for d in details
        )

        return score, reason

    # -------------------------------------------------------------------
    # JSON Parsing Helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _parse_json_list(text: str) -> list[str]:
        """Parse clean JSON list from LLM output block."""
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)

    @staticmethod
    def _parse_json_dict(text: str) -> dict[str, Any]:
        """Parse clean JSON dict from LLM output block."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
