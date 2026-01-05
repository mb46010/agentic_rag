# tests/unit/answer/test_postprocess_answer.py
"""Unit tests for postprocess_answer node."""

import pytest

from agentic_rag.answer.nodes.postprocess_answer import make_postprocess_answer_node


class TestPostprocessAnswer:
    """Tests for postprocess_answer node."""

    def test_removes_code_when_no_code_constraint(self, sample_answer_state):
        """Should remove code blocks when no_code constraint is set."""
        state = {
            **sample_answer_state,
            "constraints": {"format": ["no_code"]},
            "final_answer": "Here is the answer:\n```python\nprint('hello')\n```\nEnd of answer.",
        }

        node = make_postprocess_answer_node()
        result = node(state)

        assert "```" not in result["final_answer"]
        assert "print" not in result["final_answer"]
        assert "Here is the answer:" in result["final_answer"]
        assert "End of answer." in result["final_answer"]

    def test_removes_code_when_plan_requires_no_code(self, sample_answer_state):
        """Should remove code blocks when plan specifies no_code."""
        state = {
            **sample_answer_state,
            "plan": {"answer_requirements": {"format": ["no_code"]}},
            "final_answer": "Explanation:\n```javascript\nconst x = 1;\n```\nDone.",
        }

        node = make_postprocess_answer_node()
        result = node(state)

        assert "```" not in result["final_answer"]
        assert "const x" not in result["final_answer"]

    def test_keeps_code_when_no_restriction(self, sample_answer_state):
        """Should keep code blocks when no restriction is set."""
        state = {
            **sample_answer_state,
            "constraints": {"format": []},
            "final_answer": "Here is code:\n```python\nprint('hello')\n```\nUse it wisely.",
        }

        node = make_postprocess_answer_node()
        result = node(state)

        assert "```python" in result["final_answer"]
        assert "print('hello')" in result["final_answer"]

    def test_filters_citations_with_invalid_evidence_ids(self, sample_answer_state):
        """Should filter out citations referencing unknown evidence IDs."""
        state = {
            **sample_answer_state,
            "final_evidence": [
                {"evidence_id": "ev_001", "text": "Valid evidence"},
                {"evidence_id": "ev_002", "text": "Another valid"},
            ],
            "citations": [
                {"evidence_id": "ev_001", "text": "Citation 1"},  # Valid
                {"evidence_id": "ev_999", "text": "Citation 2"},  # Invalid - should be removed
                {"evidence_id": "ev_002", "text": "Citation 3"},  # Valid
            ],
        }

        node = make_postprocess_answer_node()
        result = node(state)

        assert len(result["citations"]) == 2
        assert result["citations"][0]["evidence_id"] == "ev_001"
        assert result["citations"][1]["evidence_id"] == "ev_002"

    def test_filters_used_evidence_ids(self, sample_answer_state):
        """Should filter used_evidence_ids to only include valid IDs."""
        state = {
            **sample_answer_state,
            "final_evidence": [
                {"evidence_id": "ev_001", "text": "Valid"},
                {"evidence_id": "ev_002", "text": "Valid"},
            ],
            "answer_meta": {
                "used_evidence_ids": ["ev_001", "ev_999", "ev_002", "ev_888"],
            },
        }

        node = make_postprocess_answer_node()
        result = node(state)

        assert len(result["answer_meta"]["used_evidence_ids"]) == 2
        assert "ev_001" in result["answer_meta"]["used_evidence_ids"]
        assert "ev_002" in result["answer_meta"]["used_evidence_ids"]
        assert "ev_999" not in result["answer_meta"]["used_evidence_ids"]
        assert "ev_888" not in result["answer_meta"]["used_evidence_ids"]

    def test_handles_empty_final_answer(self, sample_answer_state):
        """Should provide default text when final_answer is empty."""
        state = {**sample_answer_state, "final_answer": ""}

        node = make_postprocess_answer_node()
        result = node(state)

        assert result["final_answer"] == "(No answer produced.)"

    def test_handles_none_final_answer(self, sample_answer_state):
        """Should handle None final_answer."""
        state = {**sample_answer_state, "final_answer": None}

        node = make_postprocess_answer_node()
        result = node(state)

        assert result["final_answer"] == "(No answer produced.)"

    def test_handles_missing_final_evidence(self, sample_answer_state):
        """Should handle missing final_evidence gracefully."""
        state = {
            **sample_answer_state,
            "final_evidence": None,
            "citations": [{"evidence_id": "ev_001", "text": "Citation"}],
        }

        node = make_postprocess_answer_node()
        result = node(state)

        # All citations should be filtered out
        assert len(result["citations"]) == 0

    def test_handles_invalid_final_evidence_items(self, sample_answer_state):
        """Should handle invalid evidence items gracefully."""
        state = {
            **sample_answer_state,
            "final_evidence": [
                {"evidence_id": "ev_001", "text": "Valid"},
                "invalid_item",  # Invalid
                {"invalid": "structure"},  # Invalid
                {"evidence_id": "ev_002", "text": "Valid"},
            ],
            "citations": [
                {"evidence_id": "ev_001", "text": "Citation 1"},
                {"evidence_id": "ev_002", "text": "Citation 2"},
            ],
        }

        node = make_postprocess_answer_node()
        result = node(state)

        # Should only recognize valid evidence items
        assert len(result["citations"]) == 2

    def test_handles_invalid_citation_format(self, sample_answer_state):
        """Should handle invalid citation formats gracefully."""
        state = {
            **sample_answer_state,
            "final_evidence": [{"evidence_id": "ev_001", "text": "Valid"}],
            "citations": [
                {"evidence_id": "ev_001", "text": "Valid citation"},
                "invalid_citation",  # Not a dict
                {"no_evidence_id": "data"},  # Missing evidence_id
                {"evidence_id": "ev_001", "text": "Another valid"},
            ],
        }

        node = make_postprocess_answer_node()
        result = node(state)

        # Should only keep valid citations
        assert len(result["citations"]) == 2
        assert all(isinstance(c, dict) for c in result["citations"])
        assert all("evidence_id" in c for c in result["citations"])

    def test_handles_missing_citations(self, sample_answer_state):
        """Should handle missing citations gracefully."""
        state = {**sample_answer_state, "citations": None}

        node = make_postprocess_answer_node()
        result = node(state)

        assert result["citations"] == []

    def test_handles_missing_answer_meta(self, sample_answer_state):
        """Should handle missing answer_meta gracefully."""
        state = {**sample_answer_state, "answer_meta": None}

        node = make_postprocess_answer_node()
        result = node(state)

        assert "answer_meta" in result
        assert result["answer_meta"]["used_evidence_ids"] == []

    def test_preserves_other_answer_meta_fields(self, sample_answer_state):
        """Should preserve other answer_meta fields."""
        state = {
            **sample_answer_state,
            "final_evidence": [{"evidence_id": "ev_001", "text": "Valid"}],
            "answer_meta": {
                "answer_version": "answer_v1",
                "mode": "answer",
                "coverage_confidence": 0.8,
                "used_evidence_ids": ["ev_001", "ev_999"],
                "custom_field": "value",
            },
        }

        node = make_postprocess_answer_node()
        result = node(state)

        # Should preserve other fields
        assert result["answer_meta"]["answer_version"] == "answer_v1"
        assert result["answer_meta"]["mode"] == "answer"
        assert result["answer_meta"]["coverage_confidence"] == 0.8
        assert result["answer_meta"]["custom_field"] == "value"
        # But filter used_evidence_ids
        assert result["answer_meta"]["used_evidence_ids"] == ["ev_001"]

    def test_strips_whitespace_after_code_removal(self, sample_answer_state):
        """Should strip whitespace after removing code blocks."""
        state = {
            **sample_answer_state,
            "constraints": {"format": ["no_code"]},
            "final_answer": "   \n\n```python\ncode\n```\n\n   ",
        }

        node = make_postprocess_answer_node()
        result = node(state)

        # Should not be just whitespace
        assert result["final_answer"] == "(No answer produced.)"

    def test_handles_multiple_code_blocks(self, sample_answer_state):
        """Should remove all code blocks when no_code is set."""
        state = {
            **sample_answer_state,
            "constraints": {"format": ["no_code"]},
            "final_answer": """First paragraph.
```python
code1
```
Middle paragraph.
```javascript
code2
```
Last paragraph.""",
        }

        node = make_postprocess_answer_node()
        result = node(state)

        assert "```" not in result["final_answer"]
        assert "code1" not in result["final_answer"]
        assert "code2" not in result["final_answer"]
        assert "First paragraph" in result["final_answer"]
        assert "Middle paragraph" in result["final_answer"]
        assert "Last paragraph" in result["final_answer"]

    def test_handles_nested_code_markers(self, sample_answer_state):
        """Should handle code blocks with various markers."""
        state = {
            **sample_answer_state,
            "constraints": {"format": ["no_code"]},
            "final_answer": "Text before ```typescript\nconst x = 1;\n``` text after.",
        }

        node = make_postprocess_answer_node()
        result = node(state)

        assert "```" not in result["final_answer"]
        assert "const x" not in result["final_answer"]
        assert "Text before" in result["final_answer"]
        assert "text after" in result["final_answer"]
