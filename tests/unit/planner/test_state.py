# tests/unit/planner/test_state.py
"""Unit tests for planner state models."""

import pytest
from pydantic import ValidationError

from agentic_rag.planner.state import (
    AcceptanceCriteria,
    AnswerRequirements,
    BudgetSpec,
    ClarifyingQuestion,
    LiteralConstraints,
    PlannerMeta,
    PlannerState,
    RerankSpec,
    RetrievalModeSpec,
    RetrievalRound,
    RoundFilters,
    RoundOutputSpec,
    SafetySpec,
    StopConditions,
)


class TestClarifyingQuestion:
    """Tests for ClarifyingQuestion model."""

    def test_valid_clarifying_question(self):
        """Test valid ClarifyingQuestion."""
        q = ClarifyingQuestion(
            question="What version of the library?",
            reason="missing_version",
            blocking=True,
        )
        assert q.question == "What version of the library?"
        assert q.reason == "missing_version"
        assert q.blocking is True

    def test_non_blocking_clarifying_question(self):
        """Test non-blocking ClarifyingQuestion."""
        q = ClarifyingQuestion(
            question="Which cloud provider?",
            reason="missing_scope",
            blocking=False,
        )
        assert q.blocking is False

    def test_clarifying_question_min_length(self):
        """Test question minimum length validation."""
        with pytest.raises(ValidationError):
            ClarifyingQuestion(
                question="Yes",  # Too short (< 5 chars)
                reason="missing_version",
                blocking=True,
            )

    def test_clarifying_question_invalid_reason(self):
        """Test invalid clarification reason."""
        with pytest.raises(ValidationError):
            ClarifyingQuestion(
                question="What is your question?",
                reason="invalid_reason",
                blocking=True,
            )

    def test_clarifying_question_extra_field_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ClarifyingQuestion(
                question="What version?",
                reason="missing_version",
                blocking=True,
                extra_field="not allowed",
            )


class TestRetrievalModeSpec:
    """Tests for RetrievalModeSpec model."""

    def test_retrieval_mode_bm25(self):
        """Test BM25 retrieval mode."""
        mode = RetrievalModeSpec(type="bm25", k=30)
        assert mode.type == "bm25"
        assert mode.k == 30
        assert mode.alpha is None

    def test_retrieval_mode_vector(self):
        """Test vector retrieval mode."""
        mode = RetrievalModeSpec(type="vector", k=20)
        assert mode.type == "vector"

    def test_retrieval_mode_hybrid_with_alpha(self):
        """Test hybrid mode with alpha."""
        mode = RetrievalModeSpec(type="hybrid", k=25, alpha=0.6)
        assert mode.type == "hybrid"
        assert mode.alpha == 0.6

    def test_retrieval_mode_default_k(self):
        """Test default k value."""
        mode = RetrievalModeSpec(type="vector")
        assert mode.k == 20

    def test_retrieval_mode_k_bounds(self):
        """Test k value bounds."""
        # Too small
        with pytest.raises(ValidationError):
            RetrievalModeSpec(type="vector", k=0)

        # Too large
        with pytest.raises(ValidationError):
            RetrievalModeSpec(type="vector", k=201)

        # Edge cases (valid)
        RetrievalModeSpec(type="vector", k=1)
        RetrievalModeSpec(type="vector", k=200)

    def test_retrieval_mode_alpha_bounds(self):
        """Test alpha value bounds."""
        # Valid values
        RetrievalModeSpec(type="hybrid", alpha=0.0)
        RetrievalModeSpec(type="hybrid", alpha=1.0)
        RetrievalModeSpec(type="hybrid", alpha=0.5)

        # Invalid values
        with pytest.raises(ValidationError):
            RetrievalModeSpec(type="hybrid", alpha=-0.1)

        with pytest.raises(ValidationError):
            RetrievalModeSpec(type="hybrid", alpha=1.1)


class TestRoundFilters:
    """Tests for RoundFilters model."""

    def test_round_filters_empty(self):
        """Test empty RoundFilters."""
        filters = RoundFilters()
        assert filters.doc_types == []
        assert filters.domains == []
        assert filters.entities == []
        assert filters.time_range is None

    def test_round_filters_with_values(self):
        """Test RoundFilters with values."""
        filters = RoundFilters(
            doc_types=["documentation", "tutorial"],
            domains=["azure"],
            entities=["Azure OpenAI"],
            time_range="last_6_months",
        )
        assert len(filters.doc_types) == 2
        assert "azure" in filters.domains
        assert filters.time_range == "last_6_months"

    def test_round_filters_partial(self):
        """Test partial RoundFilters."""
        filters = RoundFilters(doc_types=["guide"])
        assert len(filters.doc_types) == 1
        assert filters.domains == []


class TestRerankSpec:
    """Tests for RerankSpec model."""

    def test_rerank_spec_defaults(self):
        """Test RerankSpec default values."""
        spec = RerankSpec()
        assert spec.enabled is True
        assert spec.model == "cross_encoder"
        assert spec.rerank_top_k == 60

    def test_rerank_spec_custom(self):
        """Test RerankSpec with custom values."""
        spec = RerankSpec(enabled=False, rerank_top_k=40)
        assert spec.enabled is False
        assert spec.rerank_top_k == 40

    def test_rerank_top_k_bounds(self):
        """Test rerank_top_k bounds."""
        # Valid edge cases
        RerankSpec(rerank_top_k=5)
        RerankSpec(rerank_top_k=200)

        # Invalid
        with pytest.raises(ValidationError):
            RerankSpec(rerank_top_k=4)

        with pytest.raises(ValidationError):
            RerankSpec(rerank_top_k=201)


class TestRoundOutputSpec:
    """Tests for RoundOutputSpec model."""

    def test_round_output_defaults(self):
        """Test RoundOutputSpec defaults."""
        spec = RoundOutputSpec()
        assert spec.max_docs == 8

    def test_round_output_custom(self):
        """Test RoundOutputSpec with custom value."""
        spec = RoundOutputSpec(max_docs=12)
        assert spec.max_docs == 12

    def test_max_docs_bounds(self):
        """Test max_docs bounds."""
        RoundOutputSpec(max_docs=1)
        RoundOutputSpec(max_docs=50)

        with pytest.raises(ValidationError):
            RoundOutputSpec(max_docs=0)

        with pytest.raises(ValidationError):
            RoundOutputSpec(max_docs=51)


class TestRetrievalRound:
    """Tests for RetrievalRound model."""

    def test_retrieval_round_minimal(self):
        """Test minimal RetrievalRound."""
        round_data = RetrievalRound(
            round_id=0,
            purpose="recall",
            query_variants=["test query"],
        )
        assert round_data.round_id == 0
        assert round_data.purpose == "recall"
        assert len(round_data.query_variants) == 1
        assert round_data.use_hyde is False
        assert round_data.rrf is True

    def test_retrieval_round_full(self):
        """Test full RetrievalRound."""
        round_data = RetrievalRound(
            round_id=1,
            purpose="precision",
            query_variants=["query 1", "query 2"],
            retrieval_modes=[
                RetrievalModeSpec(type="hybrid", k=30, alpha=0.5)
            ],
            filters=RoundFilters(doc_types=["tutorial"]),
            use_hyde=True,
            rrf=False,
            rerank=RerankSpec(enabled=True, rerank_top_k=80),
            output=RoundOutputSpec(max_docs=10),
        )
        assert round_data.round_id == 1
        assert round_data.purpose == "precision"
        assert len(round_data.query_variants) == 2
        assert round_data.use_hyde is True
        assert round_data.rrf is False

    def test_retrieval_round_id_bounds(self):
        """Test round_id bounds."""
        RetrievalRound(round_id=0, purpose="recall", query_variants=["q"])
        RetrievalRound(round_id=10, purpose="recall", query_variants=["q"])

        with pytest.raises(ValidationError):
            RetrievalRound(round_id=-1, purpose="recall", query_variants=["q"])

        with pytest.raises(ValidationError):
            RetrievalRound(round_id=11, purpose="recall", query_variants=["q"])

    def test_retrieval_round_empty_query_variants(self):
        """Test that empty query_variants fails validation."""
        with pytest.raises(ValidationError):
            RetrievalRound(
                round_id=0,
                purpose="recall",
                query_variants=[],  # Empty not allowed
            )

    def test_retrieval_round_invalid_purpose(self):
        """Test invalid purpose value."""
        with pytest.raises(ValidationError):
            RetrievalRound(
                round_id=0,
                purpose="invalid_purpose",
                query_variants=["q"],
            )


class TestLiteralConstraints:
    """Tests for LiteralConstraints model."""

    def test_literal_constraints_defaults(self):
        """Test LiteralConstraints defaults."""
        constraints = LiteralConstraints()
        assert constraints.must_preserve_terms == []
        assert constraints.must_match_exactly is False

    def test_literal_constraints_with_terms(self):
        """Test LiteralConstraints with terms."""
        constraints = LiteralConstraints(
            must_preserve_terms=["ERROR-123", "config.yaml"],
            must_match_exactly=True,
        )
        assert len(constraints.must_preserve_terms) == 2
        assert constraints.must_match_exactly is True


class TestAcceptanceCriteria:
    """Tests for AcceptanceCriteria model."""

    def test_acceptance_criteria_defaults(self):
        """Test AcceptanceCriteria defaults."""
        criteria = AcceptanceCriteria()
        assert criteria.min_independent_sources == 1
        assert criteria.require_authoritative_source is False
        assert criteria.must_cover_entities == []
        assert criteria.must_answer_subquestions == []

    def test_acceptance_criteria_custom(self):
        """Test AcceptanceCriteria with custom values."""
        criteria = AcceptanceCriteria(
            min_independent_sources=3,
            require_authoritative_source=True,
            must_cover_entities=["Azure OpenAI"],
            must_answer_subquestions=["How to authenticate?"],
        )
        assert criteria.min_independent_sources == 3
        assert criteria.require_authoritative_source is True
        assert len(criteria.must_cover_entities) == 1

    def test_acceptance_criteria_min_sources_bounds(self):
        """Test min_independent_sources bounds."""
        AcceptanceCriteria(min_independent_sources=1)
        AcceptanceCriteria(min_independent_sources=5)

        with pytest.raises(ValidationError):
            AcceptanceCriteria(min_independent_sources=0)

        with pytest.raises(ValidationError):
            AcceptanceCriteria(min_independent_sources=6)


class TestStopConditions:
    """Tests for StopConditions model."""

    def test_stop_conditions_defaults(self):
        """Test StopConditions defaults."""
        conditions = StopConditions()
        assert conditions.max_rounds == 2
        assert conditions.max_total_docs == 12
        assert conditions.confidence_threshold is None
        assert conditions.no_new_information_rounds == 1

    def test_stop_conditions_custom(self):
        """Test StopConditions with custom values."""
        conditions = StopConditions(
            max_rounds=3,
            max_total_docs=20,
            confidence_threshold=0.85,
            no_new_information_rounds=2,
        )
        assert conditions.max_rounds == 3
        assert conditions.max_total_docs == 20
        assert conditions.confidence_threshold == 0.85

    def test_stop_conditions_bounds(self):
        """Test StopConditions bounds."""
        # max_rounds
        with pytest.raises(ValidationError):
            StopConditions(max_rounds=0)
        with pytest.raises(ValidationError):
            StopConditions(max_rounds=4)

        # max_total_docs
        with pytest.raises(ValidationError):
            StopConditions(max_total_docs=0)
        with pytest.raises(ValidationError):
            StopConditions(max_total_docs=31)

        # confidence_threshold
        with pytest.raises(ValidationError):
            StopConditions(confidence_threshold=-0.1)
        with pytest.raises(ValidationError):
            StopConditions(confidence_threshold=1.1)


class TestAnswerRequirements:
    """Tests for AnswerRequirements model."""

    def test_answer_requirements_defaults(self):
        """Test AnswerRequirements defaults."""
        req = AnswerRequirements()
        assert req.format == []
        assert req.tone is None
        assert req.length is None
        assert req.citation_style is None

    def test_answer_requirements_custom(self):
        """Test AnswerRequirements with custom values."""
        req = AnswerRequirements(
            format=["bullet_points", "citations"],
            tone="professional",
            length="concise",
            citation_style="APA",
        )
        assert "bullet_points" in req.format
        assert req.tone == "professional"


class TestBudgetSpec:
    """Tests for BudgetSpec model."""

    def test_budget_spec_defaults(self):
        """Test BudgetSpec defaults."""
        budget = BudgetSpec()
        assert budget.max_tokens == 8000
        assert budget.max_latency_ms is None

    def test_budget_spec_custom(self):
        """Test BudgetSpec with custom values."""
        budget = BudgetSpec(max_tokens=4000, max_latency_ms=5000)
        assert budget.max_tokens == 4000
        assert budget.max_latency_ms == 5000

    def test_budget_spec_bounds(self):
        """Test BudgetSpec bounds."""
        # max_tokens
        BudgetSpec(max_tokens=256)
        BudgetSpec(max_tokens=100000)

        with pytest.raises(ValidationError):
            BudgetSpec(max_tokens=255)

        with pytest.raises(ValidationError):
            BudgetSpec(max_tokens=100001)

        # max_latency_ms
        BudgetSpec(max_latency_ms=100)
        BudgetSpec(max_latency_ms=120000)

        with pytest.raises(ValidationError):
            BudgetSpec(max_latency_ms=99)

        with pytest.raises(ValidationError):
            BudgetSpec(max_latency_ms=120001)


class TestSafetySpec:
    """Tests for SafetySpec model."""

    def test_safety_spec_defaults(self):
        """Test SafetySpec defaults."""
        safety = SafetySpec()
        assert safety.sensitivity == "normal"
        assert safety.pii_allowed is False

    def test_safety_spec_custom(self):
        """Test SafetySpec with custom values."""
        safety = SafetySpec(sensitivity="elevated", pii_allowed=True)
        assert safety.sensitivity == "elevated"
        assert safety.pii_allowed is True

    def test_safety_spec_restricted(self):
        """Test SafetySpec with restricted sensitivity."""
        safety = SafetySpec(sensitivity="restricted")
        assert safety.sensitivity == "restricted"


class TestPlannerMeta:
    """Tests for PlannerMeta model."""

    def test_planner_meta_defaults(self):
        """Test PlannerMeta defaults."""
        meta = PlannerMeta()
        assert meta.planner_version == "planner_v1"
        assert meta.rationale_tags == []

    def test_planner_meta_custom(self):
        """Test PlannerMeta with custom values."""
        meta = PlannerMeta(
            planner_version="planner_v2",
            rationale_tags=["multi_round", "high_precision"],
        )
        assert meta.planner_version == "planner_v2"
        assert "multi_round" in meta.rationale_tags


class TestPlannerState:
    """Tests for PlannerState model."""

    def test_planner_state_minimal_direct_answer(self):
        """Test minimal PlannerState for direct_answer strategy."""
        state = PlannerState(
            goal="What is 2+2?",
            strategy="direct_answer",
        )
        assert state.goal == "What is 2+2?"
        assert state.strategy == "direct_answer"
        assert state.retrieval_rounds == []
        assert state.clarifying_questions == []

    def test_planner_state_retrieve_then_answer(self):
        """Test PlannerState with retrieve_then_answer strategy."""
        state = PlannerState(
            goal="Configure Azure OpenAI",
            strategy="retrieve_then_answer",
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="recall",
                    query_variants=["Azure OpenAI configuration"],
                )
            ],
        )
        assert state.strategy == "retrieve_then_answer"
        assert len(state.retrieval_rounds) == 1

    def test_planner_state_clarify_then_retrieve(self):
        """Test PlannerState with clarify_then_retrieve strategy."""
        state = PlannerState(
            goal="Configure library",
            strategy="clarify_then_retrieve",
            clarifying_questions=[
                ClarifyingQuestion(
                    question="Which version?",
                    reason="missing_version",
                    blocking=True,
                )
            ],
        )
        assert state.strategy == "clarify_then_retrieve"
        assert len(state.clarifying_questions) == 1

    def test_planner_state_defer_or_refuse(self):
        """Test PlannerState with defer_or_refuse strategy."""
        state = PlannerState(
            goal="Hack into system",
            strategy="defer_or_refuse",
        )
        assert state.strategy == "defer_or_refuse"

    def test_planner_state_full(self):
        """Test full PlannerState with all fields."""
        state = PlannerState(
            goal="Configure Azure OpenAI for production",
            strategy="retrieve_then_answer",
            clarifying_questions=[],
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="recall",
                    query_variants=["Azure OpenAI production setup"],
                    retrieval_modes=[
                        RetrievalModeSpec(type="hybrid", k=25, alpha=0.5)
                    ],
                    filters=RoundFilters(doc_types=["documentation"]),
                    use_hyde=False,
                    rrf=True,
                    rerank=RerankSpec(enabled=True),
                    output=RoundOutputSpec(max_docs=8),
                )
            ],
            literal_constraints=LiteralConstraints(
                must_preserve_terms=["production"],
                must_match_exactly=False,
            ),
            acceptance_criteria=AcceptanceCriteria(
                min_independent_sources=2,
                require_authoritative_source=True,
            ),
            stop_conditions=StopConditions(max_rounds=2, max_total_docs=15),
            answer_requirements=AnswerRequirements(
                format=["bullet_points", "citations"],
                tone="professional",
            ),
            budget=BudgetSpec(max_tokens=10000),
            safety=SafetySpec(sensitivity="normal", pii_allowed=False),
            planner_meta=PlannerMeta(rationale_tags=["production_config"]),
        )
        assert state.goal == "Configure Azure OpenAI for production"
        assert len(state.retrieval_rounds) == 1
        assert state.acceptance_criteria.min_independent_sources == 2

    def test_planner_state_goal_min_length(self):
        """Test goal minimum length validation."""
        with pytest.raises(ValidationError):
            PlannerState(
                goal="Hi",  # Too short
                strategy="direct_answer",
            )

    def test_planner_state_invalid_strategy(self):
        """Test invalid strategy."""
        with pytest.raises(ValidationError):
            PlannerState(
                goal="Valid goal",
                strategy="invalid_strategy",
            )

    def test_planner_state_extra_field_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            PlannerState(
                goal="Valid goal",
                strategy="direct_answer",
                extra_field="not allowed",
            )

    def test_planner_state_multiple_rounds(self):
        """Test PlannerState with multiple retrieval rounds."""
        state = PlannerState(
            goal="Complex query requiring multiple rounds",
            strategy="retrieve_then_answer",
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="recall",
                    query_variants=["broad query"],
                ),
                RetrievalRound(
                    round_id=1,
                    purpose="precision",
                    query_variants=["specific query"],
                ),
            ],
        )
        assert len(state.retrieval_rounds) == 2
        assert state.retrieval_rounds[0].round_id == 0
        assert state.retrieval_rounds[1].round_id == 1

    def test_planner_state_model_dump(self):
        """Test that PlannerState can be dumped to dict."""
        state = PlannerState(
            goal="Test goal",
            strategy="direct_answer",
        )
        dumped = state.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["goal"] == "Test goal"
        assert dumped["strategy"] == "direct_answer"
