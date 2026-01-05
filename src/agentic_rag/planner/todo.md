What you should do next (practical)

Add a smoke test for planner similar to intake.

Add 2-3 behavior contract cases:

ambiguous acronym -> clarify_then_retrieve, no retrieval_rounds

stacktrace + ids -> retrieve_then_answer, must_match_exactly=true, use_hyde=false, bm25/hybrid alpha<=0.4

conceptual architecture -> retrieve_then_answer, HyDE enabled, vector-heavy/hybrid alpha>=0.6

If you paste your current IntakeState (latest) and where you want plan stored (top-level vs nested), I can align the typing and return dict exactly to your codebase conventions.