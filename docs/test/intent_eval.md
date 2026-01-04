```bash
LANGFUSE_ENABLED=0 pytest -p no:langsmith tests/intent_eval -q

LANGFUSE_ENABLED=0 pytest -p no:langsmith tests/intent_eval/test_schema_contract.py -vv -s

LANGFUSE_ENABLED=0 pytest -p no:langsmith tests/intent_eval/test_stability.py -vv -s

```

```tree
tests/intent_eval/
  cases/
    intake_v1/
      c001_internal_procedure_no_code.json
      c002_ambiguous_acronym_blocking.json
      c003_troubleshoot_artifacts.json
  expected/
    intake_v1/
      c001_internal_procedure_no_code.expected.json
      c002_ambiguous_acronym_blocking.expected.json
      c003_troubleshoot_artifacts.expected.json
```


