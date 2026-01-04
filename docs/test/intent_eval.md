```bash
pytest tests/intent_eval -q

pytest tests/intent_eval/test_schema_contract.py -q
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


