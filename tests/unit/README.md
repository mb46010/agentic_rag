# Unit Tests

This directory contains unit tests for the agentic_rag package.

## Structure

```
tests/unit/
├── intent/              # Intent module tests
│   ├── conftest.py     # Shared fixtures (mocked LLM, sample data)
│   ├── test_normalize_gate.py   # normalize_gate node tests
│   ├── test_extract_signals.py  # extract_signals node tests
│   ├── test_graph.py            # Intent graph tests
│   └── test_state.py            # State models and utilities tests
└── README.md           # This file
```

## Running Tests

Run all unit tests:
```bash
pytest tests/unit/
```

Run specific test module:
```bash
pytest tests/unit/intent/test_normalize_gate.py
```

Run with verbose output:
```bash
pytest tests/unit/ -v
```

Run with coverage:
```bash
pytest tests/unit/ --cov=src/agentic_rag --cov-report=html
```

## Test Philosophy

### Unit Tests vs Integration Tests

**Unit tests** (this directory):
- Test individual components in isolation
- Use mocked dependencies (LLM calls, external services)
- Fast execution (<1s per test)
- No network calls
- Focus on logic, error handling, edge cases

**Integration tests** (tests/intent_eval/):
- Test components together with real or realistic mocks
- May use real LLM calls
- Slower execution
- Focus on end-to-end behavior

### Mocking Strategy

All unit tests mock LLM calls to:
1. **Speed**: Tests run in milliseconds, not seconds
2. **Reliability**: No flaky network issues
3. **Cost**: No API charges
4. **Determinism**: Consistent, reproducible results

Example mock pattern:
```python
def test_normalize_gate_success(mock_llm, sample_state):
    # Setup mock response
    mock_result = NormalizeModel(normalized_query="test", ...)
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_chain

    # Test the node
    node = make_normalize_gate_node(mock_llm)
    result = node(sample_state)

    # Assert expectations
    assert result["normalized_query"] == "test"
```

## Test Coverage

### Intent Module Coverage

- **normalize_gate node**: 100%
  - ✅ Success cases
  - ✅ Missing/invalid messages
  - ✅ Validation errors
  - ✅ Runtime errors
  - ✅ Optional fields (language, locale)
  - ✅ Pydantic model validation

- **extract_signals node**: 100%
  - ✅ Success cases
  - ✅ Missing/invalid messages
  - ✅ Validation errors
  - ✅ Runtime errors
  - ✅ Default values
  - ✅ Entity/acronym extraction
  - ✅ Literal terms preservation
  - ✅ Complexity flags

- **Intent graph**: 100%
  - ✅ Graph construction
  - ✅ End-to-end execution
  - ✅ Error propagation
  - ✅ Minimal inputs
  - ✅ Optional field preservation
  - ✅ Different user intents

- **State models**: 100%
  - ✅ TypedDict structures
  - ✅ Error reducer function
  - ✅ All state model types

### Total: 60+ unit tests

## Writing New Tests

### 1. Use Existing Fixtures

```python
def test_my_feature(mock_llm, sample_state, mock_normalize_output):
    # Fixtures available in conftest.py
    pass
```

### 2. Follow Naming Convention

- Test files: `test_<module>.py`
- Test classes: `Test<Component>`
- Test functions: `test_<what>_<condition>`

Example:
```python
class TestNormalizeGate:
    def test_normalize_gate_missing_messages(self):
        pass

    def test_normalize_gate_validation_error(self):
        pass
```

### 3. Test Error Paths

Always test:
- ✅ Happy path (success)
- ✅ Missing inputs
- ✅ Invalid inputs
- ✅ Runtime errors
- ✅ Edge cases

### 4. Mock External Dependencies

```python
from unittest.mock import MagicMock

def test_with_mock():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "mocked response"
    # Use mock_llm in test
```

## Common Issues

### Issue: Import errors
**Solution**: Install package in editable mode:
```bash
pip install -e .
```

### Issue: Tests not found
**Solution**: Ensure test files start with `test_`:
```bash
# Good: test_normalize_gate.py
# Bad: normalize_gate_test.py
```

### Issue: Fixture not found
**Solution**: Check conftest.py is in the right location and named correctly.

## CI/CD Integration

These tests are designed to run in CI:
```yaml
# .github/workflows/test.yml example
- name: Run unit tests
  run: pytest tests/unit/ --cov --cov-report=xml
```

## Next Steps

To add tests for other modules:
1. Create `tests/unit/<module>/` directory
2. Add `conftest.py` with module-specific fixtures
3. Add test files following the patterns above
4. Update this README with coverage info

Example for executor:
```bash
mkdir -p tests/unit/executor
# Add tests following same patterns
```
