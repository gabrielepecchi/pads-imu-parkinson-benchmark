# Running the Tests

All tests use synthetic toy data created inside the test files.
No real PADS dataset, patient files, result CSVs, internet access, or GPU are required.

## Prerequisites

The following project dependencies must already be installed in the active environment:
`numpy`, `scipy`, `scikit-learn`, `torch`, `pandas`, `pyyaml`.

pytest must be available:

```bash
pip install pytest
```

## Run all tests

From the repository root:

```bash
python -m pytest
```

## Run a specific test file

```bash
python -m pytest tests/test_cross_val.py
python -m pytest tests/test_feature_valid_lengths.py
python -m pytest tests/test_imports.py
```

## Run with verbose output

```bash
python -m pytest -v
```

## Test descriptions

| File | What it checks |
|---|---|
| `tests/test_cross_val.py` | Subject-level leakage prevention in CV splits |
| `tests/test_feature_valid_lengths.py` | Feature extraction respects valid_lengths, ignores padding |
| `tests/test_imports.py` | All main modules import without errors |
| `tests/test_cnn_smoke.py` | CNN forward pass runs on CPU with synthetic data |

## Notes

- The CNN smoke test runs on CPU; `cnn_device` in `pipeline.yaml` does not affect it.
- All tests are fast (no training loops, no real data loading).
