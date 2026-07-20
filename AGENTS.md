# Agent instructions

## Running the Python tests

Always run pytest from the `py` directory with Python's module entry point and
importlib import mode:

```powershell
Set-Location .\py
python -m pytest --import-mode=importlib .\test -q
```

For a targeted test, retain the same entry point and import-mode option:

```powershell
python -m pytest --import-mode=importlib .\test\test_curriculum.py -q
```

On the Linux training node, use the equivalent command:

```bash
cd /workspace/chess/source/py
python -m pytest --import-mode=importlib test -q
```

Do not invoke `pytest` or `uv run pytest` directly, and do not omit
`--import-mode=importlib`. This repository has a top-level Python package named
`py`; pytest's default import mode can consequently try to import tests as
`py.test.*` and fail during collection with:

```text
ModuleNotFoundError: No module named 'py.test'; 'py' is not a package
```

Treat that message as a test-runner invocation error, not a failing project
test. Rerun with the command above.
