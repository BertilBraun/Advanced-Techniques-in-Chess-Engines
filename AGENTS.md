# Agent instructions

## Pytest

Run tests from `py`:

```powershell
python -m pytest --import-mode=importlib .\test -q
```

Always retain `--import-mode=importlib`; otherwise the repository's `py` package
can cause `No module named 'py.test'` during collection.
