# Agent instructions

## Rework authority

Read `documentation/architecture/platform-rework.md` completely before changing rework code. For Python runtime
work, also read `documentation/architecture/python-runtime-rework.md` completely. Work only on the phase the user
has authorized. Preserve unrelated changes and make feature-sized commits after relevant validation.

At every handoff, report completed work, outstanding phase work, commits,
validation results, changes needing special review, and unresolved decisions.
Only the user accepts a phase or authorizes another phase.

## Remote execution

Use `deployment/setup_remote.sh` for a fresh compute node. It clones the
requested revision, installs the locked training dependencies, builds the
Release C++ extension, exports `ENGINE_SOURCE_REVISION`, and starts the supplied
runner command. Keep production run configuration and approval files explicit.

### Current Vast validation node

Connect to the current rented validation node from Windows with the dedicated
Vast key and the TensorBoard/local-service forward:

```powershell
ssh -i C:\Users\berti\.ssh\codex_vast_ed25519 -p 56488 root@171.101.230.38 -L 8080:localhost:8080
```

The private key is local-only and must never be copied into the repository or
onto the node. Read `/etc/vast-agents-guide.md` completely before changing the
instance. The node filesystem is ephemeral; copy required run evidence off the
node before it is destroyed or recycled.

## Pytest

Run tests from `py`:

```powershell
python -m pytest --import-mode=importlib .\test -q
```

Always retain `--import-mode=importlib`; otherwise the repository's `py` package
can cause `No module named 'py.test'` during collection.
