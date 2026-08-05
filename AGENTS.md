# Agent instructions

## Rework authority

Read `REWORK.md` completely before changing rework code. Work only on the phase
the user has authorized. Preserve unrelated changes and make feature-sized
commits after relevant validation.

At every handoff, report completed work, outstanding phase work, commits,
validation results, changes needing special review, and unresolved decisions.
Only the user accepts a phase or authorizes another phase.

## Models and boundaries

Strong typing means precise canonical models, not a separate model for every
layer. Define each semantic concept once and reuse that type wherever its
meaning and representation are unchanged.

Create a boundary type only when serialization, ownership, validation, or an
external dependency requires a genuinely different representation. Convert at
that boundary exactly once. A conversion must perform a meaningful change; do
not add adapters that copy fields one-for-one, rename fields without necessity,
or translate between duplicate enums. When both sides are under this project's
control, change them to share the canonical type instead.

Model genuine alternatives with discriminated unions whose variants make
invalid combinations unrepresentable. Do not encode variants as a mode plus
unrelated nullable or ignored fields. Do not create one-member enums or fields
that can only contain one invariant value; express the invariant in the type's
name and contract.

Configuration has one canonical typed owner per component. Pass cohesive
configuration objects instead of repeatedly expanding and reconstructing long
parameter lists. Defaults and validation live with that canonical
configuration. Transport-specific configuration contains only transport
concerns and composes the shared component configuration rather than mirroring
it.

Use frozen dataclasses for internal domain data and frozen Pydantic models with
`extra='forbid'` at serialization boundaries. Do not introduce wrapper classes,
protocols, or names such as `Native`, `Optimized`, or `Legacy` unless multiple
current implementations make the distinction real. Remove superseded layers
when their replacement becomes authoritative.

## Remote execution

Use `deployment/setup_remote.sh` for a fresh compute node. It clones the
requested revision, installs the locked training dependencies, builds the
Release C++ extension, exports `ENGINE_SOURCE_REVISION`, and starts the supplied
runner command. Keep production run configuration and approval files explicit.

## Pytest

Run tests from `py`:

```powershell
python -m pytest --import-mode=importlib .\test -q
```

Always retain `--import-mode=importlib`; otherwise the repository's `py` package
can cause `No module named 'py.test'` during collection.
