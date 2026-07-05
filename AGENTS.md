# Agent Instructions

## Static Typing Discipline

Keep VSCode/Pylance and Pyright clean when editing this repository. Do not
write code that relies on runtime knowledge while leaving the type checker with
only broad `str`, `object`, or `None` types.

- Use existing `Literal` aliases, or add a narrowly scoped alias, for closed
  value sets such as retrieval modes, transcript roles, page names, block
  types, command names, and provider names. Do not pass an arbitrary `str` into
  APIs that accept only a fixed set of string values.
- If a helper function proves that an optional mapping or object has a narrower
  type, annotate it with `TypeGuard`; a plain `bool` helper does not narrow
  values for Pylance/Pyright.
- Annotate functions that always raise with `Never`, not `None`, when callers
  depend on that branch being non-returning.
- Use `Protocol` for injected services and test fakes when code depends on a
  structural interface rather than a concrete class.
- Avoid indexing or calling methods on values typed as `object`. Narrow with
  `isinstance`, use a `TypedDict`, or model the response shape explicitly.
- Keep dataclass and Pydantic model fields as precise as their real contents,
  especially tuple/list element types and warning/result payloads.

## Verification

Before committing changes that touch code, run:

```powershell
pyright src tests --pythonpath .venv\Scripts\python.exe
uv run --extra dev ruff check .
uv run --extra dev pytest -q
```

Use the `dev` extra for tests because PDF-related coverage depends on
`reportlab`.
