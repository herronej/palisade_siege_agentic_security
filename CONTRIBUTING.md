# Contributing

## Running SIEGE against your own defense

This is the contribution we most want. The corpus is declarative — one file is
one instance, one directory is one class — and the runner takes any gate stack
exposing the same admission interface, so SIEGE is not tied to PALISADE.

Implement the admission interface in `src/siege/eval/runner.py` and pass your
stack to the session runner. Score with `siege.scorer`, which returns soft wins
and hard wins separately; report both. A defense that closes soft wins and
leaves hard wins where they are has moved inspection, not enforcement, and the
separation exists to make that visible.

If you close the propagation residual — the one experiment Section VI-A names
as most obviously called for — please tell us. We will cite it.

## Adding an attack class

1. Add a template under `src/siege/templates/`, following an existing class.
2. Every action must name its gate, carry an attack flag, and — for chained
   attacks — the capability tag whose taint must be tracked to the sink.
3. Declare a **programmatic** success criterion. Only an indeterminate verdict
   may defer to a language-model judge, whose default is a deterministic stub
   returning "not met", so offline runs stay reproducible.
4. Generate instances into `src/siege/corpus/<class>/`.
5. Run `uv run pytest tools/tests/test_release_safety.py`. Payload
   destinations must use reserved placeholder hosts — see
   [`docs/palisade/RELEASE_SAFETY.md`](docs/palisade/RELEASE_SAFETY.md). This
   is not optional; two live mining pools reached the corpus before this check
   existed.
6. Regenerate the datasheet: `uv run python -m tools.corpus_datasheet`.

New classes shift published denominators. Say so in the PR.

## Adding a contract

Contracts are deterministic, model-free checks over claim types, authored as
plain policy files so a domain expert extends enforcement without touching gate
code. Drop one in `palisade_contracts/` following the existing files; an
operator directory loads beside the built-ins.

Contracts are coverage-defined: a claim outside declared coverage passes. State
your coverage explicitly and do not imply a guarantee the check does not make.

## Adding an analysis module

Every reported number names the module that produces it. If you add a module:

1. Put it in `tools/`, reading paths from `palisade.paths` — never by counting
   parent directories.
2. Write its report into `docs/palisade/`, starting with the provenance banner
   naming the module.
3. Add a row to the result map in `docs/palisade/README.md`.
4. Add a test in `tools/tests/`.

`tools/tests/test_result_map.py` enforces 1–3: a module absent from the map, or
a map row pointing at nothing, fails the suite.

## House rules

- **Do not float dependencies.** Versions are pinned because the reported
  numbers were produced against them. `pydantic-ai>=1.90` resolves to a 2.x
  that removes an API G2 depends on.
- **Never let a check's own failure read as a pass.** Both the payload scan and
  the corpus-count tests assert their input is non-empty first. Two bugs during
  this repository's extraction were regexes that silently matched nothing and
  reported clean.
- **The slow tier can only tighten a decision.** A quarantined model's verdict
  is one-directional data: it can deny or rewrite, never loosen a deterministic
  block or clear a label. Do not add a path that lets it.
- **Nothing the model emits is a registry write.** Enforcement state lives
  outside the model's context window. Keep it there.

## Tests

```bash
uv run pytest
```

Deterministic, no network. Tests needing the VISTA reference deployment skip
automatically; tests needing a served model or the Slurm container are marked
`served_model` and `slurm`.
