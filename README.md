# PyQu Testing Agent

`pyqu` is a command-line tool that automates test generation and regression checking for Python projects using Pynguin.

In short, it:

- creates (or reuses) a virtual environment,
- installs project dependencies,
- discovers target modules,
- runs Pynguin to generate tests,
- runs generated tests with pytest,
- compares `before` vs `after` results to detect regressions.

## Main use case

This is useful when you have two versions of code (for example `before/` and `after/`) and want to verify that the `after` version did not introduce new failures.

The comparison rule is simple:

- if a test passes in `before` and fails in `after`, it is marked as a **REGRESSION**,
- if behavior is the same (both pass or both fail), it is **OK**,
- if `after` fixes a failure, it is marked as **IMPROVED**.

## Quick run

From the project root:

```bash
python __main__.py --project-path . --all-modules
```

Or target a single module:

```bash
python __main__.py --project-path . --module-name your_package.your_module
```

## Outputs

By default:

- generated tests go to `pynguin_tests/`
- reports go to `pynguin-report/`

