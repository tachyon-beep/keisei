# Contributing to Keisei

Thank you for your interest in contributing to Keisei.

## Development Setup

```bash
# Clone the repository
git clone git@github.com:tachyon-beep/keisei.git
cd keisei

# Install Python dependencies (editable, with dev tools)
uv pip install -e ".[dev]"

# Build the Rust engine
cd shogi-engine && cargo build --release && cd ..

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Python tests
uv run pytest

# Rust tests
cd shogi-engine && cargo test

# Linting
uv run ruff check .

# Type checking
uv run mypy keisei/
```

## Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes. Ensure all tests pass and linting is clean.
3. Update the `[Unreleased]` section of `CHANGELOG.md` with a brief description
   of your changes.
4. Open a pull request against `main`.

## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
style:

- `feat:` new features
- `fix:` bug fixes
- `refactor:` code changes that neither fix bugs nor add features
- `test:` adding or updating tests
- `docs:` documentation changes
- `chore:` maintenance tasks

Scoped commits (e.g., `feat(shogi-gym):`) are used when the change is specific
to a single crate or module.

## Code Style

- **Python:** Formatted and linted by [ruff](https://docs.astral.sh/ruff/).
  Type-checked with [mypy](http://mypy-lang.org/) in strict mode. Line length
  limit is 100 characters.
- **Rust:** Standard `rustfmt` formatting, no Clippy warnings allowed.
- All tool configuration lives in `pyproject.toml` (Python) or
  `Cargo.toml` (Rust). No standalone config files.

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/tachyon-beep/keisei/issues)
with:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python/Rust version and OS
