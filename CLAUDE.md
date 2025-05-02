# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

# Install dependencies (via uv)
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload

## Tests

# Run full suite
python tests.py

# Skip streaming tests
python tests.py --no-streaming

# Simple tests only
python tests.py --simple

# Filter single test
python tests.py --filter <test_name>

## Code Style Guidelines

- Use Black and isort for formatting and import order: 1) stdlib, 2) third-party, 3) local modules.
- Indent with 4 spaces; trailing whitespace forbidden.
- Use snake_case for functions/variables, PascalCase for classes, type hints everywhere.
- Pydantic models must extend BaseModel and validate fields.
- Use f-strings for string interpolation; avoid string concatenation.
- Catch exceptions, log via `logger`, and raise HTTPException for client errors.
- Follow existing logging filter and ColorizedFormatter conventions for console output.
