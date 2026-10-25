# Thin wrappers around uv; every target works on a fresh clone after `make install`.
.PHONY: install test lint format typecheck check figures reproduce-figures report clean

install:
	uv sync --locked

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

check: lint typecheck test

# One script per figure ID in carryon/viz/fig_*.py; each reads runs/ and writes figures/<ID>.{png,svg}.
figures:
	@scripts=$$(ls carryon/viz/fig_*.py 2>/dev/null); \
	if [ -z "$$scripts" ]; then echo "no figure scripts yet"; fi; \
	for s in $$scripts; do echo "== $$s"; uv run python $$s || exit 1; done

reproduce-figures: figures

report:
	mkdir -p report/build
	cat report/chapter_0.md report/chapter_A.md report/chapter_B.md report/chapter_C.md > report/build/report.md
	@echo "wrote report/build/report.md"

clean:
	rm -rf report/build .pytest_cache .mypy_cache .ruff_cache
