# Thin wrappers around uv; every target works on a fresh clone after `make install`.
.PHONY: install test lint format typecheck check figures reproduce-figures report container clean

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

# One script per figure ID in carryon/viz/fig_*.py; each reads runs/ and writes
# figures/<ID>.{png,svg,inputs.json}.
figures:
	@scripts=$$(ls carryon/viz/fig_*.py 2>/dev/null); \
	if [ -z "$$scripts" ]; then echo "no figure scripts yet"; fi; \
	for s in $$scripts; do echo "== $$s"; uv run python $$s || exit 1; done

reproduce-figures: figures

# Chapters + figures/ -> one self-contained report/build/report.html (run `make figures` first).
report:
	uv run python -m carryon.viz.report

# Build the Apptainer image next to the repo (needs sudo; temp files go to carryon-data, not /tmp).
DATA ?= ../carryon-data
SIF ?= $(DATA)/sif/carryon.sif
container:
	mkdir -p $(DATA)/sif $(DATA)/cache/apptainer-tmp
	sudo APPTAINER_TMPDIR=$(abspath $(DATA))/cache/apptainer-tmp apptainer build --force $(SIF) container/carryon.def
	@ls -lh $(SIF)

clean:
	rm -rf report/build .pytest_cache .mypy_cache .ruff_cache
