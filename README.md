# application-wait-times
Bayesian modeling for application wait times.

## Setup

1. `uv sync`
2. `cp .env.example .env`. Fill in the database and table values.

## Run

- `explore.py`: run the `# %%` cells in VS Code.
- `uv run python run.py`: fits every combination of `CANDIDATES` and `GROUPS`, prints a LOO comparison, and replaces the results table with the posterior predictive densities.
- `uv run pytest`: tests, no database needed.

## Public repo

`.env`, `data/`, and `outputs/` are gitignored. Do not commit credentials, hostnames, table names, or row-level data.
