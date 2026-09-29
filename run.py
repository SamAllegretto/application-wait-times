import os
from itertools import combinations
from pathlib import Path

import arviz as az

from db import fetch, get_engine
from features import OUTCOMES, build_features
from model import WaitTimeModel, export_all

CANDIDATES = ["minimal_risk_criteria"]
GROUPS = ["board_idx"]
OUTPUTS = Path(__file__).parent / "outputs"

if __name__ == "__main__":
    df = build_features(fetch(os.environ["SOURCE_SCHEMA"], os.environ["SOURCE_TABLE"]))
    variables = CANDIDATES + GROUPS
    models = []
    OUTPUTS.mkdir(exist_ok=True)
    for outcome in OUTCOMES:
        fitted = [
            WaitTimeModel([v for v in c if v in CANDIDATES], [v for v in c if v in GROUPS], outcome)
            for r in range(len(variables) + 1)
            for c in combinations(variables, r)
        ]
        for m in fitted:
            print(f"Fit {outcome}: {m.name}")
            m.fit(df)
            m.idata.to_netcdf(OUTPUTS / f"{outcome}_{m.name}.nc")
        print(outcome)
        print(az.compare({m.name: m.idata for m in fitted}))
        models += fitted

    results = export_all(models, df, variables)
    results.to_parquet(OUTPUTS / "results.parquet", index=False)
    with get_engine() as engine:
        # Postgres allows at most 65535 bind parameters per INSERT.
        results.to_sql(os.environ["RESULTS_TABLE"], engine, schema=os.environ["RESULTS_SCHEMA"],
                       if_exists="replace", index=False, method="multi", chunksize=1000)
    print(f"{len(results)} rows written")
