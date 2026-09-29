import arviz as az
import numpy as np
import pandas as pd

from model import WaitTimeModel, export_all


def test_fit_and_export():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"flag": rng.integers(0, 2, n), "g_idx": rng.integers(0, 3, n)})
    df["g"] = df["g_idx"].map({0: "A", 1: "B", 2: "C"})
    flag_std = (df["flag"] - df["flag"].mean()) / df["flag"].std(ddof=0)
    df["submit_approved"] = np.exp(3.0 + 0.4 * flag_std + rng.normal(0, 0.5, n))

    kw = dict(draws=500, tune=500, chains=2, random_seed=0, progressbar=False)
    full = WaitTimeModel(["flag"], ["g_idx"])
    base = WaitTimeModel()
    full.fit(df, **kw)
    base.fit(df, **kw)

    post = full.idata.posterior
    assert full.name == "flag+g_idx" and base.name == "intercept"
    assert abs(float(post["beta"].mean()) - 0.4) < 0.1
    assert abs(float(post["sigma"].mean()) - 0.5) < 0.1
    assert az.compare({full.name: full.idata, base.name: base.idata}).index[0] == "flag+g_idx"

    out = export_all([full, base], df, ["flag", "g_idx"])
    assert list(out.columns) == ["model", "outcome", "x", "y", "series", "flag", "g"]
    assert len(out) == (2 * 3 + 1) * 208
    f = out[out["model"] == "flag+g_idx"]
    assert set(f["flag"]) == {"Yes", "No"} and set(f["g"]) == {"A", "B", "C"}
    assert (out["outcome"] == "total").all()
    b = out[out["model"] == "intercept"]
    assert (b[["flag", "g"]] == "Unknown").all().all()
    assert set(out["series"]) == {"density", "Mean", "Median", "95% CI lower (2.5%)", "95% CI upper (97.5%)"}

    median = b.loc[b["series"] == "Median", "x"].iloc[0]
    assert abs(median / df["submit_approved"].median() - 1) < 0.1
