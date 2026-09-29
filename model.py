from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import gaussian_kde

from features import OUTCOMES, label

STATS = {
    "Mean": np.mean,
    "Median": np.median,
    "95% CI lower (2.5%)": lambda s: np.quantile(s, 0.025),
    "95% CI upper (97.5%)": lambda s: np.quantile(s, 0.975),
}


@dataclass
class WaitTimeModel:
    predictors: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    outcome: str = "total"
    idata: object = None

    @property
    def variables(self) -> list[str]:
        return self.predictors + self.groups

    @property
    def name(self) -> str:
        return "+".join(self.variables) or "intercept"

    def fit(self, df: pd.DataFrame, **sample_kwargs):
        coords = {"predictor": self.predictors} | {g: np.arange(df[g].max() + 1) for g in self.groups}
        with pm.Model(coords=coords):
            mu = pm.Normal("intercept", 3, 2)
            if self.predictors:
                X = df[self.predictors].to_numpy(float)
                self.x_mean, self.x_std = X.mean(0), X.std(0)
                mu = mu + (X - self.x_mean) / self.x_std @ pm.Normal("beta", 0, 1, dims="predictor")
            for g in self.groups:
                z = pm.Normal(f"{g}_z", 0, 1, dims=g)
                mu = mu + (z * pm.HalfNormal(f"{g}_sd", 1))[df[g].to_numpy()]
            pm.LogNormal("wait", mu=mu, sigma=pm.HalfNormal("sigma", 1), observed=df[OUTCOMES[self.outcome]].to_numpy(float))
            self.idata = pm.sample(**sample_kwargs)
            pm.compute_log_likelihood(self.idata)
        return self.idata

    def predict(self, values: dict, rng: np.random.Generator) -> np.ndarray:
        post = self.idata.posterior
        mu = post["intercept"].values.ravel()
        if self.predictors:
            x = (np.array([values[p] for p in self.predictors], float) - self.x_mean) / self.x_std
            mu = mu + post["beta"].values.reshape(-1, len(self.predictors)) @ x
        for g in self.groups:
            mu = mu + post[f"{g}_z"].sel({g: values[g]}).values.ravel() * post[f"{g}_sd"].values.ravel()
        return np.exp(rng.normal(mu, post["sigma"].values.ravel()))

    def export(self, df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        blocks = []
        for combo in product(*[sorted(df[v].unique()) for v in self.variables]):
            values = dict(zip(self.variables, combo))
            block = density(self.predict(values, rng))
            for v, val in values.items():
                block[v.removesuffix("_idx")] = label(df, v, val)
            blocks.append(block)
        return pd.concat(blocks, ignore_index=True).assign(model=self.name, outcome=self.outcome)


def density(samples: np.ndarray, n: int = 200) -> pd.DataFrame:
    # KDE on the log scale: a wait-time sample has no mass at x <= 0.
    grid = np.linspace(*np.quantile(samples, [0.005, 0.995]), n)
    y = gaussian_kde(np.log(samples))(np.log(grid)) / grid
    rows = [pd.DataFrame({"x": grid, "y": y, "series": "density"})]
    for name, stat in STATS.items():
        v = stat(samples)
        rows.append(pd.DataFrame({"x": [v, v], "y": [0.0, y.max()], "series": name}))
    return pd.concat(rows, ignore_index=True)


def export_all(models: list[WaitTimeModel], df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    cols = [v.removesuffix("_idx") for v in variables]
    out = pd.concat([m.export(df) for m in models], ignore_index=True)
    out = out.reindex(columns=["model", "outcome", "x", "y", "series", *cols])
    out[cols] = out[cols].fillna("Unknown")
    return out
