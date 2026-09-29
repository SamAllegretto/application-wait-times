# %% Load
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from db import fetch
from features import OUTCOMES

WAIT_COL = OUTCOMES["total"]

df = fetch(os.environ["SOURCE_SCHEMA"], os.environ["SOURCE_TABLE"])
df["year"] = df["submitted"].dt.year
print(df.shape)
df[WAIT_COL].describe()

# %% Wait time distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
df[WAIT_COL].hist(bins=100, ax=ax1)
ax1.set(title=WAIT_COL, xlabel="days")
np.log(df[WAIT_COL]).hist(bins=100, ax=ax2)
ax2.set(title=f"log({WAIT_COL})", xlabel="log days")
plt.show()

# %% Median wait by year and flag
print(df.groupby("year")[WAIT_COL].agg(["count", "median"]))
for col in ["minimal_risk_criteria", "deferred", "graduate_student_led", "show_on_report"]:
    print(df.groupby(col, dropna=False)[WAIT_COL].agg(["count", "median"]), "\n")

# %% Stage durations
stages = [
    "submit_depappr", "depappr_rebmeeting", "rebmeeting_provisoentered",
    "provisoentered_respondtoproviso", "respondtoproviso_approved",
    "time_with_researcher", "time_with_reb",
]
pd.DataFrame({
    "nulls": df[stages].isna().sum(),
    "negative": (df[stages] < 0).sum(),
    "median": df[stages].median(),
})
