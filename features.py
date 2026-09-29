import pandas as pd

OUTCOMES = {"total": "submit_approved", "reb": "time_with_reb", "researcher": "time_with_researcher"}
GROUPS = ["board", "institutionname", "facultyname", "departmentname"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["show_on_report"].eq("Yes")].copy()

    # Null is 0: several columns hold only "Yes" or null.
    for col in df.columns:
        vals = set(df[col].dropna().unique()) if df[col].dtype in (object, "str") else set()
        if vals and vals <= {"Yes", "No"}:
            df[col] = df[col].eq("Yes").astype(int)

    df["clinical"] = df["typeofstudy"].eq("Clinical").astype(int)
    df["full_board_review"] = df["typeofboardreview"].eq("Full Board Review").astype(int)

    # "Missing", not "Unknown": the export uses "Unknown" for a variable the model does not use.
    for col in GROUPS:
        df[col] = df[col].fillna("Missing")
        df[f"{col}_idx"] = df[col].astype("category").cat.codes

    return df


def label(df: pd.DataFrame, col: str, value) -> str:
    if col.endswith("_idx"):
        return str(df.loc[df[col] == value, col.removesuffix("_idx")].iloc[0])
    if set(df[col].unique()) <= {0, 1}:
        return "Yes" if value else "No"
    return str(value)
