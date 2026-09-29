import numpy as np
import pandas as pd

from features import build_features


def test_build_features():
    raw = pd.DataFrame({
        "show_on_report": ["Yes", "Yes", "No"],
        "submit_approved": [np.e, 1.0, 500.0],
        "submitted": pd.to_datetime(["2021-05-01", "2023-01-01", "2020-01-01"]),
        "deferred": ["Yes", "No", "No"],
        "use_deception": ["Yes", None, None],
        "board": ["BREB", None, "CREB"],
        "typeofboardreview": ["Full Board Review", "Expedited Review", "Expedited Review"],
        "typeofstudy": ["Clinical", "Behavioural", "Clinical"],
        "facultyname": ["Arts", "Arts", "Arts"],
        "institutionname": ["UBC", "UBC", "UBC"],
        "departmentname": ["History", None, "History"],
    })
    df = build_features(raw)
    assert len(df) == 2
    assert df["deferred"].tolist() == [1, 0]
    assert df["use_deception"].tolist() == [1, 0]
    assert df["board_idx"].nunique() == 2
    assert df["typeofstudy"].tolist() == ["Clinical", "Behavioural"]
    assert df["clinical"].tolist() == [1, 0]
    assert df["full_board_review"].tolist() == [1, 0]
    assert df["departmentname"].tolist() == ["History", "Missing"]
