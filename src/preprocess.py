from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def identify_column_types(X: pd.DataFrame) -> Tuple[List, List]:
    num_cols = X.select_dtypes("float").columns.tolist()
    cat_cols = X.select_dtypes("object").columns.tolist()

    return num_cols, cat_cols


def make_preprocess_pipeline(
    num_cols: List[str], cat_cols: List[str]
) -> ColumnTransformer:
    num_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", MinMaxScaler()),
        ]
    )

    cat_processor = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value="None"))]
    )

    preprocessor = ColumnTransformer(
        [("num", num_processor, num_cols), ("cat", cat_processor, cat_cols)]
    )
    return preprocessor
