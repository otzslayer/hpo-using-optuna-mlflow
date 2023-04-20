from typing import Tuple

import numpy as np
import pandas as pd


def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv("./data/train.csv")

    y = np.log1p(df["SalePrice"].ravel())
    X = df.drop(["Id", "Alley", "PoolQC", "Fence", "MiscFeature", "SalePrice"], axis=1)
    return X, y 

def get_data_description(df: pd.DataFrame, **kwargs): 
    return df.describe(**kwargs)
