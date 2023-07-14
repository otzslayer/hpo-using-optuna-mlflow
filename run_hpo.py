import argparse

import pandas as pd
import optuna
import catboost

from src.data import load_data
from src.preprocess import identify_column_types, make_preprocess_pipeline
from src.callback import make_mlflow_callback


class Objective(object):
    def __init__(self, pool):
        self.pool = pool

    def __call__(self, trial):
        pool = self.pool

        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3)
        subsample = trial.suggest_float("subsample", 0.75, 1)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-5, 1e-1, log=True)

        params = {
            "loss_function": "RMSE",
            "depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "l2_leaf_reg": l2_leaf_reg,
        }

        cv_result = catboost.cv(
            pool=pool,
            params=params,
            num_boost_round=1000,
            nfold=5,
            seed=0,
            early_stopping_rounds=30,
            verbose=False,
        )

        rmsle = cv_result["test-RMSE-mean"].min()

        return rmsle


def main():
    parser = argparse.ArgumentParser(
        description="HPO Experiment Tracking using Optuna and MLflow."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        help="The number of trials for exploring hyperparmeter space.",
    )

    args = parser.parse_args()

    print("Load data...")
    X, y = load_data()

    num_cols, cat_cols = identify_column_types(X)

    print("Preprocess the data...")
    preprocessor = make_preprocess_pipeline(num_cols, cat_cols)

    X_ = pd.DataFrame(preprocessor.fit_transform(X, y), columns=num_cols + cat_cols)

    pool = catboost.Pool(X_, label=y, cat_features=cat_cols)
    objective = Objective(pool)

    study = optuna.create_study(
        study_name="house_price_prediction",
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(max_resource="auto"),
    )
    #test0522 by paik
    print("Optimize a model...")
    mlflow_cb = make_mlflow_callback(tracking_uri="mlruns", metric_name="RMSLE")
    study.optimize(
        objective, n_trials=args.n_trials, callbacks=[mlflow_cb], show_progress_bar=True
    )


if __name__ == "__main__":
    main()
