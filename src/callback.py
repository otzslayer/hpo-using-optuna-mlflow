from optuna.integration.mlflow import MLflowCallback


def make_mlflow_callback(tracking_uri: str, metric_name: str) -> MLflowCallback:
    cb = MLflowCallback(tracking_uri=tracking_uri, metric_name=metric_name)
    return cb
