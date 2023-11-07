import logging 
from zenml import step
import pandas as pd
from src.evaluation import MSE,RMSE,R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow 
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(model : RegressorMixin,
    X_test: pd.DataFrame,
    y_test : pd.Series
)-> Tuple[
    Annotated[float, "r2"],
    Annotated[float, "rmse"],
]:
    """
    Evaluates the model on the ingested data.
    Args:
        Model : RegressoeMixin
        X_test: pd.DataFreame
        y_test: pd.Series
    Returns:
        r2 : r2_score of model
        rmse : root_mean_square_error of model
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e