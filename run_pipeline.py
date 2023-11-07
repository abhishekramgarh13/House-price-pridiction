from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r"/home/abhishek13/house_price_pred/data/housing.csv/housing.csv")