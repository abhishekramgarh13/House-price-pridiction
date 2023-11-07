import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main



def main():
    st.title("End to End House Price Pipeline with ZenML")

   
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the  median housing price in any district in California. You can input the features of the block listed below and get the  median housing price in any district in California.
    | Models        | Description   | 
    | ------------- | -     | 
    |  longitude  |  Longitude value for the block in California, USA  |
    |  latitude  |   Latitude value for the block in California, USA  |
    |  housing_median_age  |  Median age of the house in the block   |
    |  total_rooms  |  Count of the total number of rooms (excluding bedrooms) in all houses in the block |
    |  total_bedrooms  |  Count of the total number of bedrooms in all houses in the block   |
    |  population  |   Count of the total number of population in the block  |
    |  households  |   Count of the total number of households in the block  |
    |  median_income  |   Median of the total household income of all the houses in the block  |
    """
    )
    longitude = st.number_input("Longitude")
    latitude = st.number_input("Latitude")
    housing_median_age = st.number_input("Housing_Median_Age")
    total_rooms = st.number_input("Total_Rooms")
    total_bedrooms = st.number_input("Total_Bedrooms")
    population = st.number_input("Population")
    households  = st.number_input("Households")
    median_income = st.number_input("Median_Income")

    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

        df = pd.DataFrame(
            {
                "longitude": [longitude],
                "latitude": [latitude],
                "housing_median_age": [housing_median_age],
                "total_rooms": [total_rooms],
                "total_bedrooms": [total_bedrooms],
                "population": [population],
                "households":[households],
                "median_income": [median_income],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "The median housing price in this block in California:-{}".format(
                pred
            )
        )
   
if __name__ == "__main__":
    main()