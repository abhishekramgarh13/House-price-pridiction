import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handel_data(self,data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    strategy for preprocessing data
    """
    def handel_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Data
        """
        try:
            data = data.drop(["ocean_proximity"],axis=1)

            # calculating upperbond of column that in present outliers
            upper_bond_total_rooms= data['total_rooms'].quantile(0.75)+1.5*(data['total_rooms'].quantile(0.75) - data['total_rooms'].quantile(0.25))
            upper_bond_total_bedrooms= data['total_bedrooms'].quantile(0.75)+1.5*(data['total_bedrooms'].quantile(0.75) - data['total_bedrooms'].quantile(0.25))
            upper_bond_population= data['population'].quantile(0.75)+1.5*(data['population'].quantile(0.75) - data['population'].quantile(0.25))
            upper_bond_households= data['households'].quantile(0.75)+1.5*(data['households'].quantile(0.75) - data['households'].quantile(0.25))
            upper_bond_median_income= data['median_income'].quantile(0.75)+1.5*(data['median_income'].quantile(0.75) - data['median_income'].quantile(0.25))
            upper_bond_median_house_value= data['median_house_value'].quantile(0.75)+1.5*(data['median_house_value'].quantile(0.75) - data['median_house_value'].quantile(0.25))

            # replacing outliers with upperbond
            data.loc[data['total_rooms']>upper_bond_total_rooms,'total_rooms']=upper_bond_total_rooms
            data.loc[data['total_bedrooms']>upper_bond_total_bedrooms,'total_bedrooms']=upper_bond_total_bedrooms
            data.loc[data['population']>upper_bond_population,'population']=upper_bond_population
            data.loc[data['households']>upper_bond_households,'households']=upper_bond_households
            data.loc[data['median_income']>upper_bond_median_income,'median_income']=upper_bond_median_income
            data.loc[data['median_house_value']>upper_bond_median_house_value,'median_house_value']=upper_bond_median_house_value
            
            #filling null value with median of column
            data["total_bedrooms"].fillna(data["total_bedrooms"].median(), inplace=True)

            return data
        except Exception as e:
            logging.error("Error in preprocessing data {}".format(e))
            raise e
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handel_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test 
        """
        try:
            X = data.drop(["median_house_value"], axis=1)
            y = data["median_house_value"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e
        
class DataCleaning:
    """
    class for cleaning data which processes the data and divides it into train and test
    """
    def __init__(self, data : pd.DataFrame, strategy: DataStrategy) :
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        handle data
        """
        try:
            return self.strategy.handel_data(self.data)
        except Exception as e:
            logging.error("Error in handling data : {}".format(e))
            raise e