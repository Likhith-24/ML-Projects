# Plays a role in reading the data from the source and splitting it into train and test data
# Also, it is responsible for saving the data in the artifacts folder
# We have separate big data team in companies who are responsible for collecting data and storing it in any database

import os, sys #For custom Exception
from src.exception import CustomException #To call CustomException
from src.logger import logging
import pandas as pd #For working with dataframes

from sklearn.model_selection import train_test_split #For splitting the data into train and test data
from dataclasses import dataclass #Used for creating class variables

from src.components.data_transformation import DataTransformation #To get the preprocessor object
from src.components.data_transformation import DataTransformationConfig #To get the path of preprocessor object


@dataclass #Used for defining custom class variables
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # Path for output train data
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    

class DataIngestion: #For other functions inside class within custom class
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # The 3 paths are saved in ingestion_config

    def initiate_data_ingestion(self): #Creating our own function
        # It is used for making the data stored in DB more complex and then reading it
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv') # Reading the data from the source
            logging.info("Data read successfully as Data Frame")
        
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #Creating the path for train data
        
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
        
            logging.info("Train test split is started")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) #Splitting the data into train and test data

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) #Saving the train data
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) #Saving the train data
            logging.info("Data Ingestion is completed")
        
            return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
            )
    
    

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
