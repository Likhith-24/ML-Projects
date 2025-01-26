from src.logger import logging

class ModelPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        logging.info("Starting predictions")
        predictions = self.model.predict(features)
        logging.info("Predictions completed")
        return predictions