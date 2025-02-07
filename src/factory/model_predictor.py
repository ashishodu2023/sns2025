import time
import numpy as np
from src.utils.logger import logger

class ModelPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        start_time = time.time()
        output = self.model.predict(input_data)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1e6  # Convert to microseconds (µs)
        logger.log(f"Inference Time: {inference_time:.2f} µs")
        return output, inference_time
