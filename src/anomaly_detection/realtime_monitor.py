import numpy as np
from src.utils.tensorboard_logger import TensorBoardLogger
from src.utils.logger import logger

class RealTimeAnomalyMonitor:
    def __init__(self, model):
        self.model = model
        self.tensorboard_logger = TensorBoardLogger()

    def detect_anomalies(self, num_steps=100):
        input_dim = 100
        for step in range(num_steps):
            input_data = np.random.rand(1, input_dim).astype(np.float32)
            output, _ = self.model.predict(input_data)

            anomaly_score = np.mean(output)
            self.tensorboard_logger.log_scalar("Anomaly Score", anomaly_score, step)
            self.tensorboard_logger.log_histogram("Real-Time Predictions", output, step)

            logger.log(f"[Step {step+1}/{num_steps}] Anomaly Score: {anomaly_score:.4f}")
