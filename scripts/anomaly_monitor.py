import numpy as np
from src.utils.tensorboard_logger import TensorBoardLogger
from src.utils.logger import logger

class RealTimeAnomalyMonitor:
    """
    Monitors real-time anomalies by running model predictions on streaming data
    and logging the anomaly scores dynamically.
    """

    def __init__(self, model):
        self.model = model
        self.tensorboard_logger = TensorBoardLogger()

    def detect_anomalies(self, num_steps=100, threshold=0.8):
        """
        Runs real-time anomaly detection.
        
        Args:
        - num_steps (int): Number of iterations for real-time monitoring.
        - threshold (float): Anomaly threshold for logging alerts.
        """

        input_dim = 100  # Adjust based on model input size
        anomaly_count = 0  # Track detected anomalies

        logger.log(f"🔍 Starting real-time anomaly detection for {num_steps} steps...")

        for step in range(num_steps):
            # Generate synthetic streaming data (simulate real-time input)
            input_data = np.random.rand(1, input_dim).astype(np.float32)
            output = self.model.predict(input_data)

            # Compute anomaly score (mean activation)
            anomaly_score = np.mean(output)

            # Log to TensorBoard
            self.tensorboard_logger.log_scalar("Anomaly Score", anomaly_score, step)
            self.tensorboard_logger.log_histogram("Real-Time Predictions", output, step)

            # Detect anomaly based on threshold
            if anomaly_score > threshold:
                anomaly_count += 1
                logger.log(f"🚨 Anomaly Detected at Step {step+1}! Score: {anomaly_score:.4f}", level="error")

        logger.log(f"✅ Anomaly detection complete. Total anomalies detected: {anomaly_count}/{num_steps}")

        # Print TensorBoard command for monitoring
        print(f"\n🎯 View live anomaly detection in TensorBoard:\n➡ tensorboard --logdir=logs/")
