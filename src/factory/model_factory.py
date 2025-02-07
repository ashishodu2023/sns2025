from src.factory.model_loader import ModelLoader
from src.factory.model_trainer import ModelTrainer
from src.factory.model_predictor import ModelPredictor
from src.anomaly_detection.realtime_monitor import RealTimeAnomalyMonitor
from src.utils.benchmarking import ModelBenchmark
from src.utils.logger import logger

class ModelFactory:
    """
    Factory class to handle dynamic model operations including:
    - Loading the model
    - Training the model
    - Making predictions
    - Benchmarking the model
    - Running real-time anomaly detection
    """

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model_loader = ModelLoader(model_path)
        self.model = self.model_loader.model  # Load the model

    def train(self, input_data, output_data, epochs=10, batch_size=32):
        """
        Trains the model using the ModelTrainer module.
        """
        logger.log(f"Initializing training for {self.model_name}...")
        trainer = ModelTrainer(self.model)
        log_dir = trainer.train(input_data, output_data, epochs=epochs, batch_size=batch_size)
        logger.log(f"Training completed. Logs available at: {log_dir}")
        return log_dir

    def predict(self, input_data):
        """
        Runs inference using the ModelPredictor module.
        """
        logger.log(f"Running inference on {self.model_name}...")
        predictor = ModelPredictor(self.model)
        output, inference_time = predictor.predict(input_data)
        logger.log(f"Inference completed. Time taken: {inference_time:.2f} µs")
        return output, inference_time

    def benchmark(self, input_data, device="CPU"):
        """
        Benchmarks the model inference speed using the ModelBenchmark module.
        """
        logger.log(f"Benchmarking {self.model_name} on {device}...")
        benchmarker = ModelBenchmark(self.model_path)
        inference_time = benchmarker.benchmark(input_data, device=device)
        logger.log(f"Benchmarking completed. Time: {inference_time:.2f} µs")
        return inference_time

    def run_anomaly_detection(self, num_steps=100):
        """
        Executes real-time anomaly detection using RealTimeAnomalyMonitor.
        """
        logger.log(f"Starting real-time anomaly detection for {self.model_name}...")
        anomaly_monitor = RealTimeAnomalyMonitor(self.model)
        anomaly_monitor.detect_anomalies(num_steps=num_steps)
