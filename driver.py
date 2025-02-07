import argparse
import numpy as np
from src.factory.model_loader import ModelLoader
from src.factory.model_trainer import ModelTrainer
from src.factory.model_predictor import ModelPredictor
from src.anomaly_detection.realtime_monitor import RealTimeAnomalyMonitor
from src.utils.benchmarking import ModelBenchmark
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Model Training, Inference, and Benchmarking Driver")

    # Command-line arguments
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Autoencoder, VAE, Siamese, Transformer)")
    parser.add_argument("--path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "anomaly", "benchmark"], required=True, help="Operation mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (for training mode)")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size (for training mode)")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps for real-time anomaly detection")
    parser.add_argument("--device", type=str, choices=["CPU", "GPU"], default="CPU", help="Device for benchmarking")

    args = parser.parse_args()

    # Load model
    logger.log(f"Loading model: {args.model} from {args.path}")
    model_loader = ModelLoader(args.path)
    model = model_loader.model

    # Create synthetic input data
    input_dim = 100
    input_data = np.random.rand(1, input_dim).astype(np.float32)

    # Execute based on mode
    if args.mode == "train":
        logger.log(f"Training {args.model} for {args.epochs} epochs...")
        output_data = np.random.rand(1, input_dim).astype(np.float32)
        trainer = ModelTrainer(model)
        log_dir = trainer.train(input_data, output_data, epochs=args.epochs, batch_size=args.batch_size)
        logger.log(f"Training completed. Logs available at: {log_dir}")

    elif args.mode == "predict":
        logger.log(f"Running inference on {args.model}...")
        predictor = ModelPredictor(model)
        output, inference_time = predictor.predict(input_data)
        logger.log(f"Inference completed. Time taken: {inference_time:.2f} µs")

    elif args.mode == "anomaly":
        logger.log(f"Starting real-time anomaly detection for {args.model}...")
        anomaly_monitor = RealTimeAnomalyMonitor(model)
        anomaly_monitor.detect_anomalies(num_steps=args.steps)

    elif args.mode == "benchmark":
        logger.log(f"Benchmarking {args.model} on {args.device}...")
        benchmarker = ModelBenchmark(args.path)
        inference_time = benchmarker.benchmark(input_data, device=args.device)
        logger.log(f"Benchmarking completed. Time: {inference_time:.2f} µs")

if __name__ == "__main__":
    main()
