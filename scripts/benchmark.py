import argparse
import numpy as np
from src.utils.benchmarking import ModelBenchmark
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Benchmark Model Inference Speed")

    # Command-line arguments
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Autoencoder, VAE, Transformer)")
    parser.add_argument("--path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--device", type=str, choices=["CPU", "GPU"], default="CPU", help="Device to run benchmark")

    args = parser.parse_args()

    # Create synthetic input data
    input_dim = 100
    input_data = np.random.rand(1, input_dim).astype(np.float32)

    # Run benchmark
    logger.log(f"🚀 Starting Benchmark for {args.model} on {args.device}...")
    benchmarker = ModelBenchmark(args.path)
    inference_time = benchmarker.benchmark(input_data, device=args.device)
    logger.log(f"📊 Final Benchmark Result: {inference_time:.2f} µs")

if __name__ == "__main__":
    main()
