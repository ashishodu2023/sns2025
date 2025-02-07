import time
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tensorflow.lite as tflite
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from src.utils.logger import logger

class ModelBenchmark:
    """
    Benchmarks model inference speed on CPU/GPU for different model types.
    Supports:
    - TensorFlow (.h5)
    - ONNX (.onnx)
    - TensorFlow Lite (.tflite)
    - TensorRT (.trt)
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the appropriate model based on the file extension.
        """
        if self.model_path.endswith(".h5"):
            logger.log(f"🔄 Loading TensorFlow Model for Benchmarking: {self.model_path}")
            return tf.keras.models.load_model(self.model_path)
        elif self.model_path.endswith(".onnx"):
            logger.log(f"🔄 Loading ONNX Model for Benchmarking: {self.model_path}")
            return ort.InferenceSession(self.model_path)
        elif self.model_path.endswith(".tflite"):
            logger.log(f"🔄 Loading TensorFlow Lite Model for Benchmarking: {self.model_path}")
            interpreter = tflite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        elif self.model_path.endswith(".trt"):
            logger.log(f"🔄 Loading TensorRT Model for Benchmarking: {self.model_path}")
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(self.model_path, "rb") as f:
                engine_data = f.read()
            return runtime.deserialize_cuda_engine(engine_data)
        else:
            raise ValueError("❌ Unsupported model format!")

    def benchmark(self, input_data, device="CPU"):
        """
        Benchmarks the model inference time.
        """
        logger.log(f"⚡ Running Benchmark on {device} for {self.model_path}")

        start_time = time.time()
        
        if isinstance(self.model, tf.keras.Model):
            with tf.device(f"/{device}:0"):
                self.model.predict(input_data)
        elif isinstance(self.model, ort.InferenceSession):
            self.model.run(None, {"input": input_data})
        elif isinstance(self.model, tflite.Interpreter):
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            self.model.set_tensor(input_details[0]['index'], input_data)
            self.model.invoke()
        else:
            raise ValueError("❌ Invalid model type!")

        end_time = time.time()
        inference_time = (end_time - start_time) * 1e6  # Convert to microseconds (µs)
        logger.log(f"✅ Benchmark Completed: {self.model_path} on {device}: {inference_time:.2f} µs")
        return inference_time
