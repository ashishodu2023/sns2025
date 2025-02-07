import tensorflow as tf
import onnxruntime as ort
import tensorflow.lite as tflite
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from src.utils.logger import logger

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        if self.model_path.endswith(".h5"):
            logger.log(f"Loading TensorFlow Model: {self.model_path}")
            return tf.keras.models.load_model(self.model_path)
        elif self.model_path.endswith(".onnx"):
            logger.log(f"Loading ONNX Model: {self.model_path}")
            return ort.InferenceSession(self.model_path)
        elif self.model_path.endswith(".tflite"):
            logger.log(f"Loading TensorFlow Lite Model: {self.model_path}")
            interpreter = tflite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        elif self.model_path.endswith(".trt"):
            logger.log(f"Loading TensorRT Model: {self.model_path}")
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(self.model_path, "rb") as f:
                engine_data = f.read()
            return runtime.deserialize_cuda_engine(engine_data)
        else:
            raise ValueError("Unsupported model format!")
