import tensorflow as tf
from src.utils.logger import logger
from src.utils.tensorboard_logger import TensorBoardLogger

class ModelTrainer:
    def __init__(self, model, log_dir="logs"):
        self.model = model
        self.tensorboard_logger = TensorBoardLogger(log_dir)

    def train(self, input_data, output_data, epochs=10, batch_size=32):
        logger.log(f"Training model for {epochs} epochs... Logs saved in {self.tensorboard_logger.get_log_dir()}")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_logger.get_log_dir(), histogram_freq=1)

        self.model.fit(input_data, output_data, epochs=epochs, batch_size=batch_size,
                       validation_split=0.2, callbacks=[tensorboard_callback])

        return self.tensorboard_logger.get_log_dir()
