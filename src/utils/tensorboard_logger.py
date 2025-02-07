import tensorflow as tf
import os
from datetime import datetime

class TensorBoardLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def log_scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def log_histogram(self, tag, values, step):
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)

    def get_log_dir(self):
        return self.log_dir
