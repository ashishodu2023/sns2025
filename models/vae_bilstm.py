import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional, RepeatVector, TimeDistributed, Lambda
)

from utils.logger import Logger

class MyVAE(Model):
    """
    Subclassed VAE with BiLSTM encoder/decoder + KL divergence.
    """

    def __init__(self, window_size, num_features, latent_dim=16):
        super(MyVAE, self).__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.logger = Logger()

        # Encoder
        self.encoder_bilstm_1 = Bidirectional(LSTM(64, return_sequences=True))
        self.encoder_bilstm_2 = Bidirectional(LSTM(32, return_sequences=False))
        self.z_mean_dense = Dense(latent_dim, name="z_mean")
        self.z_log_var_dense = Dense(latent_dim, name="z_log_var")

        # Decoder
        self.repeat_vector = RepeatVector(window_size)
        self.decoder_bilstm_1 = Bidirectional(LSTM(32, return_sequences=True))
        self.decoder_bilstm_2 = Bidirectional(LSTM(64, return_sequences=True))
        self.output_dense = TimeDistributed(Dense(num_features))

    def encode(self, x):
        self.logger.info("====== Inside encode ======")
        x = self.encoder_bilstm_1(x)
        x = self.encoder_bilstm_2(x)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        self.logger.info("====== Inside reparameterize ======")
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        self.logger.info("====== Inside decode ======")
        x = self.repeat_vector(z)
        x = self.decoder_bilstm_1(x)
        x = self.decoder_bilstm_2(x)
        return self.output_dense(x)

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        # KL Divergence
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1
        )
        self.add_loss(kl_loss)
        return x_recon