import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class AutoencoderModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.autoencoder, self.encoder = self.build_autoencoder()

    def build_autoencoder(self):
        encoder = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(self.input_dim,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu')
        ], name="encoder")

        decoder = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(8,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.input_dim, activation='sigmoid')
        ], name="decoder")

        input_layer = keras.layers.Input(shape=(self.input_dim,))
        encoded = encoder(input_layer)
        decoded = decoder(encoded)

        autoencoder = keras.Model(input_layer, decoded, name="autoencoder")
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder, encoder

    def train(self, normal_data, epochs=50, batch_size=32):
        self.autoencoder.fit(normal_data, normal_data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    def extract_features(self, data):
        return self.encoder.predict(data)

    def save_models(self, autoencoder_path, encoder_path):
        self.autoencoder.save(autoencoder_path)
        self.encoder.save(encoder_path)
        print(f"Autoencoder model saved at {autoencoder_path}")
        print(f"Encoder model saved at {encoder_path}")

class AnomalyDetector:
    def __init__(self):
        self.ocsvm = None
        self.scaler = None

    def train_ocsvm(self, features):
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        self.ocsvm = OneClassSVM(kernel='rbf', gamma='auto').fit(features)

    def save_model(self, ocsvm_path, scaler_path):
        joblib.dump(self.ocsvm, ocsvm_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"One-Class SVM model saved at {ocsvm_path}")
        print(f"Scaler saved at {scaler_path}")

class ModelTrainer:
    def __init__(self, normal_data_path):
        self.normal_data_path = normal_data_path
        self.normal_data = None

    def load_data(self):
        self.normal_data = np.load(self.normal_data_path)
        print(f"Loaded normal data with shape: {self.normal_data.shape}")

    def train(self, autoencoder_path, encoder_path, ocsvm_path, scaler_path):
        input_dim = self.normal_data.shape[1]

        # Train Autoencoder
        autoencoder_model = AutoencoderModel(input_dim)
        autoencoder_model.train(self.normal_data)

        # Save Autoencoder
        autoencoder_model.save_models(autoencoder_path, encoder_path)

        # Extract Features
        features = autoencoder_model.extract_features(self.normal_data)

        # Train One-Class SVM
        anomaly_detector = AnomalyDetector()
        anomaly_detector.train_ocsvm(features)

        # Save SVM model
        anomaly_detector.save_model(ocsvm_path, scaler_path)

if __name__ == '__main__':
    normal_data_file = "normal_data.npy"  # Path to preprocessed normal data

    trainer = ModelTrainer(normal_data_file)
    trainer.load_data()

    trainer.train(
        autoencoder_path="autoencoder_model.h5",
        encoder_path="encoder_model.h5",
        ocsvm_path="ocsvm_model.pkl",
        scaler_path="scaler.pkl"
    )
