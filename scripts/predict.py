import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

class AnomalyPredictor:
    def __init__(self, autoencoder_path, encoder_path, ocsvm_path, scaler_path):
        # Load trained models
        self.autoencoder = keras.models.load_model(autoencoder_path)
        self.encoder = keras.models.load_model(encoder_path)
        self.ocsvm = joblib.load(ocsvm_path)
        self.scaler = joblib.load(scaler_path)

    def preprocess_data(self, data):
        """Ensure the input data is correctly formatted."""
        return np.array(data)

    def extract_features(self, data):
        """Extracts compressed features from the encoder."""
        return self.encoder.predict(data)

    def predict_anomalies(self, data):
        """Predict if the input data is normal or an anomaly."""
        # Preprocess and extract features
        data = self.preprocess_data(data)
        features = self.extract_features(data)

        # Scale features using the trained scaler
        features = self.scaler.transform(features)

        # Predict using One-Class SVM
        predictions = self.ocsvm.predict(features)

        # Convert predictions to human-readable format
        results = ["Normal" if p == 1 else "Anomaly" for p in predictions]
        return results

    def evaluate(self, data, true_labels):
        """Evaluate model performance on labeled test data."""
        predictions = self.predict_anomalies(data)
        binary_predictions = [1 if p == "Normal" else -1 for p in predictions]
        binary_labels = [1 if l == "Normal" else -1 for l in true_labels]

        from sklearn.metrics import classification_report
        print(classification_report(binary_labels, binary_predictions, target_names=["Anomaly", "Normal"]))

# Example Usage
if __name__ == '__main__':
    predictor = AnomalyPredictor(
        autoencoder_path="autoencoder_model.h5",
        encoder_path="encoder_model.h5",
        ocsvm_path="ocsvm_model.pkl",
        scaler_path="scaler.pkl"
    )

    # Load test data (Replace with actual test dataset)
    test_data = np.load("test_data.npy")  # Assuming test data is stored in NumPy format
    predictions = predictor.predict_anomalies(test_data)

    print("Predictions:", predictions)

    # Evaluate if true labels are available (Replace with actual labels)
    true_labels = ["Normal"] * 90 + ["Anomaly"] * 10  # Example labels for evaluation
    predictor.evaluate(test_data, true_labels)
