#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.fftpack import fft
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import argparse

# Jlab Packages
from data_utils import get_traces
from beam_settings_parser_hdf5 import BeamConfigParserHDF5
from beam_settings_prep import BeamConfigPreProcessor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TensorFlow
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional, RepeatVector, TimeDistributed, Lambda
)

# Local modules (assuming you have them in your Python path)
from models.vae_bilstm import MyVAE
from config.bpm_config import BPMDataConfig
from config.dcm_config import DCMDatConfig
from data.beam_data_loader import BeamDataLoader
from data.data_preprocessor import DataPreprocessor
from data.merge_datasets import MergeDatasets
from utils.logger import Logger

pd.options.display.max_columns = None
pd.options.display.max_rows = None


class SNSRawPrepSepDNNFactory:
    """
    The main factory that orchestrates the entire pipeline.
    Subcommands:
      1) train_pipeline(...)  - Load data, train VAE-BiLSTM, save model
      2) predict_pipeline(...) - Load model, run anomaly detection
    """

    def __init__(self):
        self.bpm_config = BPMDataConfig()
        self.dcm_config = DCMDatConfig()
        self.logger = Logger()
        self.window_size = 100
        self.num_features = 51

    def create_beam_data(self) -> pd.DataFrame:
        """Load beam config CSV and do minimal merges."""
        self.logger.info("====== Inside create_beam_data ======")
        loader = BeamDataLoader(self.bpm_config)
        beam_df = loader.load_beam_config_df()
        return beam_df

    def create_dcm_data(self) -> pd.DataFrame:
        self.logger.info("====== Inside create_dcm_data ======")
        normal_files, anomaly_files = self.dcm_config.get_sep_filtered_files()
        merger = MergeDatasets(length_of_waveform=self.dcm_config.length_of_waveform)
        dcm_normal = merger.process_files(normal_files, 'Sep24', 0, data_type=0)
        dcm_anormal = merger.process_files(anomaly_files, 'Sep24', 1, data_type=-1, alarm=48)
        return pd.concat([dcm_normal, dcm_anormal], ignore_index=True)

    def preprocess_merged_data(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("====== Inside preprocess_merged_data ======")
        dp = DataPreprocessor(merged_df)
        dp.remove_nan().convert_float64_to_float32()
        return dp.get_dataframe()

    def create_vae_bilstm_model(self, latent_dim: int = 16) -> MyVAE:
        """Factory method to build the VAE-BiLSTM model."""
        self.logger.info("====== Inside create_vae_bilstm_model ======")
        model = MyVAE(
            window_size=self.window_size,
            num_features=self.num_features,
            latent_dim=latent_dim
        )
        return model

    def extract_trace_features(self, trace_row: np.ndarray) -> np.ndarray:
        """Downsample + basic stats + partial FFT, etc."""
        downsampled = trace_row[::20]  # from 10k to 500
        mean_val = np.mean(downsampled)
        std_val = np.std(downsampled)
        peak_val = np.max(downsampled)
        fft_val = np.abs(fft(downsampled)[1])
        # example: keep first 50 + stats
        return np.hstack([downsampled[:50], mean_val, std_val, peak_val, fft_val])

    def _prepare_final_df(self) -> (pd.DataFrame, list):
        """
        A single method that:
          1) Loads BPM + DCM data
          2) Merges them
          3) Preprocesses (NaN, type conversions)
          4) Extracts trace features + PCA
        Returns:
          df_final: final DataFrame with PCA columns
          trace_feature_names: list of PCA column names
        """
        self.logger.info("====== Inside _prepare_final_df ======")

        # 1) Load data
        beam_df = self.create_beam_data()
        dcm_df = self.create_dcm_data()

        # 2) Merge
        merged_df = pd.merge_asof(
            dcm_df.sort_values("timestamps"),
            beam_df.sort_values("timestamps"),
            on="timestamps",
            direction="nearest"
        )

        # 3) Preprocess
        cleaned_df = self.preprocess_merged_data(merged_df)
        cleaned_df["traces"] = merged_df["traces"]
        cleaned_df["timestamps"] = merged_df["timestamps"]
        for col_to_drop in ["file", "anomoly_flag"]:
            if col_to_drop in cleaned_df.columns:
                cleaned_df.drop(columns=[col_to_drop], inplace=True, errors='ignore')

        cleaned_df["timestamp_seconds"] = pd.to_datetime(cleaned_df["timestamps"], errors="coerce").astype(int) / 10**9
        cleaned_df["time_diff"] = cleaned_df["timestamp_seconds"].diff().fillna(0)
        cleaned_df["traces"] = cleaned_df["traces"].apply(
            lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
        )

        # 4) Feature extraction + PCA
        trace_features = np.array(
            cleaned_df["traces"].apply(self.extract_trace_features).tolist()
        )
        pca = PCA(n_components=50)
        trace_features_pca = pca.fit_transform(trace_features)
        trace_feature_names = [f"PCA_Trace_{i}" for i in range(trace_features_pca.shape[1])]

        df_pca = pd.DataFrame(trace_features_pca, columns=trace_feature_names)
        df_final = pd.concat([cleaned_df.drop(columns=["traces"], errors="ignore"), df_pca], axis=1)

        return df_final, trace_feature_names

    # ---------------------------
    # TRAIN PIPELINE
    # ---------------------------
    def train_pipeline(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        latent_dim: int = 16,
        model_path: str = "saved_models/vae_bilstm_model.h5"
    ):
        """
        1) Prepares final DataFrame (df_final)
        2) Builds & trains VAE-BiLSTM
        3) Saves model weights
        """
        self.logger.info("====== Inside train_pipeline ======")

        df_final, trace_feature_names = self._prepare_final_df()

        # Build model
        vae_model = self.create_vae_bilstm_model(latent_dim=latent_dim)

        # Compile
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        vae_model.compile(optimizer=optimizer, loss='mae')

        # Windowing
        X_train_combined = []
        for i in range(self.window_size, len(df_final)):
            past_pulses = df_final.iloc[i - self.window_size : i][trace_feature_names + ["time_diff"]]
            X_train_combined.append(past_pulses.values)
        X_train_combined = np.array(X_train_combined, dtype=np.float32)
        X_train_combined = np.nan_to_num(X_train_combined)

        # Train
        history = vae_model.fit(
            X_train_combined, X_train_combined,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1
        )

        # Save
        vae_model.save_weights("saved_models/vae_bilstm_model.weights.h5")
        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info("====== Training pipeline completed ======")

    # ---------------------------
    # PREDICT PIPELINE
    # ---------------------------
    def predict_pipeline(
        self,
        model_path: str = "vae_bilstm_model.h5",
        threshold_percentile: float = 95.0
    ):
        """
        1) Prepares final DataFrame (df_final)
        2) Loads VAE-BiLSTM weights
        3) Computes reconstruction errors & anomalies
        """
        self.logger.info("====== Inside predict_pipeline ======")

        df_final, trace_feature_names = self._prepare_final_df()

        # Build same model architecture
        new_vae_model = MyVAE(window_size=100, num_features=51, latent_dim=16)
        new_vae_model.build((None, 100, 51))  # or a dummy forward pass
        weights_path = os.path.expanduser(model_path)
        new_vae_model.load_weights(weights_path)
        self.logger.info(f"Model weights loaded from: {model_path}")

        # Windowing
        X_test_combined = []
        for i in range(self.window_size, len(df_final)):
            past_pulses = df_final.iloc[i - self.window_size : i][trace_feature_names + ["time_diff"]]
            X_test_combined.append(past_pulses.values)
        X_test_combined = np.array(X_test_combined, dtype=np.float32)
        X_test_combined = np.nan_to_num(X_test_combined)

        # Predict
        X_pred = loaded_vae_model.predict(X_test_combined)
        reconstruction_errors = np.mean(np.abs(X_test_combined - X_pred), axis=(1, 2))
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        anomalies = reconstruction_errors > threshold

        df_anomalies = pd.DataFrame({
            "Timestamp": df_final["timestamps"].iloc[self.window_size:],
            "Reconstruction_Error": reconstruction_errors,
            "Anomaly": anomalies
        })

        self.logger.info(f"Top 20 anomalies (threshold={threshold:.4f} at {threshold_percentile} percentile):")
        self.logger.info(df_anomalies.sort_values(by="Reconstruction_Error", ascending=False).head(20))

        self.logger.info("====== Prediction pipeline completed ======")

