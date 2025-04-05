#!/usr/bin/env python
# coding: utf-8

"""
SNSRawPrepSepDNNFactory: Factory-based framework for SNS data processing, VAE-BiLSTM training, and anomaly detection.
This updated version uses a sliding window of size 3 so that it uses the (n-2)th and (n-1)th pulses
to forecast (and reconstruct) the nth pulse. During prediction, reconstruction error is computed only on the nth pulse.
Subcommands:
  train   - Train the model and save weights (with TensorBoard logs).
  predict - Load the model and perform anomaly detection, saving plots.
  
Usage:
  python sns_raw_prep_sep_dnn_factory.py train --epochs 50 --batch_size 16 --learning_rate 1e-5 --latent_dim 16 --model_path saved_models/vae_bilstm_model.weights.h5 --tensorboard_logdir logs/fit
  python sns_raw_prep_sep_dnn_factory.py predict --model_path saved_models/vae_bilstm_model.weights.h5 --threshold_percentile 95.0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.fftpack import fft
from datetime import datetime, timedelta
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

# Local modules (ensure these are in your PYTHONPATH)
from models.vae_bilstm import MyVAE
from config.bpm_config import BPMDataConfig
from config.dcm_config import DCMDatConfig
from data.beam_data_loader import BeamDataLoader
from data.data_preprocessor import DataPreprocessor
from data.merge_datasets import MergeDatasets
from utils.logger import Logger
from visualization.plots import plot_and_save_anomalies

pd.options.display.max_columns = None
pd.options.display.max_rows = None


class SNSRawPrepSepDNNFactory:
    """
    The main factory that orchestrates the entire pipeline.
    Subcommands:
      1) train_pipeline(...)  - Loads data, trains VAE-BiLSTM (window size = 3) with TensorBoard, saves model weights.
      2) predict_pipeline(...) - Loads model, predicts anomalies using the reconstruction error on the nth pulse.
    """

    def __init__(self):
        self.bpm_config = BPMDataConfig()
        self.dcm_config = DCMDatConfig()
        self.logger = Logger()
        # For this forecasting setup, our window size is 3:
        self.window_size = 3  
        # Our model's output will have num_features equal to (number of PCA components + 1 for "time_diff").
        # For consistency, we assume PCA produces 50 features; then num_features = 50 + 1 = 51.
        self.num_features = 51

    def create_beam_data(self) -> pd.DataFrame:
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
        self.logger.info("====== Inside create_vae_bilstm_model ======")
        model = MyVAE(
            window_size=self.window_size,
            num_features=self.num_features,
            latent_dim=latent_dim
        )
        return model

    def extract_trace_features(self, trace_row: np.ndarray) -> np.ndarray:
        """Downsample + basic stats + partial FFT from 10k points."""
        downsampled = trace_row[::20]  # from 10k to 500 points
        mean_val = np.mean(downsampled)
        std_val = np.std(downsampled)
        peak_val = np.max(downsampled)
        fft_val = np.abs(fft(downsampled)[1])
        # Keep first 50 points plus the computed statistics:
        return np.hstack([downsampled[:50], mean_val, std_val, peak_val, fft_val])

    def _prepare_final_df(self) -> (pd.DataFrame, list):
        """
        Loads, merges, preprocesses, and extracts PCA features.
        Returns:
          df_final: final DataFrame with PCA columns.
          trace_feature_names: list of PCA column names.
        """
        self.logger.info("====== Inside _prepare_final_df ======")
        beam_df = self.create_beam_data()
        dcm_df = self.create_dcm_data()

        merged_df = pd.merge_asof(
            dcm_df.sort_values("timestamps"),
            beam_df.sort_values("timestamps"),
            on="timestamps",
            direction="nearest"
        )

        cleaned_df = self.preprocess_merged_data(merged_df)
        cleaned_df["traces"] = merged_df["traces"]
        cleaned_df["timestamps"] = merged_df["timestamps"]

        # Drop unnecessary columns if present.
        for col_to_drop in ["file", "anomoly_flag"]:
            if col_to_drop in cleaned_df.columns:
                cleaned_df.drop(columns=[col_to_drop], inplace=True, errors="ignore")

        cleaned_df["timestamp_seconds"] = pd.to_datetime(cleaned_df["timestamps"], errors="coerce").astype(int) / 10**9
        cleaned_df["time_diff"] = cleaned_df["timestamp_seconds"].diff().fillna(0)
        cleaned_df["traces"] = cleaned_df["traces"].apply(
            lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
        )

        # Extract features from traces and reduce dimensions via PCA.
        trace_features = np.array(cleaned_df["traces"].apply(self.extract_trace_features).tolist())
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
        model_path: str = "saved_models/vae_bilstm_model.weights.h5",
        tensorboard_logdir: str = "logs/fit"
    ):
        """
        Prepares final data, trains VAE-BiLSTM on windows of 3 pulses (n-2 & n-1 to reconstruct nth pulse),
        logs training to TensorBoard, and saves model weights.
        """
        self.logger.info("====== Inside train_pipeline ======")

        # Prepare data
        df_final, trace_feature_names = self._prepare_final_df()

        # Build model
        vae_model = self.create_vae_bilstm_model(latent_dim=latent_dim)
        vae_model.build((None, self.window_size, self.num_features))
        self.logger.info(vae_model.summary())

        # Compile
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        vae_model.compile(optimizer=optimizer, loss='mae')

        # Create rolling windows.
        # With window_size = 3, each window consists of 3 pulses.
        # For forecasting anomaly, we will later compute reconstruction error only on the last (nth) pulse.
        X_train_combined = []
        for i in range(self.window_size, len(df_final)):
            window = df_final.iloc[i - self.window_size : i][trace_feature_names + ["time_diff"]]
            X_train_combined.append(window.values)
        X_train_combined = np.array(X_train_combined, dtype=np.float32)
        X_train_combined = np.nan_to_num(X_train_combined)

        # Setup TensorBoard callback
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(tensorboard_logdir, time_str)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        self.logger.info(f"TensorBoard logs will be saved to: {log_dir}")

        # Train the model (autoencoder reconstructs full window)
        history = vae_model.fit(
            X_train_combined, X_train_combined,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[tensorboard_callback]
        )

        # Save weights
        vae_model.save_weights(model_path)
        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info("====== Training pipeline completed ======")

    # ---------------------------
    # PREDICT PIPELINE
    # ---------------------------
    def predict_pipeline(
        self,
        model_path: str = "saved_models/vae_bilstm_model.weights.h5",
        threshold_percentile: float = 95.0
    ):
        """
        Prepares final data, loads the trained model, and performs anomaly detection.
        For each window (of 3 pulses), it computes the reconstruction error only on the last pulse.
        """
        self.logger.info("====== Inside predict_pipeline ======")

        df_final, trace_feature_names = self._prepare_final_df()

        # Build the model architecture and perform a dummy forward pass to initialize weights.
        new_vae_model = MyVAE(window_size=self.window_size, num_features=self.num_features, latent_dim=16)
        dummy_input = tf.zeros((1, self.window_size, self.num_features), dtype=tf.float32)
        _ = new_vae_model(dummy_input)

        weights_path = os.path.expanduser(model_path)
        new_vae_model.load_weights(weights_path)
        self.logger.info(f"Model weights loaded from: {model_path}")

        # Create test windows
        X_test_combined = []
        for i in range(self.window_size, len(df_final)):
            window = df_final.iloc[i - self.window_size : i][trace_feature_names + ["time_diff"]]
            X_test_combined.append(window.values)
        X_test_combined = np.array(X_test_combined, dtype=np.float32)
        X_test_combined = np.nan_to_num(X_test_combined)

        # Predict: new_vae_model predicts an entire window (3 pulses)
        X_pred = new_vae_model.predict(X_test_combined)
        # Compute reconstruction error only for the nth pulse (i.e. last timestep)
        reconstruction_errors = np.mean(np.abs(X_test_combined[:,-1,:] - X_pred[:,-1,:]), axis=1)
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

        self.logger.info("====== Plotting and saving reconstruction error plots ======")
        plot_and_save_anomalies(
            df_anomalies, 
            threshold=threshold, 
            dist_filename="dist_plot.png", 
            time_filename="time_plot.png"
        )
        self.logger.info("====== Saved Plots (PNG) ======")



