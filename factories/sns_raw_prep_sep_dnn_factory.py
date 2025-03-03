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


#Jlab Packages
from data_utils import get_traces
from beam_settings_parser_hdf5 import BeamConfigParserHDF5
from beam_settings_prep import BeamConfigPreProcessor


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Tensorflow 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional, RepeatVector, TimeDistributed, Lambda
)
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
    The main factory that orchestrates the entire pipeline:
    1) Load BPM config + DCM config
    2) Merge data
    3) Preprocess
    4) Build VAE-BiLSTM
    5) Train and do anomaly detection
    """

    def __init__(self):
        # Initialize config classes
        self.bpm_config = BPMDataConfig()
        self.dcm_config = DCMDatConfig()
        self.logger = Logger()

    def create_beam_data(self) -> pd.DataFrame:
        """Load beam config CSV and do minimal merges."""
        self.logger.info("====== Inside create_beam_data ======")
        loader = BeamDataLoader(self.bpm_config)
        beam_df = loader.load_beam_config_df()
        return beam_df

    def create_dcm_data(self) -> pd.DataFrame:
        self.logger.info("====== Inside create_dcm_data ======")
        # 1) get normal + anomaly file lists
        normal_files, anomaly_files = self.dcm_config.get_sep_filtered_files()

        # 2) process them
        merger = MergeDatasets(length_of_waveform=self.dcm_config.length_of_waveform)
        dcm_normal = merger.process_files(normal_files, 'Sep24', 0, data_type=0)
        dcm_anormal = merger.process_files(anomaly_files, 'Sep24', 1, data_type=-1, alarm=48)
        return pd.concat([dcm_normal, dcm_anormal], ignore_index=True)

    def preprocess_merged_data(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        data cleaning, removing NaNs, etc.
        """
        self.logger.info("====== Inside preprocess_merged_data ======")
        # Example usage:
        dp = DataPreprocessor(merged_df)
        dp.remove_nan().convert_float64_to_float32()
        return dp.get_dataframe()

    def create_vae_bilstm_model(self, window_size: int, num_features: int, latent_dim: int = 16) -> MyVAE:
        self.logger.info("====== Inside create_vae_bilstm_model ======")
        """Factory method to build the VAE-BiLSTM model."""
        model = MyVAE(window_size=window_size, num_features=num_features, latent_dim=latent_dim)
        return model

    def extract_trace_features(self, trace_row: np.ndarray) -> np.ndarray:
        self.logger.info("====== Inside extract_trace_features ======")
        """Downsample + basic stats + partial FFT, etc."""
        downsampled = trace_row[::20]  # from 10k to 500
        mean_val = np.mean(downsampled)
        std_val = np.std(downsampled)
        peak_val = np.max(downsampled)
        fft_val = np.abs(fft(downsampled)[1])
        # example: keep first 50 + stats
        return np.hstack([downsampled[:50], mean_val, std_val, peak_val, fft_val])

    def sampling(args):
        self.logger.info("====== Inside sampling ======")
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    def run_pipeline(self):
        """
        1) Loads beam + dcm data
        2) Merges & preprocesses
        3) Extracts trace features & PCA
        4) Builds + trains VAE-BiLSTM
        5) Performs anomaly detection
        """
        self.logger.info("====== Inside run_pipeline======")

        self.logger.info("====== Load BPM Data ======")
        beam_df = self.create_beam_data()
        self.logger.info(f"====== The total row count in BPM : {beam_df.shape[0]}")

        self.logger.info("====== Load & Merge DCM Data ======")
        dcm_df = self.create_dcm_data()
        #self.logger.info(dcm_df.head())
        self.logger.info(f"====== The total row count in DCM : {dcm_df.shape[0]}")

        self.logger.info("====== Merge with BPM on timestamps (placeholder)")
        merged_df = pd.DataFrame()  
        merged_df = pd.merge_asof(dcm_df.sort_values("timestamps"), beam_df.sort_values("timestamps"), on="timestamps", direction="nearest")
        self.logger.info(f"======= The total row count in merged dataframe : {merged_df.shape[0]}")

        self.logger.info("====== Preprocess Merged Data ======")
        cleaned_df = self.preprocess_merged_data(merged_df)
        cleaned_df['traces']=merged_df['traces']
        cleaned_df['timestamps']=merged_df['timestamps']
        cleaned_df.drop(columns=['file','anomoly_flag'],inplace=True,axis=1)
        cleaned_df["timestamp_seconds"] = pd.to_datetime(cleaned_df["timestamps"], errors="coerce").astype(int) / 10**9
        cleaned_df["time_diff"] = cleaned_df["timestamp_seconds"].diff().fillna(0)
        cleaned_df["traces"] = cleaned_df["traces"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
        self.logger.info(f"====== The total row count in cleaned dataframe : {cleaned_df.shape[0]}")

        self.logger.info("====== Feature Extraction for Traces ======")
        trace_features = np.array(cleaned_df["traces"].apply(self.extract_trace_features).tolist())
        pca = PCA(n_components=50)
        trace_features_pca = pca.fit_transform(trace_features)
        trace_feature_names = [f"PCA_Trace_{i}" for i in range(trace_features_pca.shape[1])]
        df_pca = pd.DataFrame(trace_features_pca, columns=trace_feature_names)
        df = pd.concat([cleaned_df.drop(columns=["traces"], errors="ignore"), df_pca], axis=1)
        #self.logger.info(df.head())

        self.logger.info("====== Build VAE-BiLSTM Model ======")
        window_size, num_features = 100, 51
        vae_model = self.create_vae_bilstm_model(window_size, num_features, latent_dim=16)

        self.logger.info("======  Train Model ======")
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
        vae_model.compile(optimizer='sgd', loss='mae')
        X_train_combined = []
        for i in range(window_size, len(df)):
            past_pulses = df.iloc[i - window_size:i][trace_feature_names + ["time_diff"]]
            X_train_combined.append(past_pulses.values)
        X_train_combined = np.array(X_train_combined, dtype=np.float32)
        X_train_combined = np.nan_to_num(X_train_combined)
        history = vae_model.fit(X_train_combined, X_train_combined, epochs=50, batch_size=16, validation_split=0.1)

        self.logger.info("====== Anomaly Detection ======")
        X_pred_combined = vae_model.predict(X_train_combined)
        reconstruction_errors = np.mean(np.abs(X_train_combined - X_pred_combined), axis=(1, 2))  # Mean absolute error
        threshold = np.percentile(reconstruction_errors, 95)
        anomalies = reconstruction_errors > threshold
        df_anomalies_combined = pd.DataFrame({
            "Timestamp": df["timestamps"].iloc[window_size:],  # Align with pulse times
            "Reconstruction_Error": reconstruction_errors,
            "Anomaly": anomalies})
        self.logger.info("====== Top 20 Anomalous Pulses ======")
        self.logger.info(df_anomalies_combined.sort_values(by="Reconstruction_Error", ascending=False).head(20))   

        self.logger.info("====== Pipeline run completed ======")