from data_utils import get_traces
from beam_settings_parser_hdf5 import BeamConfigParserHDF5
from beam_settings_prep import BeamConfigPreProcessor
import pandas as pd
import numpy as np 
import os
from utils.logger import Logger

class DCMDatConfig:
    """
    Holds configuration for DCM dataset,
    e.g., dataset paths, anomaly codes, etc.
    """
    def __init__(self):
        self.dataset1_loc = "/work/data_science/suf_sns/DCM_Errant/"
        self.dataset2_loc = "/w/data_science-sciwork24/suf_sns/DCML_dataset_Sept2024"
        self.start_date = 20220218
        self.end_date = 20220318
        self.anomaly_type = "00110000"
        self.length_of_waveform = 10000
        self.exclude_dates = [20220220, 20220221, 20220222, 20220223, 20220301, 20220308, 20220309, 20220315]
        self.filtered_normal_files2 = []
        self.filtered_anomaly_files2 = []
        self.traces = []
        self.timestamps = []
        self.logger = Logger()

    def get_sep_filtered_files(self):
        """Identify normal and anomaly files from dataset2_loc."""
        self.logger.info("====== Inside the get_sep_filtered_files ======")
        subfolders = [f.path for f in os.scandir(self.dataset2_loc) if f.is_dir()]
        for directory in subfolders:
            if "normal" in directory or "anomal" in directory:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if ".gz" in file:
                            if 'normal' in directory:
                                self.filtered_normal_files2.append(os.path.join(root, file))
                            elif "anomal" in directory:
                                self.filtered_anomaly_files2.append(os.path.join(root, file))

        self.logger.info(len(self.filtered_normal_files2))
        self.logger.info(len(self.filtered_anomaly_files2))
        return self.filtered_normal_files2, self.filtered_anomaly_files2

    def get_sep_filtered_files(self):
        self.logger.info("====== Inside the get_sep_filtered_files ======")
        subfolders = [f.path for f in os.scandir(self.dataset2_loc) if f.is_dir()]
        for directory in subfolders:
            if "normal" in directory or "anomal" in directory:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.startswith("."):
                            continue  # Skip hidden/cache files
                        if not file.endswith(".gz"):
                            full_path = os.path.join(root, file)
                            if 'normal' in directory:
                                self.filtered_normal_files2.append(full_path)
                            elif 'anomal' in directory:
                                self.filtered_anomaly_files2.append(full_path)

        self.logger.info(f"Normal files found: {len(self.filtered_normal_files2)}")
        self.logger.info(f"Anomaly files found: {len(self.filtered_anomaly_files2)}")
        return self.filtered_normal_files2, self.filtered_anomaly_files2
