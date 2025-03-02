import pandas as pd
from utils.logger import Logger

class DataPreprocessor:
    """
    Cleans, removes NaN/duplicates, and transforms numeric columns.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = Logger()

    def remove_nan(self):
        self.logger.info("====== Inside remove_nan ======")
        self.df.dropna(inplace=True)
        return self

    def convert_float64_to_float32(self):
        self.logger.info("====== Inside convert_float64_to_float32 ======")
        float64_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float64_cols] = self.df[float64_cols].astype('float32')
        return self

    def get_dataframe(self):
        self.logger.info("====== Inside get_dataframe ======")
        return self.df