class DataPreprocessor:
    """
    Cleans, removes NaN/duplicates, and transforms numeric columns.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_nan(self):
        self.df.dropna(inplace=True)
        return self

    def convert_float64_to_float32(self):
        float64_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float64_cols] = self.df[float64_cols].astype('float32')
        return self

    def get_dataframe(self):
        return self.df