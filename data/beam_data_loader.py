class BeamDataLoader:
    """
    Loads and merges BPM beam configuration data.
    """

    def __init__(self, config: BPMDataConfig):
        self.config = config

    def load_beam_config_df(self) -> pd.DataFrame:
        """Loads beam config CSV and updates columns."""
        beam_config_df = pd.read_csv(self.config.beam_settings_data_path)
        beam_config_df.drop("Unnamed: 0", axis=1, errors='ignore', inplace=True)
        beam_config_df['timestamps'] = pd.to_datetime(beam_config_df['timestamps'])
        beam_config_df = self.config.update_beam_config(beam_config_df)
        return beam_config_df