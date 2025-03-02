import pandas as pd 
import numpy as np 

class MergeDatasets:
    """
    Merges normal/anomaly traces into a single DataFrame, merges with BPM data.
    """

    def __init__(self, length_of_waveform=10000):
        self.length_of_waveform = length_of_waveform

    def process_files(self, file_list, label, flag_value, data_type, alarm=None):
        """
        Loops through file_list, extracts traces/timestamps, and appends flags.
        This is an example using get_traces(...) placeholders.
        """
        from data_utils import get_traces  # example
        traces, timestamps, flags, files = [], [], [], []

        for dcml in file_list[:50]:
            if alarm is not None:
                tmp_trace, tmp_timestamp = get_traces(
                    dcml, var_id="Trace2", begin=3000,
                    shift=self.length_of_waveform, data_type=data_type, alarm=alarm
                )
            else:
                tmp_trace, tmp_timestamp = get_traces(
                    dcml, var_id="Trace1", begin=3000,
                    shift=self.length_of_waveform, data_type=data_type
                )

            if not tmp_trace or not tmp_timestamp:
                print(f"Skipping {dcml} due to empty trace/timestamp")
                continue

            tmp_trace = np.array(tmp_trace)
            tmp_timestamp = np.array(tmp_timestamp)
            if len(tmp_trace) > 1:
                tmp_trace = tmp_trace[1:]
            if len(tmp_timestamp) > 1:
                tmp_timestamp = tmp_timestamp[1:]

            traces.extend(tmp_trace.tolist())
            timestamps.extend(tmp_timestamp.tolist())
            flags.extend([flag_value] * len(tmp_trace))
            files.extend([label] * len(tmp_trace))

        return pd.DataFrame({
            'anomoly_flag': flags,
            'file': files,
            'timestamps': timestamps,
            'traces': traces
        })