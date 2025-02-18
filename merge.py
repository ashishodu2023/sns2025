### Direct Nearest Timestamp Matching ###
import pandas as pd
import struct
import os


csv_df = pd.read_csv("bpm.csv")
csv_df["timestamp"] = pd.to_datetime(csv_df["timestamp"])

def process_bin_file(file_path):
    """ Read binary file and return DataFrame """
    data = []
    with open(file_path, "rb") as f:
        while chunk := f.read(8):  
            timestamp, trace = struct.unpack("ff", chunk)
            data.append((timestamp, trace))
    
    df = pd.DataFrame(data, columns=["timestamp", "trace"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  
    return df

merged_data = []
bin_files = os.listdir("path_to_bin_files")

for file in bin_files:
    bin_df = process_bin_file(f"path_to_bin_files/{file}")
    merged_df = pd.merge_asof(bin_df.sort_values("timestamp"), csv_df.sort_values("timestamp"), on="timestamp", direction="nearest")
    merged_data.append(merged_df)

final_df = pd.concat(merged_data, ignore_index=True)
final_df.to_csv("merged_dataset.csv", index=False)


### Interpolating CSV Data for More Granularity ###
csv_df = csv_df.set_index("timestamp").resample("1S").interpolate().reset_index()
merged_df = pd.merge_asof(bin_df.sort_values("timestamp"), csv_df.sort_values("timestamp"), on="timestamp", direction="nearest")

### Time Window Aggregation (Bucketing) ###


bin_df["timestamp"] = pd.to_datetime(bin_df["timestamp"])
bin_df = bin_df.set_index("timestamp").resample("1Min").agg({"trace": ["mean", "max", "min", "std"]}).reset_index()
merged_df = pd.merge_asof(bin_df, csv_df, on="timestamp", direction="nearest")


### Machine Learning-Based Timestamp Matching ###
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


X = csv_df["trace"].values.reshape(-1, 1)  # Features: Trace values
y = csv_df.drop(columns=["timestamp", "trace"])  # Target: Other bpm

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict missing bpm for binary data
bin_df["predicted_bpm"] = model.predict(bin_df["trace"].values.reshape(-1, 1))
