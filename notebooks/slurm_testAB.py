#!/usr/bin/env python
# coding: utf-8

###### Package Imports and Class Creations
### Import Packages ###
import pandas as pd
import numpy as np
import seaborn as sn
import struct
import os
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta

#Jlab Packages
from data_utils import get_traces
from beam_settings_parser_hdf5 import BeamConfigParserHDF5
from beam_settings_prep import BeamConfigPreProcessor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.display.max_columns = None

start_time = datetime.now()
print(start_time.strftime("%H:%M:%S"))

### Need to clean up since we are now only working with the September Data
class BPMDataConfig:

    def __init__(self):
        self.beam_settings_data_path = "/work/data_science/suf_sns/beam_configurations_data/processed_data/clean_beam_config_processed_df.csv"
        self.beam_param_parser_cfg = {"data_location": "/work/data_science/suf_sns/beam_configurations_data/hdf5_sept2024/"}
        self.beam_settings_prep_cfg = {
            "rescale": False,
            "beam_config": [
                'FE_IS:Match:TunerPos',
                'LEBT:Chop_N:V_Set',
                'LEBT:Chop_P:V_Set',
                'LEBT:Focus_1:V_Set',
                'LEBT:Focus_2:V_Set',
                'LEBT:Steer_A:V_Set',
                'LEBT:Steer_B:V_Set',
                'LEBT:Steer_C:V_Set',
                'LEBT:Steer_D:V_Set',
                'Src:Accel:V_Set',
                'Src:H2:Flw_Set',
                'Src:Ign:Pwr_Set',
                'Src:RF_Gnd:Pwr_Set',
                'ICS_Chop:RampDown:PW',
                'ICS_Chop:RampUp:PWChange',
                'ICS_MPS:Gate_Source:Offset',
                'ICS_Tim:Chop_Flavor1:BeamOn',
                'ICS_Tim:Chop_Flavor1:OnPulseWidth',
                'ICS_Tim:Chop_Flavor1:RampUp',
                'ICS_Tim:Chop_Flavor1:StartPulseWidth',
                'ICS_Tim:Gate_BeamRef:GateWidth',
                'ICS_Tim:Gate_BeamOn:RR'
            ]
        }
        self.beam_config = [
            'timestamps',
            'FE_IS:Match:TunerPos',
            'LEBT:Chop_N:V_Set',
            'LEBT:Chop_P:V_Set',
            'LEBT:Focus_1:V_Set',
            'LEBT:Focus_2:V_Set',
            'LEBT:Steer_A:V_Set',
            'LEBT:Steer_B:V_Set',
            'LEBT:Steer_C:V_Set',
            'LEBT:Steer_D:V_Set',
            'Src:Accel:V_Set',
            'Src:H2:Flw_Set',
            'Src:Ign:Pwr_Set',
            'Src:RF_Gnd:Pwr_Set',
            'ICS_Chop:RampDown:PW',
            'ICS_Chop:RampUp:PWChange',
            'ICS_MPS:Gate_Source:Offset',
            'ICS_Tim:Chop_Flavor1:BeamOn',
            'ICS_Tim:Chop_Flavor1:OnPulseWidth',
            'ICS_Tim:Chop_Flavor1:RampUp',
            'ICS_Tim:Chop_Flavor1:StartPulseWidth',
            'ICS_Tim:Gate_BeamRef:GateWidth',
            'ICS_Tim:Gate_BeamOn:RR'
        ]

        self.column_to_add = [
    'FE_IS:Match:TunerPos',
    'LEBT:Chop_N:V_Set',
    'LEBT:Chop_P:V_Set',
    'LEBT:Focus_1:V_Set',
    'LEBT:Focus_2:V_Set',
    'LEBT:Steer_A:V_Set',
    'LEBT:Steer_B:V_Set',
    'LEBT:Steer_C:V_Set',
    'LEBT:Steer_D:V_Set',
    'Src:Accel:V_Set',
    'Src:H2:Flw_Set',
    'Src:Ign:Pwr_Set',
    'Src:RF_Gnd:Pwr_Set',
    'ICS_Tim:Gate_BeamOn:RR',
    'ICS_Chop-RampDown-PW',
    'ICS_Chop-RampUp-PWChange',
    'ICS_Tim-Gate_BeamRef-GateWidth'
]

        self.rename_mappings = {
    'ICS_Chop-RampDown-PW': 'ICS_Chop:RampDown:PW',
    'ICS_Chop-RampUp-PWChange': 'ICS_Chop:RampUp:PWChange',
    'ICS_MPS-Gate_Source-Offset': 'ICS_MPS:Gate_Source:Offset',
    'ICS_Chop-BeamOn-Width': 'ICS_Tim:Chop_Flavor1:BeamOn',
    'ICS_Chop-BeamOn-PW': 'ICS_Tim:Chop_Flavor1:OnPulseWidth',
    'ICS_Chop-RampUp-Width': 'ICS_Tim:Chop_Flavor1:RampUp',
    'ICS_Chop-RampUp-PW': 'ICS_Tim:Chop_Flavor1:StartPulseWidth',
    'ICS_Tim-Gate_BeamRef-GateWidth': 'ICS_Tim:Gate_BeamRef:GateWidth'
}


    def configs_hist(self, dataframe, timestamp):
        subset_columns = dataframe.columns.tolist()
        subset_columns.remove(timestamp)
        df_shifted = dataframe[subset_columns].shift(1)
        mask = (dataframe[subset_columns] == df_shifted).all(axis=1)
        dataframe = dataframe[~mask]

        dataframe['time_diff'] = dataframe[timestamp].diff()
        dataframe['timestamps_trm'] = dataframe[timestamp] + dataframe['time_diff'].shift(-1) - timedelta(seconds=0.000001)

        subset_columns.insert(0, timestamp)
        subset_columns.insert(1, "timestamps_trm")

        return dataframe[subset_columns]

    def summary(self, text, df):
        print(f'{text} shape: {df.shape}')

        # Filter for numeric columns only
        numeric_cols = df.select_dtypes(include=['number'])

        summ = pd.DataFrame(numeric_cols.dtypes, columns=['dtypes'])
        summ['null'] = numeric_cols.isnull().sum()
        summ['unique'] = numeric_cols.nunique()
        summ['min'] = numeric_cols.min()
        summ['median'] = numeric_cols.median()
        summ['max'] = numeric_cols.max()
        summ['mean'] = numeric_cols.mean()
        summ['std'] = numeric_cols.std()
        summ['duplicate'] = df.duplicated().sum()

        return summ


    def update_beam_config(self,beam_config_df):
        for col in self.column_to_add:
            if col not in beam_config_df.columns:
                beam_config_df[col] = np.nan

        beam_config_df.rename(columns=self.rename_mappings, inplace=True)
        return beam_config_df


# Create an instance of BPMDataConfig
dc = BPMDataConfig()

### Sep24 hdf5 beam settings ###
parser = BeamConfigParserHDF5(dc.beam_param_parser_cfg)
data, _ = parser.run()

### Get Prepared datasets ###
prep = BeamConfigPreProcessor(dc.beam_settings_prep_cfg)
prepared_settings, run_cfg = prep.run(data)

normal_settings = prepared_settings
normal_settings['n_beam'] = normal_settings['ICS_Tim:Gate_BeamOn:RR'].apply(lambda x: 0 if x <= 59.90 else 1)
normal_settings['n_beamon'] = normal_settings['ICS_Tim:Chop_Flavor1:BeamOn'].apply(lambda x: 0 if x <= 850 else 1)
normal_settings = normal_settings[(normal_settings['n_beam'] == 1) & (normal_settings['n_beamon'] == 1)]

par_inv = [
    'ICS_MPS:Gate_Source:Offset',
    'Src:H2:Flw_Set',
    'ICS_Tim:Chop_Flavor1:OnPulseWidth',
    'Src:Accel:V_Set',
    'Src:Ign:Pwr_Set',
    'ICS_Tim:Chop_Flavor1:BeamOn',
    'ICS_Tim:Gate_BeamRef:GateWidth',
    'ICS_Chop:RampUp:PWChange'

]

###### Load Tracing Data
dataset2_loc = "/w/data_science-sciwork24/suf_sns/DCML_dataset_Sept2024"
length_of_waveform = 10000

filtered_normal_files = []
filtered_anomaly_files = []
subfolders = [ f.path for f in os.scandir(dataset2_loc) if f.is_dir() ]
for directory in subfolders:
    if "normal" in directory or "anomal" in directory:                
        for root, subfolders, files in os.walk(directory):
            for file in files:
                full_path = root
                if ".gz" in file:
                    if 'normal' in directory:
                        filtered_normal_files.append(os.path.join(full_path, file))
                    elif "anomal" in directory:
                        filtered_anomaly_files.append(os.path.join(full_path, file))

file_date = []
flag = []
for file in filtered_normal_files:
    file_time = file[-32:-12]
    file_time = file_time.replace("_","")
    file_time = datetime.strptime(file_time,'%Y%m%d%H%M%S.%f')
    file_date.append(file_time)
    flag.append(0)

for file in filtered_anomaly_files:
    file_time = file[-32:-12]
    file_time = file_time.replace("_","")
    file_time = datetime.strptime(file_time,'%Y%m%d%H%M%S.%f')
    file_date.append(file_time)
    flag.append(1)

### Differential Current Monitor (DCM)
dcm = pd.DataFrame({'anomaly_flag':flag, 'timestamps':file_date, 'file':filtered_normal_files+filtered_anomaly_files}, )

merged_df = pd.merge_asof(
    dcm.sort_values("timestamps"), 
    prepared_settings.sort_values("timestamps"), 
    on="timestamps", 
    direction="nearest"
)
merged_df = merged_df[(merged_df['n_beam'] == 1) & (merged_df['n_beamon'] == 1)]
merged_df = merged_df.drop(columns=['n_beamon','n_beam'])
merged_df = merged_df.sort_values("timestamps")
merged_df = merged_df.reset_index(drop=True)


####### Create Groups of files based on Parameter settings

prepared_settings = prepared_settings.drop(columns=['n_beamon','n_beam'])
subset_columns = prepared_settings.columns.tolist()
subset_columns.remove('timestamps')

df_shifted = merged_df[subset_columns].shift(1)

mask = (merged_df[subset_columns] == df_shifted).all(axis=1)
mask_df = pd.DataFrame(mask, columns = ['mtch_flg'])

merged_df = merged_df.join(mask_df)

def parametergroup(df, columns, column_name='group'):

    df[column_name] = 0
    for i in range(1, len(df)):
        if df.iloc[i][columns] is np.True_:
        # if df[columns][i] == True:
            df.loc[i, column_name] = df.loc[i - 1, column_name]
        else:
            df.loc[i, column_name] = df.loc[i - 1, column_name] + 1
    return df

merged_df = parametergroup(merged_df, 'mtch_flg', 'group')
merged_df = merged_df.drop(columns=['mtch_flg'])
groups = pd.DataFrame(merged_df.groupby(['group'])['file'].count())
groups = groups.rename(columns={'file': 'file_ct'})

merged_df = pd.merge(merged_df, groups, on=['group'], how = 'left')
merged_df = merged_df[merged_df['file_ct'] > 25]

traces = []
timestamps = []
file = []

for filepath in merged_df['file']:
    if filepath[-40:-39] == 's':
        try:
            trace, timestamp = get_traces(filepath, var_id="Trace2", begin=3000, shift=length_of_waveform, data_type=-1, alarm=48)
            for sample in trace:
                trace = sample
                traces.append(trace)
                file.append(filepath)
            for stamp in timestamp:
                timestamps.append(stamp)
        except:
            pass
    else:
        try:
            trace, timestamp = get_traces(filepath, var_id="Trace2", begin=3000, shift=length_of_waveform, data_type=-1, alarm=0) 
            for sample in trace:
                trace = sample
                traces.append(trace)
                file.append(filepath)
            for stamp in timestamp:
                timestamps.append(stamp)
        except:
            pass

trace_df = pd.DataFrame({'file':file, 'traces':traces, 'sample_timestamp': timestamps})
sns = pd.merge(merged_df, trace_df, on=['file'], how = 'inner')

print(sns.head())
print(sns.info())
end_time = datetime.now()
print(end_time.strftime("%H:%M:%S"))