# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:40:03 2024

@author: jwang5
"""

import pandas as pd
import os
import numpy as np


#%% identify unsimulated cases except timed out ones
input_path=os.getcwd()

df=pd.read_csv(os.path.join(input_path, 'simulation_log.csv'))
df2=pd.read_csv(os.path.join(input_path, 'control_scenarios_processed.csv'))

# order by upgrade and then building id
df = df.sort_values(by=["Upgrade", "Building ID"], ascending=[True, True])

# double check duplicates
consecutive_duplicates = df['Building ID'] == df['Building ID'].shift()

for i in range(len(consecutive_duplicates)):
    if consecutive_duplicates.iloc[i]:
        print(f'Found duplicate at index {i}')
        # df = df.drop(i)
        
# df.to_csv(os.path.join(input_path, 'simulation_log.csv'), index=False) 

# generate a spreadsheet for failed simulations to rerun
filtered_df = df[df["Status"].str.contains("Completed")]
# check if ochre_complete exists in result folder
for i in range(len(filtered_df)):
    upgrade = filtered_df['Upgrade'].iloc[i]
    bldg_id = filtered_df['Building ID'].iloc[i]
    # result_path = os.path.join(os.getcwd(), '..', 'upgrade'+str(upgrade), 'results', 'simulation_output', 'up01',  'bldg'+str(bldg_id).zfill(7), 'run', 'ochre_output')
    result_path = os.path.join(os.getcwd(), 'upgrade'+str(upgrade), str(bldg_id), 'ochre_output')
    if not os.path.exists(os.path.join(result_path, 'OCHRE_complete')):
        filtered_df['Status'].iloc[i] = 'Failed'

filtered_df = filtered_df[filtered_df["Status"].str.contains("Failed")]

filtered_df2 = df2[df2["building_id"].isin(filtered_df["Building ID"])]
filtered_df2.to_csv(os.path.join(input_path, 'control_scenarios_cs_failed.csv'), index=False)


#%% code for analyzing results


    
    
    
    
    
    
    
    
    
        
        
        
        
        