# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from pathlib import Path

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()  

if __name__ == "__main__":
    reg_type = "Humidity"
    reg_folder = "Regression "+reg_type
    name_dataset = "" 
    results_folder = ""
    type_data =  "genes_profile"

    df_regression = pd.DataFrame()
    
    # Load Data:
    data_MGS = pd.read_csv(name_dataset+'/'+name_dataset+'_'+type_data+'_data_56_samples.csv', header = [0], index_col=[0])
    features_name = data_MGS.columns
    sample_name = data_MGS.index
        
    data = np.array(data_MGS)
    print(data.shape) 
        
    # Load Metadata:
    metadata_df = pd.read_csv(name_dataset+"/"+name_dataset+"_metadata_56_samples.csv", header = [0])
    metadata_samples = metadata_df[metadata_df.columns[0]]

    # Load Antibiotic Data:
    antibiotic_df = pd.read_csv(name_dataset+'/'+name_dataset+'_AMR_RSI_data.csv', header = [0])
    
    args_data = pd.read_csv("CARD_ARG_drugclass_NCBI.csv", header=[0])

    print(antibiotic_df.columns[1:])
    for name_antibiotic in antibiotic_df.columns[1:]:
        print("Antibiotic: {}".format(name_antibiotic))
        
        target_str = np.array(antibiotic_df[name_antibiotic])
        
        target = np.zeros(len(target_str)).astype(int)
        idx_S = np.where(target_str == 'S')[0]
        idx_R = np.where(target_str == 'R')[0]
        idx_NaN = np.where((target_str != 'R') & (target_str != 'S'))[0]
        target[idx_R] = 1    

        if len(idx_NaN) > 0:
            target = np.delete(target,idx_NaN)
            
        # Check minimum number of samples:
        count_class = Counter(target)
        print(count_class)
        if count_class[0] < 6 or count_class[1] < 6:
            continue
        
        directory = name_dataset+"/"+results_folder+"/"+type_data
        file_name = directory+"/features_"+name_dataset+"_"+name_antibiotic+".csv"
        my_file = Path(file_name)

        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            continue
        
        df_features = pd.read_csv(file_name, header=[0])
        n_features = df_features.shape[0]
        
        # Preprocessing - Feature Selection
        std_scaler = MinMaxScaler()
        data = std_scaler.fit_transform(data)

        y = np.array(metadata_df[reg_type])
        
        update_progress(0)
        n_feat = len(df_regression)
        for j in range(n_features):
            id_feature = np.where(features_name == df_features.iloc[j,0])[0]
            x = data[:,id_feature[0]]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            id_arg = np.where(args_data["Source"] == df_features.iloc[j,0])[0]
            df_regression.loc[n_feat+j,"feature"] = df_features.iloc[j,0]
            df_regression.loc[n_feat+j,"Antibiotic Class (CARD)"] = args_data.iloc[id_arg[0],1]
            df_regression.loc[n_feat+j,"slope"] = slope
            df_regression.loc[n_feat+j,"intercept"] = intercept
            df_regression.loc[n_feat+j,"r2 value"] = r_value
            df_regression.loc[n_feat+j,"p-value"] = p_value
            df_regression.loc[n_feat+j,"std_err"] = std_err
            
            
            update_progress((j+1)/n_features)
        
    cols = np.where(df_regression["p-value"] < 0.05)[0]
    
    df_regression = df_regression.loc[cols,:]
    
    directory = name_dataset+"/"+results_folder+"/"+type_data+"/"+reg_folder

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    df_regression.to_csv(directory+"/features_"+reg_type+"_"+name_dataset+".csv", index=False)
