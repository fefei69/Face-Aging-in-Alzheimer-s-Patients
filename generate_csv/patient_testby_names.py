import pandas as pd 
import numpy as np 
import os

base_path = "C:/vscode/age_estimation_forlab/TF_csv_new/"
df = pd.read_csv("C:/vscode/age_estimation_forlab/patient-revised.csv")
file = df["filename1"].values

df_ori = pd.read_csv("C:/vscode/age_estimation_forlab/patient_50models.csv")

df1 = pd.read_csv("C:/vscode/age_estimation_forlab/Error_patient_ALL_50models_527.csv")
err = df1["error"].values
file2 = df1["filename"].values

for i,fl in enumerate(file):
    err_list = []
    for j,filename in enumerate(file2):
        if fl == filename:
            err_list.append(err[j])

    df_ori[f"{fl}"] = err_list
            
       
    
    
      
 
    


df_ori.to_csv("C:/vscode/age_estimation_forlab/patient_num_test_ALL_527.csv")