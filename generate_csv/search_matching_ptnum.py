import pandas as pd 
import numpy as np 
model = "Fourth"
#base_model = "First"

#correct family data
df = pd.read_csv("C:/vscode/age_estimation_forlab/patient.csv")
#df = pd.read_csv(f"D:/side_project_data/Error_Analyzed/Error_patient_under70_{base_model}_50models.csv")
#pred = df["predicted"].values
#groudTruth = df["groundTruth"].values
#age = df["error"].values
file = df["filename4"].values
pt = df["Patient_num"].values
#training data
df1 = pd.read_csv(f"D:/side_project_data/Error_Analyzed/Error_patient_under70_{model}_50models.csv")
#df1 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/coral-cacd-TL-family-trainingset{model}.csv")
#pred_tr = df1["predicted"].values
#groudTruth_tr = df1["groundTruth"].values
#age_tr = df1["error"].values
file_tr = df1["filename"].values
#pt1 = df1["Patient_num"].values
#validation data
# df2 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/new_family_validation{model}.csv")
# age_vl = df2["age"].values
# file_vl = df2["filename"].values

#testing data 
# df3 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/new_family_testing{model}.csv")
# age_ts = df3["age"].values
# file_ts = df3["filename"].values

cor_pt = []
#gt = []
ct = 0
d =  dict([(i,[a]) for i,a in zip(df['filename4'], df['Patient_num'])])
#d2 =  dict([(i,[a]) for i,a in zip(df['Patient_num'], df['groundTruth'])])
for i,filename in enumerate(file_tr):
    correct_pt = d[f'{filename}']
    cor_pt.append(correct_pt[0])
    #correct_gt = d2[f'{filename}']
    #gt.append(correct_gt[0])
    #print(correct_pt[0])
    ct+=1
print(ct)
d_tr = {'filename': file_tr, 'Patient_num': cor_pt}
print(len(file_tr),len(cor_pt))
tr = pd.DataFrame(data=d_tr)
tr.to_csv(f'D:/side_project_data/Error_Analyzed/{model}_under70_ptnum.csv')
#tr.to_csv(f'D:/side_project_data/Error_Analyzed/Error_real_age/under70_{base_model}_{model}.csv')

