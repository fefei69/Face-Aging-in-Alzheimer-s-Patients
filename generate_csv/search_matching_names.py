import pandas as pd 
import numpy as np 
model = "Fourth"
base_model = "Third"

# for i in range(2,17):
#     if i < 10:
#         model = f"0{i}"
#         print(f"model0{i}")
#     else:
#         model = i 
#         print(f"model{i}")
#correct family data
#df = pd.read_csv("C:/vscode/age_estimation_forlab/patient.csv")
df = pd.read_csv(f"D:/side_project_data/Error_Analyzed/Error_patient_under70_{base_model}_50models.csv")
pred = df["predicted"].values
groudTruth = df["groundTruth"] 
age = df["error"].values
file = df["filename"].values
#pt = df["Patient_num"].values
#training data
df1 = pd.read_csv(f"D:/side_project_data/Error_Analyzed/Error_patient_under70_{model}_50models.csv")
#df1 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/coral-cacd-TL-family-trainingset{model}.csv")
pred_tr = df1["predicted"].values
groudTruth_tr = df1["groundTruth"] 
age_tr = df1["error"].values
file_tr = df1["filename"].values
pt = df1["Patient_num"].values
#validation data
# df2 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/new_family_validation{model}.csv")
# age_vl = df2["age"].values
# file_vl = df2["filename"].values

#testing data 
# df3 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/new_family_testing{model}.csv")
# age_ts = df3["age"].values
# file_ts = df3["filename"].values

cor_pt = []
gt = []
PT = []
ct = 0
d =  dict([(i,[a]) for i,a in zip(df['Patient_num'], df['error'])])
d2 =  dict([(i,[a]) for i,a in zip(df['Patient_num'], df['groundTruth'])])
for i,filename in enumerate(pt):
    if filename != 'P134' and filename !='P224':
        correct_pt = d[f'{filename}']
        cor_pt.append(correct_pt[0])
        correct_gt = d2[f'{filename}']
        gt.append(correct_gt[0])
        PT.append(pt[i])
        #print(correct_pt[0])
    ct+=1
print(ct)
print(len(pt),len(cor_pt),len(gt))
d_tr = {'filename': PT, f'{base_model}_error': cor_pt ,f"{base_model}_groundTruth": gt}
tr = pd.DataFrame(data=d_tr)
#tr.to_csv(f'D:/side_project_data/Error_Analyzed/Error/{model}.csv')
tr.to_csv(f'D:/side_project_data/Error_Analyzed/Error_real_age/under70_{base_model}_{model}.csv')


