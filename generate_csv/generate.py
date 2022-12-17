import pandas as pd 
import numpy as np 
model = "02"
# for i in range(2,17):
#     if i < 10:
#         model = f"0{i}"
#         print(f"model0{i}")
#     else:
#         model = i 
#         print(f"model{i}")
#correct family data
df = pd.read_csv("C:/vscode/age_estimation_forlab/cacd-coral-TL-family_correct.csv")
age = df["agefortraining"].values
file = df["filename"].values
#training data
df1 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/new_family_training{model}.csv")
#df1 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/coral-cacd-TL-family-trainingset{model}.csv")
age_tr = df1["age"].values
file_tr = df1["filename"].values

#validation data
df2 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/new_family_validation{model}.csv")
age_vl = df2["age"].values
file_vl = df2["filename"].values

#testing data 
df3 = pd.read_csv(f"C:/vscode/age_estimation_forlab/TF_csv/new_family_testing{model}.csv")
age_ts = df3["age"].values
file_ts = df3["filename"].values

cor_age_tr = []
cor_age_vl = []
cor_age_ts = []
d =  dict([(i,[a]) for i,a in zip(df['filename'], df['agefortraining'])])
for i,filename in enumerate(file_tr):
    correct_age = d[f'{filename}']
    cor_age_tr.append(correct_age[0])

d_tr = {'filename': file_tr, 'age': cor_age_tr}
tr = pd.DataFrame(data=d_tr)
tr.to_csv(f'C:/vscode/age_estimation_forlab/TF_csv_new/new_family_training{model}.csv')

for i,filename in enumerate(file_vl):
    correct_age = d[f'{filename}']
    cor_age_vl.append(correct_age[0])

d_vl = {'filename': file_vl, 'age': cor_age_vl}
vl = pd.DataFrame(data=d_vl)
vl.to_csv(f'C:/vscode/age_estimation_forlab/TF_csv_new/new_family_validation{model}.csv')

for i,filename in enumerate(file_ts):
    correct_age = d[f'{filename}']
    cor_age_ts.append(correct_age[0])

d_ts = {'filename': file_ts, 'age': cor_age_ts}
ts = pd.DataFrame(data=d_ts)
ts.to_csv(f'C:/vscode/age_estimation_forlab/TF_csv_new/new_family_testing{model}.csv')