import pandas as pd 
import numpy as np 
model = "Second"
#base_model = "Third"

#correct family data
#df = pd.read_csv("C:/vscode/age_estimation_forlab/patient-revised.csv")  
df = pd.read_csv("D:/side_project_data/Error_Analyzed_correct/Error_patient_above70_Second_list.csv")
pt = df["Patient_num"].values
#First
df1 = pd.read_csv(f"D:/side_project_data/Error_Analyzed_correct/Error_patient_above70_{model}_50models.csv")
pt1 = df1["Patient_num"].values
files1 = df1["filename"].values
Err1 = df1["error"].values

#Second
#df2 = pd.read_csv("D:/side_project_data/Error_Analyzed_correct/Error_patient_under70_Second_50models.csv")
#Third
#df3 = pd.read_csv("D:/side_project_data/Error_Analyzed_correct/Error_patient_under70_Third_50models.csv")
#Fourth
#df4 = pd.read_csv("D:/side_project_data/Error_Analyzed_correct/Error_patient_under70_Fourth_50models.csv")



record = []
PT = []
ERR = []
#err1 = []
f1 = []
err2 = []
f2 = []
err3 = [] 
f3 = []
err4 = []
f4 = []
ct = 0
#d =  dict([(i,[a]) for i,a in zip(df['Patient_num'], df['error'])])
# d1 =  dict([(i,[a]) for i,a in zip(df1['Patient_num'], df1['error'])])
# D1 =  dict([(i,[a]) for i,a in zip(df1['Patient_num'], df1['filename'])])

# d2 =  dict([(i,[a]) for i,a in zip(df2['Patient_num'], df2['error'])])
# D2 =  dict([(i,[a]) for i,a in zip(df2['Patient_num'], df2['filename'])])

# d3 =  dict([(i,[a]) for i,a in zip(df3['Patient_num'], df3['error'])])
# D3 =  dict([(i,[a]) for i,a in zip(df3['Patient_num'], df3['filename'])])

# d4 =  dict([(i,[a]) for i,a in zip(df4['Patient_num'], df4['error'])])
# D4 =  dict([(i,[a]) for i,a in zip(df4['Patient_num'], df4['filename'])])
for i,filename in enumerate(pt):
    err1 = [] 
    for j,file in enumerate(pt1):
        if filename == pt1[j]:
            err1.append(Err1[j])
            record.append(Err1[j])
            #print(Err1[j])
   
    ERR.append(sum(err1)/len(err1))
        #print(sum(err1)/len(err1))
            #f1.append(files1[j])
        
    ct+=1
print(ERR)
print(len(ERR))
print(PT)
print(len(record))
d_tr = {'Patient_name': pt, 'Error':ERR}
tr = pd.DataFrame(data=d_tr)
tr.to_csv(f'D:/side_project_data/Error_Analyzed_correct/{model}.csv')
#tr.to_csv(f'D:/side_project_data/Error_Analyzed/Error_real_age/under70_{base_model}_{model}.csv')


