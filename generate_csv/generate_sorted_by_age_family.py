import pandas as pd 
import os
import random
import numpy as np
#df = pd.read_csv("C:/vscode/age_estimation_forlab/patient_roundup.csv")
#df = pd.read_csv("sorted_age_family.csv")
df = pd.read_csv("C:/vscode/age_estimation_forlab/matched_pt_fm.csv")
filename = df["filename"].values
ages = df["age"].values
files_valid = []
ag_valid = []
files_test = []
ag_test = []
filename_delete_list =[]
age_delete_list =[]
print(type(filename))
#assign new file number

NUMBER = int(len(os.listdir('C:/vscode/age_estimation_forlab/TF_csv_fm_603/'))/3)+1
if NUMBER<10:
    NUMBER =f"0{NUMBER}"
else:
    NUMBER=f"{NUMBER}"
print(f"FILE:{NUMBER}")
for i in range(14):
    if i == 13:
        #generate two random number
        a , b = random.choices(range(10*i,10+10*i+7),k=2)
        for rep in range(20): #repeat 20 times
            if a == b:
                a , b = random.choices(range(10*i,10+10*i),k=2)
            else:
                break
        #d = random.randint(10*i,10+10*i+1)
        filename_delete_list.append(a)
        filename_delete_list.append(b)
        age_delete_list.append(a)
        age_delete_list.append(b)
        #choose a filename 
        files_valid.append(filename[a])
        #np.delete(filename,c)
        files_test.append(filename[b])
        #np.delete(filename,d)
        #in case repeating
        ag_valid.append(ages[a])
        #np.delete(ages,c)
        ag_test.append(ages[b])
        #np.delete(ages,d)

    else:#generate two random number
        c , d = random.choices(range(10*i,10+10*i),k=2)
        for rep in range(20): #repeat 20 times
            if c == d:
                c , d = random.choices(range(10*i,10+10*i),k=2)
            else:
                break
        #d = random.randint(10*i,10+10*i+1)
        filename_delete_list.append(c)
        filename_delete_list.append(d)
        age_delete_list.append(c)
        age_delete_list.append(d)
        #choose a filename 
        files_valid.append(filename[c])
        #np.delete(filename,c)
        files_test.append(filename[d])
        #np.delete(filename,d)
        #in case repeating
        ag_valid.append(ages[c])
        #np.delete(ages,c)
        ag_test.append(ages[d])
        #np.delete(ages,d)

print(filename_delete_list)        
print("training set number",len(filename)) 
filename = np.delete(filename,filename_delete_list)
ages = np.delete(ages,age_delete_list)

print(len(filename))  
print(len(files_valid))
print(len(files_test)) 
dict_valid = {'filename': files_valid, 'age': ag_valid}
dict_test = {'filename': files_test, 'age': ag_test}
dict_train = {'filename': filename, 'age': ages}

val = pd.DataFrame(data=dict_valid)
test = pd.DataFrame(data=dict_test)
train = pd.DataFrame(data=dict_train)

val.to_csv(f'C:/vscode/age_estimation_forlab/TF_csv_fm_603/valid{NUMBER}.csv')
test.to_csv(f'C:/vscode/age_estimation_forlab/TF_csv_fm_603/test{NUMBER}.csv')
train.to_csv(f'C:/vscode/age_estimation_forlab/TF_csv_fm_603/train{NUMBER}.csv')
# val.to_csv('C:/vscode/age_estimation_forlab/TF_csv_patient/valid01.csv')
# test.to_csv('C:/vscode/age_estimation_forlab/TF_csv_patient/test01.csv')
# train.to_csv('C:/vscode/age_estimation_forlab/TF_csv_patient/train01.csv')