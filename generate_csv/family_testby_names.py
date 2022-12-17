import pandas as pd 
import numpy as np 
import os

base_path = "C:/vscode/age_estimation_forlab/TF_csv_new/"
df = pd.read_csv("C:/vscode/age_estimation_forlab/cacd-coral-TL-family_correct.csv")
file = df["filename"].values
df1 = pd.read_csv("C:/vscode/age_estimation_forlab/family_num_test2.csv")
model = df1["Models"].values
d_m =  dict([(i,[a]) for i,a in zip(df1['Models'], df1['index'])])
#model->csv
df2 = pd.read_csv("C:/vscode/age_estimation_forlab/best_model_morph_50models.csv")
#
df3 = pd.read_csv("C:/vscode/age_estimation_forlab/Error_family_testset_ALL_50models.csv")
file3 = df3["filename"].values
model3 = df3["model"].values
err3 = df3["error"].values
d_e =  dict([(i,[a]) for i,a in zip(df3['index'], df3['error'])])
d_i =  dict([(i,[a]) for i,a in zip(df3['index'], df3['filename'])])
d =  dict([(i,[a]) for i,a in zip(df['filename'], df['agefortraining'])])
d1=  dict([(i,[a]) for i,a in zip(df2['Model'], df2['CSV'])])
d2=  dict([(i,[a]) for i,a in zip(df3['filename'], df3['error'])])
#print(d1[140])
count = 0
for i,fl in enumerate(file):
    #print("###########",fl)
    Error_list = []
    count+=1
    for m in model:
        #print("m",type(m))
        m_int =int(m)
        csv = d1[m_int]
        index = d_m[m]
        index = int(index[0])
        
        #print("********************",csv[0])
        path = os.path.join(base_path,csv[0])
        df_n = pd.read_csv(f"{path}")
        filename = df_n["filename"].values
        del df_n
        ct = 15
        for f in filename:
            if f == fl:
                ct-=1
                for j in range(15*index,15*index+15):
                    if d_i[j][0] == fl:
                        err = d_e[j]
                Error_list.append(err[0])
        if ct == 15:  #if the csv file doesn't contain the filename
            Error_list.append(np.nan)
        #print(ct)
        #print(csv[0])
    #print(len(Error_list))
    try:   
        df1[f"{fl}"] = Error_list
    except ValueError:
        print("value error")
        print(fl)
        print(count)
        print(len(Error_list))

      
#print(count)
#print(df1)  

#df1.to_csv("C:/vscode/age_estimation_forlab/family_num_test3.csv")