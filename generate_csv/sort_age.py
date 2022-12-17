import pandas as pd 
import numpy as np
import operator
import collections
#df = pd.read_csv("C:/vscode/age_estimation_forlab/TF_csv_new/coral-cacd-TL-family-testset16.csv") new_family_testing02
#df = pd.read_csv("C:/vscode/age_estimation_forlab/cacd-coral-TL-family_correct.csv")
df = pd.read_csv("C:/vscode/age_estimation_forlab/coral_data_patient_0105.csv")

filename = []
age =[]
#d = dict([(i,[a]) for i,a in zip(df['agefortraining'], df['filename'])])
d = dict([(i,[a]) for i,a in zip(df['filename'], df['age'])])
sorted_x = sorted(d.items(), key=operator.itemgetter(1))
print(sorted_x[0][0])
print(type(sorted_x))
sorted_dict = collections.OrderedDict(sorted_x)
for i in range(151):
    #sorted_x[i][0]
    filename.append(sorted_x[i][0])
    age.append(sorted_dict[sorted_x[i][0]][0])

dict = {'filename': filename, 'age': age}
ts = pd.DataFrame(data=dict)
ts.to_csv('PT_sorted_age.csv')

