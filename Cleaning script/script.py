import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Input file for cleaning")
parser.add_argument("output_file",help="Cleansed output file")
args = parser.parse_args()


file = pd.read_csv(args.input_file)
#Dataset info
print(file.info())
num_cols = file.select_dtypes(include=['int64','float64']).columns
fig1= plt.figure(figsize=(12,6))
plt.scatter(file['housing_median_age'],file['households'] , alpha=0.5)
fig1.savefig("househols by housing median age - scatter.png")
plt.show()
fig2= plt.figure(figsize=(12,6))
plt.hist(file['total_rooms'],bins=50)
fig2.savefig("total rooms - histogram.png")
plt.show()
fig3= plt.figure(figsize=(12,6))
sns.heatmap(file[num_cols].corr(),annot=True,cmap='coolwarm')
fig3.savefig("feature correlation - heatmap.png")
plt.show()
# #Dropping the duplicates
file = file.drop_duplicates()
#Handling missing values
file[num_cols] = file[num_cols].fillna(file[num_cols].mean()) #Filing NA values in numerical columns with mean
file = file.dropna()# Dropping the remaining NA values
num_cols = file.select_dtypes(include=['int64','float64']).columns #Re-selection of numerical columns
#Min-Max normalization for numerical columns + min ! max test
for c in num_cols:
    if file[c].min() != file[c].max():
        file[c]= (file[c]-file[c].min())/(file[c].max()-file[c].min())
    else:
        continue
file.to_csv(args.output_file)#Saving cleaned file

