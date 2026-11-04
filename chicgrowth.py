import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

#print("succesfully imported")

df= pd.read_csv("chic_media_account_week.csv")
#print(df.head())
#print(df.describe())
print(df.describe(include='all'))

col= pd.DataFrame({"columnn name-": df.columns, "data type": df.dtypes})
print(col)