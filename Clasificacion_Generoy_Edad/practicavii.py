import pandas as pd




df=pd.read_csv('training.txt',sep=':::',header=None)
df.columns=['Usuario','Sexo','Edad','extroverted','stable','agreeable','conscientious','open']
print(df)

