import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model # library used for Machine learning, we are using this for Linear Regression
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('epo2.csv') 


# Linear regression code (unchanged)
reg=linear_model.LinearRegression() #defining a linear Regression function in reg model 
reg.fit(df[['ANX']],df[['EEG_epochs']]) #now we are provinding the x and y coordinates fuction for our Linear reg model 
print(reg.predict([[10.2]]))# pedicting a epochs value by providing a random anx value
#print("anxiety value from epochs value ")
#reg.fit(df[['EEG epochs']],df[['ANX']])
#print(reg.predict([[22.446]]))
#plotting our Linear regression graph 
plt.title('Linear Regression')
plt.xlabel('Range of ANX') 
plt.ylabel('Epochs')
plt.scatter(df['ANX'],df['EEG_epochs'])
plt.plot(df[['ANX']], reg.predict(df[['ANX']]),color = 'blue')
plt.grid() 
plt.show()