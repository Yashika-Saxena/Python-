# Task 2 Performance Analysis of Linear Regression (EEG Data Signal)
# Dr.Rajesh Muthu/21.08.2023
from glob import glob
import numpy as np
import pandas as pd
import mne # library used for extracting eeg
from matplotlib import  pyplot as plt
data_set = glob('dataverse_files/*.edf')

def read_data(file_path): # defining a function to extract data from our data from our EEG signal
   data = mne.io.read_raw_edf(file_path,preload=True) # reading our raw EEG data
   epochs = mne.make_fixed_length_epochs(data,duration = 5, overlap = 1)
   array = epochs.get_data() # extracting data inform of array
   return  max(array[0][0]*1000000) # taking out the individual maxi value & mutipling with 1L for efficient plotting
data_array = [read_data(i) for i in data_set] #appending all the maximum values of our data by function using our read_data function
df = pd.DataFrame()# creating a data frame using pandas
df['EEG_epochs']  = data_array#appending our data into a datframe
df2 =pd.read_csv('Sor.csv')# reading our source file  using pandas
df['ANX'] = df2['ANX'] # appending our anx range into df datframe
df.to_csv('epo2.csv')# converting that into a csv file
# signal Representation
raw = mne.io.read_raw_edf(data_set[0])# same process that we have done in raw_data function
epochs1 = mne.make_fixed_length_epochs(raw,duration = 5, overlap = 1)
arr = epochs1.get_data()
print(arr[0][0])
pd.Series(arr[0][0]).plot(figsize = (10,5),lw = 1,title = 'Sample 1') #plotting the data
plt.xlabel('Sample')
plt.ylabel('Epochs')
plt.grid()
plt.show()
