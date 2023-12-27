import pandas as pd 
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import librosa 
import librosa.display
from glob import glob 
import numpy as np
#import Ipython.display as ipd
data =pd.read_csv("rec.csv")
range = data.Freq
y,sr =librosa.load('abnormal.wav')
fre= y.max()*1000
print("The given audio frequency",fre)
print()
if(fre >0 and fre <range[30]):
    print("This given sample is calm")
elif(fre >range[30] and fre <range[50]):
    print("This given sample is normal")
elif(fre >range[50]):
    print("This given sample is abnormal")
pd.Series(y).plot(figsize = (10,5), lw=1, title='Test 1')
plt.grid()
plt.show()
d= librosa.stft(y)
s_db =librosa.amplitude_to_db(np.abs(d),ref =np.max)
fig , ax =plt.subplots(figsize =(10,5))
img= librosa.display.specshow(s_db, x_axis='time',y_axis='log',ax=ax)
plt.show()
