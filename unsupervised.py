import pandas as pd
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from glob import glob
import warnings
import numpy as np
warnings.filterwarnings("ignore")
audio_files= glob('audio/*.wav')
mfccs =[]
spectral_centroid =[]

l= len(audio_files)

for i in range(l):

    animal,sr= librosa.load(audio_files[i])
    #features that extracting 
    
    mfccs_anim = librosa.feature.mfcc(y= animal, sr= sr,n_mfcc=20)
    spectral_centroids_anims = librosa.feature.spectral_centroid(y= animal, sr=sr)
    mfccs.append(max(mfccs_anim[9]))
    spectral_centroid.append(max(max(spectral_centroids_anims)))


df =pd.DataFrame()
df['mfcc']= mfccs
df['spectral']= spectral_centroid
scaler= StandardScaler()
df[['mfcc_t', 'spectral_t']]= scaler.fit_transform(df[["mfcc",'spectral']])
km= KMeans(n_clusters=2)
y_predict =km.fit_predict(df[['mfcc','spectral']])
df['cluster'] =y_predict
print(df)
plt.scatter(df['mfcc_t'],df['spectral_t'], c=df['cluster'])
plt.grid()
plt.xlabel('MFCC')
plt.ylabel('SPECTRAL CENTROID')
plt.show()
y,sr = librosa.load(audio_files[0])
pd.Series(y).plot(figsize =(10,5), lw= 1,title ="test 1")
plt.grid()
d =librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs (d), ref=np.max)
fig, ax = plt.subplots(figsize = (10,5)) 
img =librosa.display.specshow(S_db, x_axis='time', y_axis= 'log',ax=ax)
plt.show()

mfc =librosa.feature.mfcc(y=y,sr=sr)
plt.figure(figsize =(10,4))
librosa.display.specshow(mfc)
plt.colorbar(format='%4.0f dB')
plt.title('MFCCs')
plt.show()