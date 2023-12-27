import pandas as pd 
import librosa 
from glob import glob 
audio_files = glob( 'Actor_01/*.wav')
lst= []
len = len(audio_files)
for i in range(len):
    y,sr =librosa.load(audio_files[i])
    lst.append(y.max()*1000)
lst.sort()
df2 =pd.DataFrame(lst)
df2.columns= ['Freq']
df2.to_csv("rec.csv")
print(df2)