import librosa
import numpy as np
import os
import pandas as pd
allparameters=[]

path_to_voice="/mnt/c/Users/DELL/Desktop/soc/voice/"


for filename in os.listdir(path_to_voice):
    # sr is samples per second and len(y_t) is total number of samples
    y_t, sr = librosa.load(path_to_voice+filename)
    y_1, index = librosa.effects.trim(y_t)
    length = 5 # in seconds
    # samples_per_chunk = sr * chunk_length
    y=y_1[0:sr * length]
    # for i in range(len(y_1)//samples_per_chunk):
    #     y_2.append(y_1[i * samples_per_chunk:(i + 1) * samples_per_chunk])
    # for y in y_2:
    parameters=[]
    chroma= np.mean(np.mean(librosa.feature.chroma_stft(y=y, sr=sr),axis=1))
    # print(librosa.feature.chroma_stft(y=y, sr=sr).shape)
    parameters.append(chroma)
    rms = np.mean(librosa.feature.rms(y=y))
    # print(librosa.feature.rms(y=y).shape)
    parameters.append(rms)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    parameters.append(centroid)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    parameters.append(bandwidth)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    parameters.append(rolloff)
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y))
    parameters.append(zero_crossings)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20),axis=1)
    for i in range(20):
        parameters.append(mfccs[i])
    parameters.append('fake')
    allparameters=[]
    allparameters.append(parameters)
    columns=['Chroma_Stft','rms','Spectral_Centroid','Spectral_Bandwidth','Rolloff','Zero_Crossing_Rate']
    for i in range(20):
        columns.append('mfcc'+str(i+1))
    columns.append('Label')
    x = pd.DataFrame(allparameters, columns=columns)
    x.to_csv("data_"+filename+".csv", mode='a', header=True, index=False)

# print('Chroma STFT:',parameters[0])
# print('RMS:',parameters[1])
# print('Spectral Centroid:',parameters[2])
# print('Spectral Bandwidth:',parameters[3])
# print('Spectral Rolloff:',parameters[4])
# print('Zero Crossing Rate:',parameters[5])
# for i in range(20):
#     print('mfcc',i+1,':',parameters[i+6])