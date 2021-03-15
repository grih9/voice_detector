import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import soundfile as sf

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import pathlib
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings('ignore')

x, sr = librosa.load("./samples/BUBLYAEV_ALEXEY_1.wav", sr=22050)
print(type(x), type(sr))

#<class 'numpy.ndarray'> <class 'int'>

print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

# sr = 22050 # частота дискретизации
# T = 5.0    # секунды
# t = np.linspace(0, T, int(T*sr), endpoint=False) # переменная времени
# x = 0.5*np.sin(2*np.pi*220*t) # чистая синусоидная волна при 220 Гц
# # сохранение аудио
# sf.write('tone_220.wav', x, sr)

#  1.Спектральный центроид
#Указывает, на какой частоте сосредоточена энергия спектра или,
# другими словами, указывает, где расположен «центр масс» для звука.
# Схож со средневзвешенным значением произведения
# где S(k) — спектральная величина элемента разрешения k, а f(k) — частота элемента k.
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
#print(spectral_centroids.shape)
#(775,)
# Вычисление временной переменной для визуализации
plt.figure(figsize=(12, 4))

frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Нормализация спектрального центроида для визуализации
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
# Построение спектрального центроида вместе с формой волны
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')
plt.show()

#(94316,) 22050

#  2.Спектральный спад
#  Это мера формы сигнала, представляющая собой частоту,
#  в которой высокие частоты снижаются до 0. Чтобы получить ее,
#  нужно рассчитать долю элементов в спектре мощности, где 85%
#  ее мощности находится на более низких частотах.

spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
plt.figure(figsize=(12, 4))

librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.show()

#  3. Спектральная ширина
#Спектральная ширина определяется как ширина полосы света
# на половине максимальной точки (или полная ширина на половине
# максимума [FWHM]) и представлена двумя вертикальными красными линиями
# и λSB на оси длин волн.

spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
plt.figure(figsize=(15, 9))

librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))
plt.show()

#  4. Скорость пересечения нуля
# Простой способ измерения гладкости сигнала — вычисление
# числа пересечений нуля в пределах сегмента этого сигнала.
# Голосовой сигнал колеблется медленно. Например, сигнал 100 Гц
# будет пересекать ноль 100 раз в секунду, тогда как «немой»
# фрикативный сигнал может иметь 3000 пересечений нуля в секунду.
# Более высокие значения наблюдаются в таких высоко ударных звуках,
# как в металле и роке. Теперь визуализируем этот процесс и рассмотрим
# вычисление скорости пересечения нуля.
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
plt.show()
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))#16

#  5. Мел-частотные кепстральные коэффициенты (MFCC)
# Представляют собой небольшой набор признаков (обычно около 10–20),
# которые кратко описывают общую форму спектральной огибающей.
# Они моделируют характеристики человеческого голоса.
mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)
# Отображение MFCC:
plt.figure(figsize=(15, 7))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()

#  6. Цветность
# Признак или вектор цветности обычно представлен вектором признаков из 12 элементов, в котором указано количество энергии каждого высотного класса {C, C#, D, D#, E, …, B} в сигнале. Используется для описания меры сходства между музыкальными произведениями.
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=12)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=12, cmap='coolwarm')
plt.show()

cmap = plt.get_cmap('inferno')
plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
plt.axis('off');
plt.savefig(f'spectrogram Alex.png')
plt.clf()

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

with open('dataset1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    #for filename in os.listdir(f'./drive/My Drive/genres/{g}'):
        #songname = f'./drive/My Drive/genres/{g}/{filename}'
        #y, sr = librosa.load(songname, mono=True, duration=30)
    rmse = librosa.feature.rms(x)
    chroma_stft = librosa.feature.chroma_stft(x, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(x, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(x, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(x)
    mfcc = librosa.feature.mfcc(x, sr=sr)
    to_append = f'{"./samples/BUBLYAEV_ALEXEY_1.wav"} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'

    with open('dataset1.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
with open('dataset1.csv', 'r', newline='') as file:
    lines = file.readlines()
    print(len(lines))
    print(lines)
    print([len(line.split(',')) for line in lines])