import librosa
import librosa.display
import sklearn

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
from sklearn.preprocessing import LabelEncoder, StandardScaler, minmax_scale
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, tree
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

USERS = ["BUBLYAEV_ALEXEY", "TOLSTIKOV_GRIGORIY", "KORSHUNOV_KIRILL",
         "FIRSOV_DANIIL", "AKHMEDOV_ABDULLA", "BESEDIN_DANIIL",
         "KOTOV_IVAN", "DENISOVA_EKATERINA", "LOGVINENKO_ALYONA"]

train_size = 0.7
n_iters = 3000
file_name = 'dataset.csv'

def write(add_new=False, users_to_add=None):
    if users_to_add is None:
        users_to_add = []
    header = 'label chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    #header += ' label'
    header = header.split()
    users = USERS
    if not add_new:
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    else:
        users = users_to_add
    for user in users:
        for frame in range(1, 50):
            try:
                x, sr = librosa.load(f"./samples/{user} ({frame}).wav", sr=44100)
            except FileNotFoundError:
                continue

            # print(type(x), type(sr))
            # # <class 'numpy.ndarray'> <class 'int'>
            print(x.shape, sr)
            #
            # plt.figure(figsize=(14, 5))
            librosa.display.waveplot(x, sr=sr)
            # plt.show()

            X = librosa.stft(x)
            Xdb = librosa.amplitude_to_db(abs(X))
            # plt.figure(figsize=(14, 5))
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
            # plt.colorbar()
            # plt.show()

            # sr = 22050 # частота дискретизации
            # T = 5.0    # секунды
            # t = np.linspace(0, T, int(T*sr), endpoint=False) # переменная времени
            # x = 0.5*np.sin(2*np.pi*220*t) # чистая синусоидная волна при 220 Гц
            # # сохранение аудио
            # sf.write('tone_220.wav', x, sr)

            #  1.Спектральный центроид
            # Указывает, на какой частоте сосредоточена энергия спектра или,
            # другими словами, указывает, где расположен «центр масс» для звука.
            # Схож со средневзвешенным значением произведения
            # где S(k) — спектральная величина элемента разрешения k, а f(k) — частота элемента k.

            spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
            print(spectral_centroids.shape)
            # (775,)
            # Вычисление временной переменной для визуализации
            # plt.figure(figsize=(12, 4))

            frames = range(len(spectral_centroids))
            t = librosa.frames_to_time(frames)

            # Нормализация спектрального центроида для визуализации
            def normalize(x, axis=0):
                return sklearn.preprocessing.minmax_scale(x, axis=axis)

            # Построение спектрального центроида вместе с формой волны
            librosa.display.waveplot(x, sr=sr, alpha=0.4)
            # plt.plot(t, normalize(spectral_centroids), color='b')
            # plt.show()

            # (94316,) 22050

            #  2.Спектральный спад
            #  Это мера формы сигнала, представляющая собой частоту,
            #  в которой высокие частоты снижаются до 0. Чтобы получить ее,
            #  нужно рассчитать долю элементов в спектре мощности, где 85%
            #  ее мощности находится на более низких частотах.

            spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
            # plt.figure(figsize=(12, 4))

            librosa.display.waveplot(x, sr=sr, alpha=0.4)
            # plt.plot(t, normalize(spectral_rolloff), color='r')
            # plt.show()

            #  3. Спектральная ширина
            # Спектральная ширина определяется как ширина полосы света
            # на половине максимальной точки (или полная ширина на половине
            # максимума [FWHM]) и представлена двумя вертикальными красными линиями
            # и λSB на оси длин волн.

            spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr)[0]
            spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=3)[0]
            spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=4)[0]
            # plt.figure(figsize=(15, 9))

            librosa.display.waveplot(x, sr=sr, alpha=0.4)
            # plt.plot(t, normalize(spectral_bandwidth_2), color='r')
            # plt.plot(t, normalize(spectral_bandwidth_3), color='g')
            # plt.plot(t, normalize(spectral_bandwidth_4), color='y')
            # plt.legend(('p = 2', 'p = 3', 'p = 4'))
            # plt.show()

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
            # plt.figure(figsize=(14, 5))
            # plt.plot(x[n0:n1])
            # plt.grid()
            # plt.show()
            zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
            print("zero crosing =", sum(zero_crossings))  # 16

            #  5. Мел-частотные кепстральные коэффициенты (MFCC)
            # Представляют собой небольшой набор признаков (обычно около 10–20),
            # которые кратко описывают общую форму спектральной огибающей.
            # Они моделируют характеристики человеческого голоса.
            mfccs = librosa.feature.mfcc(x, sr=sr)
            print(mfccs.shape)
            # Отображение MFCC:
            # plt.figure(figsize=(15, 7))
            librosa.display.specshow(mfccs, sr=sr, x_axis='time')
            # plt.show()

            #  6. Цветность
            # Признак или вектор цветности обычно представлен вектором признаков из 12 элементов,
            # в котором указано количество энергии каждого высотного класса {C, C#, D, D#, E, …, B}
            # в сигнале. Используется для описания меры сходства между музыкальными произведениями.
            chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=12)
            # plt.figure(figsize=(15, 5))
            librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=12, cmap='coolwarm')
            # plt.show()

            cmap = plt.get_cmap('inferno')
            # plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            # plt.axis('off');
            # plt.savefig(f'spectrogram Alex.png')
            # plt.clf()

            # for filename in os.listdir(f'./drive/My Drive/genres/{g}'):
            # songname = f'./drive/My Drive/genres/{g}/{filename}'
            # y, sr = librosa.load(songname, mono=True, duration=30)
            rmse = librosa.feature.rms(x)
            chroma_stft = librosa.feature.chroma_stft(x, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(x, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(x, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(x)
            mfcc = librosa.feature.mfcc(x, sr=sr)
            to_append = f'{f"{user}-({frame})"} {np.mean(chroma_stft)} {np.mean(rmse)} '
            to_append += f'{np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'

            with open(file_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

def logger(y_test, y_pr, y_test_info, matrix):
    for user in USERS:
        id = user_mapping[user]
        print(f"-------{id}. {user}:-------")
        TP = []
        FN = []
        FP = []
        for y, info, res in zip(y_test, y_test_info, y_pr):
            if y == id:
                if y == res:
                    TP.append((y, info, res))
                else:
                    FN.append((y, info, res))
            elif res == id:
                FP.append((y, info, res))
        total = len(TP) + len(FN)
        detected = len(TP) + len(FP)
        if total == 0:
            recall = 0
        else:
            recall = len(TP) / total
        if detected == 0:
            precision = 0
        else:
            precision = len(TP) / detected
        print(f"Тестовых данных: {total}\nОпознан: {detected}, Верно опознан: {len(TP)}")
        print(f"Recall: {((recall * 100)):.2f}%, precision: {(precision * 100):.2f}%")
        for elem in TP:
            print(elem[1])
        print(f"Не опознан (ошибка первого рода): {len(FN)}")
        for elem in FN:
            print(f"{elem[1]} - опознан {USERS[elem[2] - 1]} вместо {USERS[elem[0] - 1]}")
        print(f"Опознан как другой человек (ошибка второго рода): {len(FP)}")
        for elem in FP:
            print(f"{elem[1]} - опознан {USERS[elem[2] - 1]} вместо {USERS[elem[0] - 1]}")


def classification(func, name, X_train, X_test, y_train, y_test):
    model = func
    model.fit(X_train, y_train)
    y_pr = model.predict(X_test)
    print(y_test, y_pr)
    print(f"{name}, test:", end=" ")
    print(metrics.accuracy_score(y_test, y_pr))
    print(metrics.confusion_matrix(y_test, y_pr))
    print(f"{name}, train:", end=" ")
    pred = model.predict(X_train)
    print(metrics.accuracy_score(y_train, pred))
    print(metrics.confusion_matrix(y_train, pred))
    return y_pr, pred, metrics.confusion_matrix(y_test, y_pr)

if __name__ == "__main__":
    #write()
    #write(add_new=True, users_to_add=["TOLSTIKOV_GRIGORIY"])
    X = []
    y = []

    user_mapping = {u: i for u, i in zip(USERS, range(1, len(USERS) + 1))}
    print(user_mapping)

    with open(file_name, 'r') as file:
        lines = file.readlines()
        #print(lines)
        #print([len(line.split(',')) for line in lines])
        headers = lines[0].strip('\n').split(',')
        data = lines[1:]
        for line in data:
            arr = line.strip('\n').split(",")
            X.append(list(map(float, arr[1:])))
            y.append(arr[0])

    s_test = 0
    perfect_test = 0
    perfect_train = 0
    s_train = 0
    for _ in range(n_iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        y_train_info = [arg for arg in y_train]
        y_train = [user_mapping[arg[:arg.find('-')]] for arg in y_train]
        y_test_info = [arg for arg in y_test]
        y_test = [user_mapping[arg[:arg.find('-')]] for arg in y_test]

        # print(X_train, X_test, y_train, y_test)

        y_pr, pr, matrix = classification(GaussianNB(), "NAIVE BAYES", X_train, X_test, y_train, y_test)
        logger(y_test, y_pr, y_test_info, matrix)
        res_test = metrics.accuracy_score(y_test, y_pr)
        res_train = metrics.accuracy_score(y_train, pr)
        if res_test == 1.0:
            perfect_test += 1
        if res_train == 1.0:
            perfect_train += 1
        s_test += metrics.accuracy_score(y_test, y_pr)
        s_train += metrics.accuracy_score(y_train, pr)
    print("test accuracy mean:", s_test / n_iters)
    print("perfect results for tests data:", perfect_test)
    print("train accuracy mean:", s_train / n_iters)
    print("perfect results for train data:", perfect_train)
    # y_pr, matrix = classification(MLPClassifier(random_state=1, solver="adam", hidden_layer_sizes=(100, 100, 100), max_iter=100000),
    #                               "MLP1", X_train, X_test, y_train, y_test)
    # logger(y_test, y_pr, y_test_info, matrix)
    #
    # y_pr, matrix = classification(MLPClassifier(activation="tanh", solver="adam", hidden_layer_sizes=(1000, 1000, 1000, 1000),
    #                                             max_iter=100000), "MLP2", X_train, X_test, y_train, y_test)
    # logger(y_test, y_pr, y_test_info, matrix)
    #
    # y_pr, matrix = classification(MLPClassifier(activation="logistic", solver="lbfgs", hidden_layer_sizes=(100000),
    #                                             max_iter=1000000), "MLP2", X_train, X_test, y_train, y_test)
    # logger(y_test, y_pr, y_test_info, matrix)
    #
    # y_pr, matrix = classification(MLPClassifier(random_state=1,  activation="logistic", solver="lbfgs", hidden_layer_sizes=(100000),
    #                                             max_iter=1000000), "MLP2", X_train, X_test, y_train, y_test)
    # logger(y_test, y_pr, y_test_info, matrix)

    # y_pr, matrix = classification(tree.DecisionTreeClassifier(), "DECISION TREE", X_train, X_test, y_train, y_test)
    # logger(y_test, y_pr, y_test_info, matrix)
    #
    # y_pr, matrix = classification(KNeighborsClassifier(n_neighbors=3), "K NEIGHBOURS (3)", X_train, X_test, y_train, y_test)
    # logger(y_test, y_pr, y_test_info, matrix)


