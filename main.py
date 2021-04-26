import csv
import warnings

import librosa
import librosa.display
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
