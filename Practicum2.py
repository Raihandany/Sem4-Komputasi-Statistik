# MEAN
apel = [100, 200, 150, 100, 120, 80, 90, 160, 110, 170]

total_berat = 0

for i in range(len(apel)):
    total_berat = total_berat + apel[i]

avg_apel = total_berat / len(apel)
print(avg_apel)


#Media
import math

def bubblesort (elements):
    for n in range(len(elements)-1, 0, -1):
        for i in range(n):
            if elements[i] > elements[i + 1]:
                elements[i], elements[i + 1] = elements[i + 1], elements[i]
    return elements

def calc_median (elements):
    elements = bubblesort(elements)
    len_elements = len(elements)
    if len_elements % 2 is 0:
        center = math.floor(len_elements/2)
        return (elements[center-1] + elements[center])/2
    else:
        return elements[math.cell(len_elemenr/2)-1]

apel = [100, 200, 150, 100, 120, 80, 90, 160, 110, 170]
median = calc_median(apel)
print(median)


#Mode
from scipy import stats
import statistic as s

nilai = [70, 80, 70, 70, 90, 100]

value_mode = stats.mode(nilai)
modus = s.mode(nilai)
print(modus)
print(value_mode)
print(value_mode.mode)


# Interquartile_Range
import numpy as np
import pandas as pd

data_baru = np.array([23, 56, 45, 65, 59, 55, 62, 54, 85, 25, 55])
print (pd.DataFrame(data_baru).describe())


#Normalization 1
import numpy as np
import matplotlib.pyplot as plt

data_raw = np.array([
    [2, 3, 7, 30]
    [9, 4, 6, 1]
    [8, 15, 2, 40]
    [20, 10, 2, 6]
])


#Normalization 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data_raw = np.array({
    [2, 3, 7, 30]
    [9, 4, 6, 1]
    [8, 15, 2, 40]
    [20, 10, 2, 6]
})

scaler = MinMaxScaler()
data_scale = scaler.fit_transform(data_raw)


#Standardization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data_raw = np.array({
    [2, 3, 7, 30]
    [9, 4, 6, 1]
    [8, 15, 2, 40]
    [20, 10, 2, 6]
})

scaler = MinMaxScaler()
data_scale = scaler.fit_transform(data_raw)

plt.boxplot(data_stand)