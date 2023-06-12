import numpy as np
import math
import matplotlib.pyplot as plt
import pylab as pl
import itertools as it
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

import requests
import os
import re
#import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import argrelextrema
import math
import shutil

from Wishart import Wishart


k = 2
h = 1000
type_of_inst = 'Stock'

l_min = 4
l_max = 4  # работает до этого значения включая его. Чтобы было одно значение, ставь одинаковый l_min и max
l_step = 1

delta_min = 0
delta_max = 5 # работает до этого значения включая его. Чтобы было одно значение, ставь одинаковый l_min и max
delta_step = 1

params = list()

if l_min != l_max:
    l_list = np.arange(l_min, l_max + 1, l_step)
else:
    l_list = [l_max]

if delta_min != delta_max:
    delta_list = np.arange(delta_min, delta_max + 1, delta_step)
else:
    delta_list = [delta_max]

for l in l_list:
    for delta in delta_list:
        params.append([delta, l])

for i in params:
    model = Wishart(k, h)
    ext_name = 'delta=' + str(i[0]) + ',l=' + str(i[1])
    if type_of_inst != 'All':
        ext_name = '(delta=' + str(i[0]) + ',l=' + str(i[1]) + ')(' + str(type_of_inst) + ')'

    df = pd.read_csv('train ' + ext_name + '.csv')
    data_columns = df.columns[3:]
    df = df[data_columns]

    labels = model.fit(df)
    res = pd.DataFrame(data=[np.arange(0, len(labels)), labels])
    res = res.T
    res = res.rename(columns={0: 'Num_of_el', 1: 'Num_of_cluster'})
    res.to_csv('results delta=' + str(i[0]) + ',l=' + str(i[1]) + '(k=' + str(k) + ',h=' + str(h) + ').csv',
               index=False)

#подсчет точек смены тренда и проверка ряда на ступенчатость

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def search_min(a):
    el_min = a[0]
    for elm in a[1:]:
        if elm < el_min:
            el_min = elm
    return el_min

def search_max(a):
    el_max = a[0]
    for elm in a[1:]:
        if elm > el_max:
            el_max = elm
    return el_max

total_path = os.getcwd()
total_path


def find_extremums(me, num_points):
    # функция ищет точки смены тренда для конкретного временного ряда file. Принимает:
    # me - ряд.
    # n - число прошлых шагов для ME.
    # num_points - число точек, которые отступаем от экстремума для удаления шума
    # Возвращает: индексы точек и значения ряда в них.
    left = me[0]
    right = me[-1]

    idx_minimas = argrelextrema(me, np.less_equal, order=num_points)[0]
    idx_maximas = argrelextrema(me, np.greater_equal, order=num_points)[0]
    idx = list(np.sort(np.concatenate((idx_minimas, idx_maximas))))
    list_exterem = list(me[idx])

    if left in list_exterem:
        idx_l = list_exterem.index(left)
        list_exterem.remove(left)
        del idx[idx_l]

    if right in list_exterem:
        idx_r = list_exterem.index(right)
        list_exterem.remove(right)
        del idx[idx_r]

    return {'extrem_index': idx, 'extrem_value': list_exterem}


def cacl_bif_multi(n, num_points, file, plot_flag=True, ME_flag=False):
    df = pd.read_csv(current_path + '\\' + file)

    me_1 = np.array(df['Adj Close'])
    res = find_extremums(me_1, num_points)
    idx_1 = res['extrem_index']
    list_exterem_1 = res['extrem_value']

    if ME_flag == True:
        me_2 = moving_average(list(df['Adj Close']), n)
        res = find_extremums(me_2, num_points)
        idx_2 = res['extrem_index']
        list_exterem_2 = res['extrem_value']

    if plot_flag == True:
        plt.figure(figsize=(10, 6))
        plt.plot(me_1, color='r', label='Исходный ряд', linewidth=1)
        plt.scatter(idx_1, list_exterem_1, color='black', linewidth=3)
        plt.title(file)
        if ME_flag == True:
            plt.plot(range(n, len(me_2) + n), me_2, color='b', label='ME', linewidth=2)
            plt.scatter(list(np.array(idx_2) + n), list_exterem_2, color='b', linewidth=3)

        plt.legend()


def clear_from_stairs_and_cacl_bif(files, num_points, save_path, current_path, total_path):
    # функция проверяет ступенчатость экстремумов, ищет экстремумы, сохраняет экстремумы ряда в файл.
    # Принимает список рядов files, путь для сохранения результирующего датасета save_path, путь где лежит список рядов
    # current_path.
    res_df = pd.DataFrame(columns=['Ticker', 'index', 'exteremum'])  # датафрейм с результатами
    check_df = pd.DataFrame(columns=['Ticker', 'Результат'])  # чеклист
    save_path_for_tseries = save_path
    os.chdir(save_path)
    ls_dir = os.listdir()
    os.chdir(current_path)

    for i in files:

        print(files.index(i))
        df = pd.read_excel(current_path + '\\' + i)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.replace(['Infinity', '-Infinity'], np.nan, inplace=True)
        df.dropna()

        me = np.array(df['Adj Close'])
        # находим экстремумы и их индексы. Затем находим даты экстремумов.
        res = find_extremums(me, num_points)
        idx = res['extrem_index']
        list_exterem = res['extrem_value']
        list_date = list(df['Date'].loc[idx])

        stair_falg = False
        if len(list_exterem) > 1:  ## проверка на ступенчатость
            for j in range(0, len(list_exterem) - 1):
                if list_exterem[j] == list_exterem[j + 1]:
                    stair_falg = True

        if stair_falg == False:
            names = [i[:-5] for k in range(0, len(list_exterem))]
            res = [names, idx, list_exterem, list_date]
            res = np.array(res).T
            res = pd.DataFrame(res, columns=['Ticker', 'index', 'exteremum', 'Date'])
            res_df = res_df.append(res)
            text = [i[:-5], 'успешно прошел отбор, ' + str(len(idx)) + ' точек.']
            text = pd.DataFrame([text], columns=['Ticker', 'Результат'])
            check_df = check_df.append(text)
            shutil.copy(current_path + '\\' + i, save_path_for_tseries)
        else:
            text = [i[:-5], 'не прошел отбор, есть ступенчатость экстремумов.']
            text = pd.DataFrame([text], columns=['Ticker', 'Результат'])
            check_df = check_df.append(text)

    os.chdir(save_path[:-7])
    res_df = res_df.reset_index(drop=True)
    res_df.to_excel('output_extremus_part_' + str(save_path[-1]) + '.xlsx', index=False)
    check_df = check_df.reset_index(drop=True)
    check_df.to_excel('check_list_part_' + str(save_path[-1]) + '.xlsx', index=False)
    os.chdir(current_path)

num_of_part = 2

total_path

current_path= total_path  + '\\Data (длина и приращения)\\Part_'+str(num_of_part)
save_path = total_path + '\\Data (длина и приращения, ступенчатость)\\Part_'+str(num_of_part)
save_path = total_path + '\\Data (длина и приращения, ступенчатость)\\Part_'+str(num_of_part)
os.chdir(total_path)
r = os.listdir()
if not('Data (длина и приращения, ступенчатость)' in r):
    os.mkdir("Data (длина и приращения, ступенчатость)")

os.chdir(total_path+'\\Data (длина и приращения, ступенчатость)')
r = os.listdir()
if not('Part_'+str(num_of_part) in r):
    os.mkdir('Part_'+str(num_of_part))
os.chdir(current_path)
dir_list = os.listdir() ## список всех файлов в папке
os.chdir(total_path)
len(dir_list)
