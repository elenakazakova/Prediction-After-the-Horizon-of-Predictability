pip install yfinance
import pandas as pd
import os
import re
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import groupby
from collections import Counter
import random

total_path = os.getcwd()
os.chdir(total_path)
df = pd.read_excel('United_tickers.xlsx')
df

total_path = os.getcwd()
total_path

error_list = list()
r = os.listdir()
if not ('Data' in r):
    os.mkdir("Data")

os.chdir(total_path + '\\Data')
for i in list(df['Ticker'].dropna()):
    data = yf.download(i, interval="1h", period="2y")
    print(i + '.xlsx')
    if len(data) > 0:
        try:
            data = yf.download(i, interval="1d")
            data['Date'] = data.index.copy()
            data = data.reset_index(drop=True)
            data['Date'] = data['Date'].dt.tz_localize(None)

            df1 = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

            df1.dropna().to_excel(i + '.xlsx', header=True, index=False)
            print('Сохранил тикер ' + str(i))
        except:
            error_list.append(i)
            print('Не удалось сохранить тикер ' + str(i))
            continue

###### скрипт производит первичный отбор данных:
# 1. Сперва проверяется, что минимальная длина ряда больше min_len.
# 2. Затем проверяется, что приращения ряда имеют разный знак.
###### Если ряд проходит по обоим условиям, он сохраняется в папку "Data (прошли отбор по длине и приращениям)". Результат отбора (ряд прошел или не прошел по одному из условий) фиксируется в чек-листе "Чек-лист селекта по длине и приращениям".

adr = input("введите путь от папки со скриптом до папки Data: ")
# adr = '\\Тест'

## min_len = input("введите минимальную допустимую длину: ")
min_len = 365

total_path = os.getcwd()
total_path

current_path = total_path + adr
os.chdir(current_path)

r = os.listdir()
if not ('Data (прошли отбор по длине и приращениям)' in r):
    os.mkdir("Data (прошли отбор по длине и приращениям)")
    medium_flag = False
else:

    save_path = total_path + adr + '\\Data (прошли отбор по длине и приращениям)'
    os.chdir(save_path)
    lst_searched_data = os.listdir()
    medium_flag = True

current_path = total_path + adr + '\\Data (Все)'
save_path = total_path + adr + '\\Data (прошли отбор по длине и приращениям)'

os.chdir(current_path)

dir_list = os.listdir()  ## список всех файлов в папке
dir_list

случайно
удалил
кусок
кода.проверь
перед
новым
запуском

os.chdir(save_path)

f = open('Чек-лист селекта по длине и приращениям (l = ' + str(min_len) + ').txt', 'w')

for i in dir_list:
    print('______________________________________________________')
    print('Читаю ' + str(i))
    df = pd.read_excel(current_path + '\\' + str(i))
    if len(df) < min_len:  # Проверяем длину
        print(str(i) + ' слишком короткий')
        f.write(str(i) + ' слишком короткий' + '\n')
        continue

    else:
        diff_list = pd.DataFrame(df['Close']).diff(periods=1).dropna()
        diff_list = list(diff_list['Close'])
        if all(b >= 0 for b in diff_list) or all(b <= 0 for b in diff_list):
            # если все приращения неотрицательные или неположительные, не берем
            print(str(i) + ' стабильно растет/падает')
            f.write(str(i) + ' стабильно растет/падает' + '\n')
            continue
        else:
            df[['Date', 'Adj Close']].dropna().to_excel(i, header=True, index=False)
            f.write('Сохранил ' + str(i) + '\n')
            print('Сохранил ' + str(i))

f.close()

##### скрипт строит статистику по датасету и создает в подпапке "Тестовая подвыборка(по длине)" тестовую подвыборку.

total_path = os.getcwd()
total_path

current_path = total_path
os.chdir(current_path)

r = os.listdir()
if not ('Тестовая подвыборка(по длине)' in r):
    os.mkdir("Тестовая подвыборка(по длине)")

current_path = total_path + '\\Data (длина и приращения)'
save_path = total_path + '\\Тестовая подвыборка(по длине)'

os.chdir(current_path)

current_path

dir_list = os.listdir()  ## список всех файлов в папке
dir_list

len_lst = list()
for i in dir_list:
    os.chdir(current_path + '\\' + i)
    files = os.listdir()
    for j in files:
        df = pd.read_excel(j)
        l = len(df)
        print(j)
        len_lst.append(l)

ticker_lst = list()
for i in dir_list:
    os.chdir(current_path + '\\' + i)
    files = os.listdir()
    for i in files:
        ticker_lst.append(i)

summa = len(ticker_lst)
summa

os.chdir(total_path)
df = pd.DataFrame({'Тикер': ticker_lst, 'Длина': len_lst})
df.dropna().to_excel('Длина_тикеров.xlsx', header=True, index=False)

os.chdir(total_path)
df = pd.read_excel('Длина_тикеров.xlsx')
len_lst = list(df['Длина'])

ground_lst = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000,
              16000, 17000, 18000, 19000, 20000]

len_lst_2 = list()

for i in len_lst:
    for j in ground_lst:
        if i <= j:
            len_lst_2.append(j)
            break

stat_dict = {}
for i in set(len_lst_2):
    sum_of_el = 0
    for j in len_lst_2:
        if j == i:
            sum_of_el = sum_of_el + 1
    stat_dict[i] = sum_of_el

list_keys = list(stat_dict.keys())
list_keys.sort()
list_keys

index = list()
values = list()
new_dict = {}
for i in list_keys:
    new_dict[i] = stat_dict[i]
    index.append(int(i))
    values.append(int(stat_dict[i]))
stat_dict = new_dict

plt.figure(figsize=(15, 6))
values = list(stat_dict.values())
keys = list(stat_dict.keys())
plt.bar(range(len(stat_dict)), values, tick_label=keys)
plt.xlabel('Длина ряда')
plt.ylabel('Число образцов')
plt.show()

k = 0
dict_2 = {}
for i in stat_dict:
    a = stat_dict.get(i)
    d = a / summa
    if np.round(d * 100) == 0:
        dict_2[i] = 1.0
    else:
        dict_2[i] = np.round(d * 100)
    k = k + np.round(d * 100)
    # print(dict_1.get(i))

dict_2

el_list = []

res_list = list()
for i in dict_2:
    num_elements = int(dict_2.get(i))  # получаем число элементов, которое выбираем
    el_list = list()
    for k in range(0, len(len_lst_2)):

        if len_lst_2[k] == i:
            el_list.append(ticker_lst[k])
    for j in range(0, num_elements):
        if (len(el_list) - 1 == 0):
            number = 0
        else:
            number = random.randint(0, len(el_list) - 1)
        element = el_list[number]
        res_list.append(element)

len(res_list)

os.chdir(save_path)

for i in res_list:
    for j in dir_list:
        os.chdir(current_path + '\\' + j)
        files = os.listdir()
        if i in files:
            df = pd.read_excel(i)
            os.chdir(save_path)
            df.dropna().to_csv(i[:-4] + 'csv', header=True, index=False)

#### Построение тестовой подвыборки

import pandas as pd
import requests
import os
import re
# import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy.signal import argrelextrema
import math
import shutil


def moving_average(a, n):
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

n = 200  # число прошлых шагов для ME
num_points = 300  # число точек, которые отступаем от экстремума для удаления шума


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
        plt.title(file[:-4])
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
    save_path_for_tseries = save_path + '\\Data'
    os.chdir(save_path)
    ls_dir = os.listdir()
    if "Data" not in ls_dir:
        os.mkdir("Data")
    os.chdir(current_path)

    for i in files:

        df = pd.read_csv(current_path + '\\' + i)
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
            names = [i[:-4] for k in range(0, len(list_exterem))]
            res = [names, idx, list_exterem, list_date]
            res = np.array(res).T
            res = pd.DataFrame(res, columns=['Ticker', 'index', 'exteremum', 'Date'])
            res_df = res_df.append(res)
            text = [i[:-4], 'успешно прошел отбор, ' + str(len(idx)) + ' точек.']
            text = pd.DataFrame([text], columns=['Ticker', 'Результат'])
            check_df = check_df.append(text)
            shutil.copy(current_path + '\\' + i, save_path_for_tseries)
        else:
            text = [i[:-4], 'не прошел отбор, есть ступенчатость экстремумов.']
            text = pd.DataFrame([text], columns=['Ticker', 'Результат'])
            check_df = check_df.append(text)

    os.chdir(save_path)
    res_df = res_df.reset_index(drop=True)
    res_df.to_excel("output_extremus_subset.xlsx", index=False)
    check_df = check_df.reset_index(drop=True)
    check_df.to_excel("check_list_subset.xlsx", index=False)
    os.chdir(current_path)


current_path = total_path + '\\Тестовая подвыборка(по длине)'
save_path = total_path + '\\Тестовая подвыборка(длина и приращения, ступенчатость)'

os.chdir(total_path)
r = os.listdir()
if not ('Тестовая подвыборка(длина и приращения, ступенчатость)' in r):
    os.mkdir("Тестовая подвыборка(длина и приращения, ступенчатость)")

os.chdir(current_path)
dir_list = os.listdir()  ## список всех файлов в папке
len(dir_list)

for i in dir_list:  # строим графики последовательно.
    file = i
    cacl_bif_multi(n, num_points, file, plot_flag=True, ME_flag=False)

#скрипт производит первичный отбор данных:
#Сперва проверяется, что минимальная длина ряда больше min_len.
#Затем проверяется, что приращения ряда имеют разный знак.

adr = input("введите путь от папки со скриптом до папки Data: ")
#adr = '\\Тест'
## min_len = input("введите минимальную допустимую длину: ")
min_len = 365
total_path = os.getcwd()
total_path
current_path = total_path + adr
os.chdir(current_path)

r = os.listdir()
if not('Data (прошли отбор по длине и приращениям)' in r):
    os.mkdir("Data (прошли отбор по длине и приращениям)")
    medium_flag = False
else:

    save_path = total_path + adr + '\\Data (прошли отбор по длине и приращениям)'
    os.chdir(save_path)
    lst_searched_data = os.listdir()
    medium_flag = True

current_path = total_path + adr + '\\Data (Все)'
save_path = total_path + adr + '\\Data (прошли отбор по длине и приращениям)'

os.chdir(current_path)

dir_list = os.listdir() ## список всех файлов в папке
dir_list

os.chdir(save_path)

f = open('Чек-лист селекта по длине и приращениям (l = '+str(min_len)+').txt','w')

for i in dir_list:
    print('______________________________________________________')
    print('Читаю ' + str(i))
    df = pd.read_excel(current_path + '\\' + str(i))
    if len(df) < min_len:  # Проверяем длину
        print(str(i) + ' слишком короткий')
        f.write(str(i) + ' слишком короткий' + '\n')
        continue

    else:
        diff_list = pd.DataFrame(df['Close']).diff(periods=1).dropna()
        diff_list = list(diff_list['Close'])
        if all(b >= 0 for b in diff_list) or all(b <= 0 for b in diff_list):
            # если все приращения неотрицательные или неположительные, не берем
            print(str(i) + ' стабильно растет/падает')
            f.write(str(i) + ' стабильно растет/падает' + '\n')
            continue
        else:
            df[['Date', 'Adj Close']].dropna().to_excel(i, header=True, index=False)
            f.write('Сохранил ' + str(i) + '\n')
            print('Сохранил ' + str(i))

f.close()

df = pd.read_excel('0P0000WFXG.TO.xlsx')
df.columns

print('______________________________________________________')
print('Читаю')
if len(df) < min_len:  # Проверяем длину
    print('слишком короткий')

else:
    diff_list = pd.DataFrame(df['Close']).diff(periods=1).dropna()
    diff_list = list(diff_list['Close'])
    if all(b >= 0 for b in diff_list) or all(b <= 0 for b in diff_list):
        # если все приращения неотрицательные или неположительные, не берем
        print('стабильно растет/падает')
    else:
        print('Все ок
