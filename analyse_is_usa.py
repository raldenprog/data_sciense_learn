"""
Модуль содержит методы с анализом индекса промышленного производства с точки зрения data science.

Ссылка на колаб гугл, где более удобно выведены графики с анализом
https://colab.research.google.com/drive/1mz8PkBP3kUSVDZ8DR1t6vAL-2wB4_Bmw?usp=sharing
Первым блоком приведен только код реализации
После него идут вызовы функций с комментариями - пояснениями, выводами
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

LINK = 'https://github.com/anastasiarazb/skillbox-data-analyst-intensive/blob/master/%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D0%94%D0%B5%D0%BD%D1%8C%202/IPG2211A2N.xls?raw=true'
EXCEL_DATA = pd.read_excel(LINK, skiprows=10)

LINK_INFLATION = 'https://raw.githubusercontent.com/raldenprog/data_sciense_learn/master/infl.csv'
INFLATION = pd.read_csv(LINK_INFLATION)
NAME_DATE = 'observation_date'
NAME_ID_FIELD = 'IPG2211A2N'


def show_data_last_10(excel_data):
    """
    Метод отображает данные за последние 10 лет.

    :param excel_data: Данные из excel
    """
    excel_data = excel_data[excel_data[NAME_DATE] >= pd.Timestamp(year=2010, month=1, day=1)]
    excel_data = excel_data.set_index(NAME_DATE)
    plt.figure(figsize=(40, 5))  # размер графика
    plt.plot(excel_data.index, excel_data[NAME_ID_FIELD])
    plt.locator_params('x', nbins=100)
    plt.show()


def get_trend(excel_data):
    """
    Метод строит тренд по переданным данным в excel.

    :param excel_data: Данные из excel
    """
    len_excel_data = len(excel_data)  # Длина таблицы = количество строк
    x_range = list(range(len_excel_data))  # создать список, который пересчитывает натуральные числа от 0 до N
    x_line = pd.DataFrame(x_range)  # создать табличку из одной колонки
    regressor = LinearRegression()
    regressor.fit(x_line, excel_data[NAME_ID_FIELD])
    result = regressor.predict(x_line)

    plt.plot(x_range, excel_data[NAME_ID_FIELD])
    plt.plot(x_range, result)
    plt.show()


def trends(data):
    """
    Строит график тренда с 2010 года.

    :param data: excel данные
    :type data: excel
    """
    get_trend(data)
    data = data[data[NAME_DATE] >= pd.Timestamp(year=2010, month=1, day=1)]
    get_trend(data)


def moving_average(data):
    """
    График отображения скользящей средней с окнами: год, 10 лет.

    :param data: excel данные
    :type data: excel
    """
    data[NAME_ID_FIELD].rolling(12).mean()
    plt.figure(figsize=(20, 5))
    plt.plot(data.index, data[NAME_ID_FIELD], color='gray', label='Изначальные данные')
    plt.plot(data.index, data[NAME_ID_FIELD].rolling(12).mean(), color='blue', label='Среднее с окном год')
    plt.plot(data.index, data[NAME_ID_FIELD].rolling(120).mean(), color='red', label='Среднее с окном 10 лет')
    plt.legend()
    plt.show()


def analyse_last_10_years(data):
    """
    График автокорреляции для анализа последних 10 лет.
    :param data: excel данные
    :type data: excel
    """
    data = data[data[NAME_DATE] >= pd.Timestamp(year=2010, month=1, day=1)]
    plt.figure(figsize=(30, 5))
    pd.plotting.autocorrelation_plot(data[NAME_ID_FIELD])
    plt.locator_params('x', nbins=150)

    plt.show()


def show_data_last_year(data):
    """
    Грифк с данными за последний год
    :param data:
    :type data:
    :return:
    :rtype:
    """
    data = data[data[NAME_DATE] >= pd.Timestamp(year=2018, month=1, day=1)]
    plt.figure(figsize=(40, 5))  # размер графика
    plt.plot(pd.to_datetime(data[NAME_DATE], unit='s'), data[NAME_ID_FIELD])
    plt.locator_params('x', nbins=500)
    plt.show()


def analyse_last_20_years(data):
    """
    График автокорреляции для анализа последних 20 лет.
    :param data: excel данные
    :type data: excel
    """
    data = data[data[NAME_DATE] >= pd.Timestamp(year=2000, month=1, day=1)]
    plt.figure(figsize=(30, 5))
    pd.plotting.autocorrelation_plot(data[NAME_ID_FIELD])
    plt.locator_params('x', nbins=150)

    plt.show()


def analyse_1_percents_quantile(data):
    """
    Отберем данные, которые не попали в коридор квантилей: 5% - 95%ю

    :param data: excel данные
    :return:
    """
    data = data[data[NAME_DATE] >= pd.Timestamp(year=2000, month=1, day=1)]
    quantile_95 = data[NAME_ID_FIELD].quantile(0.95)
    quantile_05 = data[NAME_ID_FIELD].quantile(0.05)

    data_95 = data[data[NAME_ID_FIELD] > quantile_95]
    data_05 = data[data[NAME_ID_FIELD] < quantile_05]
    return data_95, data_05


def diagram(data):
    """
    Метод строит диаграмму по данным из excel по месяцам за 2017-2019 год.

    :param data: Excel данные
    :type data: excel
    """
    data = data[(data[NAME_DATE] >= pd.Timestamp(year=2017, month=1, day=1)) &
                (data[NAME_DATE] <= pd.Timestamp(year=2020, month=12, day=31))]
    cat_par = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь',
               'Декабрь']
    g1 = [data[(data[NAME_DATE] == pd.Timestamp(year=2017, month=i, day=1))].values[0][1] for i in range(1, 13)]
    g2 = [data[(data[NAME_DATE] == pd.Timestamp(year=2018, month=i, day=1))].values[0][1] for i in range(1, 13)]
    g3 = [data[(data[NAME_DATE] == pd.Timestamp(year=2019, month=i, day=1))].values[0][1] for i in range(1, 13)]
    width = 0.3
    x = np.arange(len(cat_par))
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.bar(x - width, g1, width, label='2017')
    ax.bar(x, g2, width, label='2018')
    ax.bar(x + width, g3, width, label='2019')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_par)
    ax.legend()
    plt.show()


def min_index_product(data):
    """
    Максимальный Индекс промышленного производства.

    :param data: Данные excel
    :type data: excel
    :return: Максимальный Индекс промышленного производства
    :rtype: float
    """
    min_is = data[NAME_ID_FIELD].min()
    min_date = data[data[NAME_ID_FIELD] == min_is][NAME_DATE]
    return min_is, np.datetime_as_string(min_date.values[0], unit='D')


def mean_last_10_years(data):
    """
    Средний Индекс промышленного производства за последние 10 лет.

    :param data: Данные excel
    :type data: excel
    :return: Средний Индекс промышленного производства
    :rtype: float
    """
    data = data[data[NAME_DATE] >= pd.Timestamp(year=2010, month=1, day=1)]
    mean_is = data[NAME_ID_FIELD].mean()
    return mean_is


def show_inflation(data, inflation, date):
    """
    Метод отображает график инфляции.

    :param data: excel_data
    :type data: excel
    :param inflation: excel_inflation
    :type inflation: excel
    :param date: Дата построения
    :type date: datetime
    """
    nbins = 70
    figsize = (40, 5)
    inflation = inflation[inflation['Год'] >= date]
    data = data[(data[NAME_DATE].dt.month == 1) & (data[NAME_DATE] >= pd.Timestamp(year=date, month=1, day=1))]
    plt.figure(figsize=figsize)  # размер графика
    plt.plot(inflation['Год'], inflation[' Всего'])
    years = [d.year for d in data[NAME_DATE]]
    plt.plot(years, data[NAME_ID_FIELD])
    plt.locator_params('x', nbins=nbins)
    plt.legend()


def model_persistence(analyse_value):
    """

    Метод реализация алгоритма постоянства.

    :param analyse_value: Анализируемое значение
    :return: Анализируемое значение
    """
    return analyse_value


def ml(series):
    """
    Попытаемся спрогнозировать ИС следующих годов с помощью алгоритма постоянства.

    Взято с https://www.machinelearningmastery.ru/persistence-time-series-forecasting-with-python/
    :param series:
    :return:
    """
    # в момент времени t-1, можно предсказать наблюдение в момент времени t + 1.
    values = pd.DataFrame(series.values)
    del values[0]  # удаляем даты
    dataframe = pd.concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't+1']

    # Оставим первые 66% наблюдений для «обучения» и оставшиеся 34% для оценки
    percent_view = 0.66
    x_line = dataframe.values
    train_size = int(len(x_line) * percent_view)
    train, test = x_line[1:train_size], x_line[train_size:]
    train_x_line, train_y_line = train[:, 0], train[:, 1]
    test_x_line, test_y_line = test[:, 0], test[:, 1]

    # составляем прогноз
    # из за алгоритма постоянства переобучение - не нужно
    predictions = []
    for x_line in test_x_line:
        predict = model_persistence(x_line)
        predictions.append(predict)

    plt.figure(figsize=(30, 5))
    # рисуем график
    plt.plot(train_y_line)
    plt.plot([None for _ in train_y_line] + [test for test in test_y_line])
    plt.plot([None for _ in train_y_line] + [predict for predict in predictions])
    plt.show()
