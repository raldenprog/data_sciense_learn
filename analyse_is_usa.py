# Ссылка на колаб гугл, где более удобно выведены графики с анализом
# https://colab.research.google.com/drive/1mz8PkBP3kUSVDZ8DR1t6vAL-2wB4_Bmw?usp=sharing
# Первым блоком приведен только код реализации
# После него идут вызовы функций с комментариями - пояснениями, выводами
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

link = 'https://github.com/anastasiarazb/skillbox-data-analyst-intensive/blob/master/%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D0%94%D0%B5%D0%BD%D1%8C%202/IPG2211A2N.xls?raw=true'
data = pd.read_excel(link, skiprows=10)

link_infl = 'https://raw.githubusercontent.com/raldenprog/data_sciense_learn/master/infl.csv'
inflation = pd.read_csv(link_infl)

def show_data_last_10(data):
    data = data[data['observation_date'] >= pd.Timestamp(year=2010, month=1, day=1)]
    data = data.set_index('observation_date')
    plt.figure(figsize=(40, 5))  # размер графика
    plt.plot(data.index, data['IPG2211A2N'])
    plt.locator_params('x', nbins=100)
    plt.show()

def get_trend(data):
    N = len(data)  # Длина таблицы = количество строк
    x_range = list(range(N))  # создать список, который пересчитывает натуральные числа от 0 до N
    X = pd.DataFrame(x_range)  # создать табличку из одной колонки
    regressor = LinearRegression()
    regressor.fit(X, data['IPG2211A2N'])
    result = regressor.predict(X)

    plt.plot(x_range, data['IPG2211A2N'])
    plt.plot(x_range, result)
    plt.show()

def trends(data):
    get_trend(data)
    data = data[data['observation_date'] >= pd.Timestamp(year=2010, month=1, day=1)]
    get_trend(data)

def moving_average(data):
    data['IPG2211A2N'].rolling(12).mean()
    plt.figure(figsize=(20, 5))
    plt.plot(data.index, data['IPG2211A2N'], color='gray', label='Изначальные данные')
    plt.plot(data.index, data['IPG2211A2N'].rolling(12).mean(), color='blue', label='Среднее с окном год')
    plt.plot(data.index, data['IPG2211A2N'].rolling(120).mean(), color='red', label='Среднее с окном 10 лет')
    plt.legend()
    plt.show()

def analyse_last_10_years(data):
    data = data[data['observation_date'] >= pd.Timestamp(year=2010, month=1, day=1)]
    plt.figure(figsize=(30, 5))
    pd.plotting.autocorrelation_plot(data['IPG2211A2N'])
    plt.locator_params('x', nbins=150)

    plt.show()

def show_data_last_year(data):
    data = data[data['observation_date'] >= pd.Timestamp(year=2018, month=1, day=1)]
    # data = data.set_index('observation_date')
    plt.figure(figsize=(40, 5))  # размер графика
    plt.plot(pd.to_datetime(data['observation_date'], unit='s'), data['IPG2211A2N'])
    plt.locator_params('x', nbins=500)
    plt.show()

def analyse_last_20_years(data):
    data = data[data['observation_date'] >= pd.Timestamp(year=2000, month=1, day=1)]
    plt.figure(figsize=(30, 5))
    pd.plotting.autocorrelation_plot(data['IPG2211A2N'])
    plt.locator_params('x', nbins=150)

    plt.show()

def analyse_1_percents_quantile(data):
    """
    Отберем данные, которые не попали в коридор квантилей: 5% - 95%
    :param data:
    :return:
    """
    data = data[data['observation_date'] >= pd.Timestamp(year=2000, month=1, day=1)]
    quantile_95 = data['IPG2211A2N'].quantile(0.95)
    quantile_05 = data['IPG2211A2N'].quantile(0.05)

    data_95 = data[data['IPG2211A2N'] > quantile_95]
    data_05 =  data[data['IPG2211A2N'] < quantile_05]
    return data_95, data_05

def max_index_product(data):
    """
    Максимальный Индекс промышленного производства
    :return:
    """
    max_is = data['IPG2211A2N'].max()
    max_date = data[data['IPG2211A2N'] == max_is]['observation_date']
    return max_is, np.datetime_as_string(max_date.values[0], unit='D')

def diagram(data):
    data = data[(data['observation_date'] >= pd.Timestamp(year=2017, month=1, day=1)) &
                (data['observation_date'] <= pd.Timestamp(year=2020, month=12, day=31))]
    cat_par = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь',
               'Декабрь']
    g1 = [data[(data['observation_date'] == pd.Timestamp(year=2017, month=i, day=1))].values[0][1] for i in range(1, 13)]
    g2 = [data[(data['observation_date'] == pd.Timestamp(year=2018, month=i, day=1))].values[0][1] for i in range(1, 13)]
    g3 = [data[(data['observation_date'] == pd.Timestamp(year=2019, month=i, day=1))].values[0][1] for i in range(1, 13)]
    width = 0.3
    x = np.arange(len(cat_par))
    fig, ax = plt.subplots(figsize=(30, 5))
    rects1 = ax.bar(x - width, g1, width, label='2017')
    rects2 = ax.bar(x, g2, width, label='2018')
    rects3 = ax.bar(x + width, g3, width, label='2019')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_par)
    ax.legend()
    plt.show()

def min_index_product(data):
    """
    Максимальный Индекс промышленного производства
    :return:
    """
    min_is = data['IPG2211A2N'].min()
    min_date = data[data['IPG2211A2N'] == min_is]['observation_date']
    return min_is, np.datetime_as_string(min_date.values[0], unit='D')

def mean_last_10_years(data):
    """
    Средний Индекс промышленного производства за последние 10 лет
    :return:
    """
    data = data[data['observation_date'] >= pd.Timestamp(year=2010, month=1, day=1)]
    mean_is = data['IPG2211A2N'].mean()
    return mean_is

def show_inflation(data, inflation, date):
    inflation = inflation[inflation['Год'] >= date]
    data = data[(data['observation_date'].dt.month == 1) & (data['observation_date'] >= pd.Timestamp(year=date, month=1, day=1))]
    plt.figure(figsize=(40, 5))  # размер графика
    plt.plot(inflation['Год'], inflation[' Всего'])
    years = [d.year for d in data['observation_date']]
    plt.plot(years, data['IPG2211A2N'])
    plt.locator_params('x', nbins=70)
    plt.legend()

def ml(series):
    """
    Попытаемся спрогнозировать ИС следующих годов с помощью алгоритма постоянства
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
    X = dataframe.values
    train_size = int(len(X) * 0.66)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:,0], train[:,1]
    test_X, test_y = test[:,0], test[:,1]

    # Алгоритм постоянства
    def model_persistence(x):
        return x

    # составляем прогноз
    # из за алгоритма постоянства переобучение - не нужно
    predictions = list()
    for x in test_X:
        yhat = model_persistence(x)
        predictions.append(yhat)
    test_score = mean_squared_error(test_y, predictions)
    print('Test MSE: %.3f' % test_score)

    plt.figure(figsize=(30, 5))
    # рисуем график
    plt.plot(train_y)
    plt.plot([None for i in train_y] + [x for x in test_y])
    plt.plot([None for i in train_y] + [x for x in predictions])
    plt.show()
