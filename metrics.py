import numpy as np
import pandas as pd
from .helpers import rasch_to_coef
from .irt import rasch
from pingouin import cronbach_alpha as p_cronbach_alpha

def metrics(test_results):
    arr = {
        # 'Delta Ferguson': delta_ferguson(test_results),
        # 'Average Rasch': rasch_metric(test_results)
        'Дельта Фергюсона': delta_ferguson(test_results),
        'Среднее отклонение Раша': rasch_metric(test_results),
        'KR 20': kr_20(test_results),
        "Альфа Кронбаха": cronbach_alpha(test_results)
    }
    return arr

def delta_ferguson(test_result: np.ndarray):
    shape = np.shape(test_result)
    N = shape[0]
    n = shape[1]
    test_result_score = np.sum(test_result, axis=1)
    unique, counts = np.unique(test_result_score, return_counts=True)
    counts = np.sum(np.power(counts, 2))
    return (n+1)*(N*N - counts)/(n*N*N)

def rasch_metric(test_result):
    return np.mean(rasch_to_coef(rasch(test_result)))

#Метод Кьюдера—Ричардсона KR-20
def kr_20(test):
    M = len(test[0])#количество тестовых заданий
    Xi = [sum(x) for x in test]#сумма баллов у каждого студента
    X = sum(Xi)/M#среднее выборочное(арифметическое)
    sx2 = sum((x-X)**2 for x in Xi) / (M-1)#дисперсия
    p = [x/M for x in Xi]#процент правильно ответивших на тестовое задание
    r = M / (M - 1) * (1 - (sum(x * (1-x) for x in p)) / sx2)
    return r

def cronbach_alpha(test_results):
    return p_cronbach_alpha(pd.DataFrame(test_results))[0]
    