from irtctt.irt import rasch
from scipy.stats import zscore
from scipy.stats import chisquare
from psython import cronbach_alpha_scale_if_deleted
import numpy as np
import pandas as pd
from . import imath

def metrics(test_results, task_weights = None):
    if task_weights is None:
      task_weights = np.ones((test_results.shape[1]))
    max_score = np.sum(task_weights)
    item_total_corr, alpha_cronbach = item_total_alpha_cronbach(test_results)
    chisquare, p = chisquare_p_metric(test_results)
    arr = {
        'Дельта Фергюсона': delta_ferguson(test_results, max_score),
        'Среднее отклонение Раша': rasch_metric(test_results),
        'KR 20': kr_20(test_results),
        "Item Total Correlation": item_total_corr,
        "Альфа Кронбаха": alpha_cronbach,
        'Z-показатель': zscore_metric(test_results),
        'Хи-квадрат': chisquare,
        'P-показатель': p,
    }
    return arr

def delta_ferguson(test_result: np.ndarray, max_score = None):
    shape = np.shape(test_result)
    if not max_score:
      max_score = shape[1]
    N = shape[0]
    n = max_score
    test_result_score = np.round(np.sum(test_result, axis=1) / max_score * 100)
    unique, counts = np.unique(test_result_score, return_counts=True)
    counts = np.sum(np.power(counts, 2))
    return (n+1)*(N*N - counts)/(n*N*N)

def rasch_metric(test_result):
    return np.mean(imath.prob_to_score(rasch(test_result)))

#Метод Кьюдера—Ричардсона KR-20
def kr_20(test):
    M = len(test[0])#количество тестовых заданий
    Xi = [sum(x) for x in test]#сумма баллов у каждого студента
    X = sum(Xi)/M#среднее выборочное(арифметическое)
    sx2 = sum((x-X)**2 for x in Xi) / (M-1)#дисперсия
    p = [x/M for x in Xi]#процент правильно ответивших на тестовое задание
    r = M / (M - 1) * (1 - (sum(x * (1-x) for x in p)) / sx2)
    return r

def zscore_metric(test_result):
  return np.abs(np.mean(np.nan_to_num(zscore(test_result, ddof=0, nan_policy='propagate'))))

def item_total_alpha_cronbach(test_results):
  d = cronbach_alpha_scale_if_deleted(pd.DataFrame(test_results))[1]
  return np.mean(np.nan_to_num(d['Corrected Item-Total Correlation'].values)), np.mean(d["Cronbach's Alpha if Item Deleted"].values)

def chisquare_p_metric(test_result):
  d = chisquare(test_result)
  chisquare_v = np.mean(np.nan_to_num(d[0])) / 100
  return chisquare_v, np.mean(np.nan_to_num(d[1]))
