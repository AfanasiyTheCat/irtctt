from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import pingouin as pg

#Метод Кьюдера—Ричардсона KR-20
def kr_20(test):
    M = len(test[0])#количество тестовых заданий
    Xi = [sum(x) for x in test]#сумма баллов у каждого студента
    X = sum(Xi)/M#среднее выборочное(арифметическое)
    sx2 = sum((x-X)**2 for x in Xi) / (M-1)#дисперсия
    p = [x/M for x in Xi]#процент правильно ответивших на тестовое задание
    r = M / (M - 1) * (1 - (sum(x * (1-x) for x in p)) / sx2)
    return r

#correlation of results
def pear_spear_corr(data: np.ndarray, threshold: float):
  temp_cor_pear = [[1 for j in range(len(data))] for x in range(len(data))]
  temp_cor_spear = [[1 for j in range(len(data))] for x in range(len(data))]
  pearson_res, spearman_res = [], []
  for i in range(len(data)):
    for j in range(i+1, len(data)):
      temp_pearson = pearsonr(data[i], data[j])
      temp_spearman = spearmanr(data[i], data[j])
      if temp_pearson[0] > threshold:
        pearson_res.append((i, j, temp_pearson[0]))
      if temp_spearman[0] > threshold:
        spearman_res.append((i, j, temp_spearman[0]))
  return pearson_res, spearman_res

