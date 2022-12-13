from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import pingouin as pg

#Метод Кьюдера—Ричардсона KR-20
def kr_20(test):
    M = len(test)#количество испытуемых
    Xi = [sum(x) for x in test]#сумма баллов у каждого студента
    X = sum(Xi)/M#среднее выборочное(арифметическое)
    sx2 = sum((x-X)**2 for x in Xi) / (M-1)#дисперсия
    sx2 = round(sx2, 2)
    p = np.sum(test, axis = 0) / M#процент выполнения задания
    temp_chislitel = round(sum(x * (1-x) for x in p), 1)
    r = M * (1 - (temp_chislitel / sx2)) / (M - 1)
    print(M, M-1, temp_chislitel, sx2)
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

def get_bisserial_coef(question_number: int, data: np.ndarray):
  x1, x0 = [], []
  #print(data.shape)
  for i in range(len(data)):
    #print(i)
    if data[i][question_number] == 1:
      x1.append(sum(data[i]))
    else:
      x0.append(sum(data[i]))
  n1 = len(x1)
  n0 = len(x0)
  n = n1 + n0
  x1 = np.mean(x1)
  x0 = np.mean(x0)
  Xi = [sum(x) for x in data]#сумма баллов у каждого студента
  X = sum(Xi)/len(Xi)#среднее выборочное(арифметическое)
  temp1 = n*sum([xi**2 for xi in Xi])
  temp2 = sum([xi for xi in Xi])**2
  sx = np.sqrt((temp1 - temp2)/(n*(n-1)))#
  temp1 = (x1-x0)/sx
  temp2 = np.sqrt(n1*n0/(n*(n-1)))
  res = ((x1-x0) / sx) * np.sqrt((n1*n0) / (n*(n-1)))
  return res

def get_correlation_matrix(data: np.ndarray):
  p = np.sum(data, axis = 0) / len(data)
  q = 1 - p
  #print(data)
  cor_matrix = np.ones((data.shape[1], data.shape[1]))
  for i in range(len(cor_matrix)):
    for j in range(len(cor_matrix[i])):
      if(i != j):
        pq = sum([1 for row in data if row[i] == row[j] == 1]) / len(data)
        cor_matrix[i][j] = (pq - p[i]*p[j]) / (np.sqrt(p[i]*q[i]*p[j]*q[j]))

  bisserial_coefs = [get_bisserial_coef(i, data) for i in range(len(data[0]))]
  return cor_matrix, np.array(bisserial_coefs)

#длина вектора может быть меньше количества заданий, так задания, на которые ответили верно ВСЕ испытуемые, исключаются из теста
def get_correction_factor(correlation_matrix: np.ndarray):#поправочные коэффициенты, исходя из корреляции между заданиями
  res = []
  ids = []
  for i in range(len(correlation_matrix)):
    temp = []
    for j in range(len(correlation_matrix[i])):
      if i != j:
        if np.isnan(correlation_matrix[i][j]) == False:
          temp.append(1 - np.abs(correlation_matrix[i][j]))
    if temp != []:
      res.append(np.mean(temp))
    else:
      res.append(0)
  return res