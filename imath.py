import numpy as np
from . import irt

def task_coef(test_results):
    rasch = irt.Birnbaum2(test_results)
    coef = np.ones((test_results.shape[1])) / 2
    coef += prob_to_score(np.mean(rasch, axis=0))
    coef += cluster_analysis_coef(test_results)
    return test_results * (coef / 3)

def discriminativity(test_result: np.ndarray, score_pass = 1):
    scores = np.sum(test_result, axis=1)
    scores_std = np.std(scores)
    scores_mean = np.mean(scores)
    shape = np.shape(test_result)
    tasks_discr = np.ndarray(shape[1])
    for task_ind in range(shape[1]):
        task_results = test_result[:, task_ind]
        task_score_max = np.max(task_results)
        task_score_pass = task_score_max * score_pass
        scores_pass = [scores[i] for i in range(shape[0]) if test_result[i, task_ind] >= task_score_pass]
        scores_pass_mean = np.mean(scores_pass)
        task_pass_count = len(scores_pass)
        if shape[0] == task_pass_count:
            discr = 0
        else:
            discr = ((scores_pass_mean - scores_mean)/scores_std) * np.sqrt(task_pass_count/(shape[0] - task_pass_count))
        tasks_discr[task_ind] = discr
    return tasks_discr

def tasks_identical(test_result):
    test_result_shape = np.shape(test_result)
    subjects_count, tasks_count = test_result_shape
    dublicates = np.zeros((tasks_count, tasks_count))
    for i in range(tasks_count):
        for j in range(tasks_count):
            dublicates[i,j] = len([1 for ir, jr in zip(test_result[:, i], test_result[:, j]) if ir == jr])
    return dublicates / subjects_count
def tasks_dublicates(test_result):
    corr = np.nan_to_num(np.abs(np.corrcoef(test_result.T)))
    dubl = tasks_identical(test_result)
    r = (corr + dubl) / 2
    return r

def activation_diff(x):
    target = 0.5
    return ((np.abs(target - x)*(2)) - 1) * -1
def activation_invert(x):
    return (x - 1) * -1
def activation_none(x):
     return x
def prob_to_score(prob, max_score = 1, activation = activation_invert):
    score = np.vectorize(activation)(prob) * max_score
    return score


from sklearn.cluster import KMeans
def cluster_analysis_coef(test_result, n = 3, activation = activation_none):
    # clusterization
    cluster_m = KMeans(n_clusters=n).fit((np.sum(test_result, axis=1)/test_result.shape[1]).reshape(1, -1).T)
    clusters = np.array([test_result[np.where(cluster_m.labels_ == i)] for i in range(n)], dtype=object)
    clusters_sorted_inds = np.argsort(cluster_m.cluster_centers_, axis=0)[:,0]
    clusters = clusters[clusters_sorted_inds]
    # процент правильного ответа на задание для каждого кластера
    tasks_percent = np.zeros((n, test_result.shape[1]))
    for i in range(n):
        tasks_percent[i] = np.sum(clusters[i], axis=0)/clusters[i].shape[0]
    # отклонение между кластерами
    clusters_std = np.std(tasks_percent, axis=0)
    # активация
    coef = np.vectorize(activation)(clusters_std)
    return coef