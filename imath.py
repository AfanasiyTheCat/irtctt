import numpy as np

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

def cluster_analysis(test_result):
    def get_clusters(n):
        pass
    pass