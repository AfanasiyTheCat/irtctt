import numpy as np

def tasks_dublicates(test_result):
    test_result_shape = np.shape(test_result)
    subjects_count, tasks_count = test_result_shape
    dublicates = np.ndarray((tasks_count, tasks_count))
    for i in range(subjects_count):
        for j in range(subjects_count):
            dublicates[i,j] = len([1 for i, j in zip(test_result[:, i], test_result[:, j]) if i == j])
    return dublicates / subjects_count
