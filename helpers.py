import numpy as np
from .imath import *



def rasch_to_coef(test_probabilities, max_score = 1):
    target_probability = 0.5
    test_probabilities = np.abs(test_probabilities - target_probability)
    test_probabilities = test_probabilities * (2 * max_score) - max_score
    test_probabilities *= -1
    return test_probabilities

def new_score(test_results):
    pass
