import numpy as np
from cmath import inf
from .imath import discriminativity

def Birnbaum2(tests):
    res = np.ndarray(np.shape(tests))
    ab, diff = _ability_difficulty(tests)
    discr = discriminativity(tests)
    for (i,j), val in np.ndenumerate(res):
        
        res[i,j] = _prob(ab[i], diff[j], twoPL, discr[j])
    return res
def rasch(tests):
    res = np.ndarray(np.shape(tests))
    ab, diff = _ability_difficulty(tests)
    for (i,j), val in np.ndenumerate(res):
        
        res[i,j] = _prob(ab[i], diff[j], onePL)
        
        res[i,j] = _prob(ab[i], diff[j], onePL)
    return res

def _prob(ab, dif, model_func, *extra_params):
    if np.abs(ab) == inf:
        return 1
    if np.abs(dif) == inf:
        return 0
    adj = ab - dif
    return model_func(adj, *extra_params)

# 1-parameter Rasch Model
def onePL(adj):
    return np.math.pow(np.math.e, adj) / (1 + np.math.pow(np.math.e, adj))

# 2-parameter Birnbaum model (1PL + discriminativity)
def twoPL(adj, discr):
    return np.math.pow(np.math.e, discr * adj) / (1 + np.math.pow(np.math.e, discr * adj))

# 3-parameter Birnbaum model (2PL + guessing parameter)
def threePL(adj, discr, guess):
    return guess + (1 - guess) * np.math.pow(np.math.e, discr * adj) / (1 + np.math.pow(np.math.e, discr * adj))

# 4-parameter Birnbaum model (3PL + probability of error)
def fourPL(adj, discr, guess, error_prob):
    return guess + (error_prob - guess) * np.math.pow(np.math.e, discr * adj) / (1 + np.math.pow(np.math.e, discr * adj))


def _logit(p):
    if p == 0:
            return 0
    if p == 1:
            return 1
    v = p/(1-p)
    return np.math.log(v)


def _ability_difficulty(tests, activation_function = _logit):
    max_t, max_u = np.shape(tests)
    sum_u, sum_t = [tests[i,:].sum() for i in range(max_t)], [tests[:,i].sum() for i in range(max_u)]
    # Chance of correct response
    p_u, p_t = [sum_u[i] / max_u for i in range(max_t)], [sum_t[i] / max_t for i in range(max_u)]
    # Function of chance
    d_u, d_t = [activation_function(p_u[i]) for i in range(max_t)], [activation_function(p_t[i]) for i in range(max_u)]
    return d_u, d_t

