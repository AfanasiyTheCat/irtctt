import numpy as np
from cmath import inf

def rasch(tests):
    res = np.ndarray(np.shape(tests))
    ab, diff = _ability_difficulty(tests)
    for (i,j), val in np.ndenumerate(res):
        res[i,j] = onePL(ab[i], diff[j])
    return res

def onePL(ab, dif):
    if ab == inf:
        return 1
    if dif == inf:
        return 0
    adj = ab - dif
    return np.math.pow(np.math.e, adj) / (1 + np.math.pow(np.math.e, adj))

def _ability_difficulty(tests):
    max_t, max_u = np.shape(tests)
    sum_u, sum_t = [tests[i,:].sum() for i in range(max_t)], [tests[:,i].sum() for i in range(max_u)]
    p_u, p_t = [sum_u[i] / max_u for i in range(max_t)], [sum_t[i] / max_t for i in range(max_u)]
    d_u, d_t = [_logit(p_u[i]) for i in range(max_t)], [_logit(p_t[i]) for i in range(max_u)]
    return d_u, d_t

def _logit(p):
    if p == 0:
            return -inf
    v = p/(1-p)
    return np.math.log(v)