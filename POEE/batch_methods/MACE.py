import numpy as np
from .nsga2_pareto_front import NSGA2_pygmo_MACE


def MACE(model, f_lb, f_ub, feval_budget, q, cf):
    X_front, musigma_front = NSGA2_pygmo_MACE(model, feval_budget,
                                              f_lb, f_ub, cf)

    Xnew = X_front[np.random.choice(X_front.shape[0], q), :]

    return Xnew
