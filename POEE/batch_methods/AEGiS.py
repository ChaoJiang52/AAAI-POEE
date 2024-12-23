"""
Code adapted from: https://github.com/georgedeath/aegis
"""

import scipy
import numpy as np
from numpy.random import normal as numpy_normal
from GPy.util.linalg import jitchol
from .acquisition_optimisers import aquisition_CMAES, minimise_with_CMAES
from .acquisition_optimisers import aquisition_LBFGSB, minimise_with_LBFGSB
from .nsga2_pareto_front import NSGA2_pygmo
from .sequential_functions import mu


def gp_sample(model, x, n_samples):
    if len(x.shape) == 1:
        x = np.reshape(x, (1, -1))
        n_points = 1
    else:
        n_points = x.shape[0]

    # special case if we're only have 1 realisation of 1 point
    if n_points == 1 and n_samples == 1:
        m, cov = model.predict(x, full_cov=False)
        L = np.sqrt(cov)
        U = numpy_normal()
        return m + L * U

    # else general case, do things properly
    m, cov = model.predict(x, full_cov=True)
    L = jitchol(cov)
    U = numpy_normal(size=(n_points, n_samples))
    return m + L @ U


def AEGiS(model, f_lb, f_ub, feval_budget, q, cf):
    n_dim = f_lb.size
    epsilon = np.min([2 / np.sqrt(n_dim), 1])
    epsilon_T = epsilon / 2
    epsilon_P = epsilon / 2

    feval_budget = feval_budget // q
    Xnew = []
    got_cf = cf is not None
    for i in range(q):
        r = np.random.uniform()
        if r < 1 - (epsilon_T + epsilon_P):

            # we can only use CMA-ES on 2 or more dimensional functions
            if n_dim > 1:
                opt_acq_func, opt_caller = aquisition_CMAES, minimise_with_CMAES

            # else use L-BFGS-B
            else:
                opt_acq_func, opt_caller = aquisition_LBFGSB, minimise_with_LBFGSB

            fopt = opt_acq_func(model, mu, cf)
            # run optimiser a max of budget/q evaluations of the gp to
            # select a new point to expensively evaluate
            xj = opt_caller(fopt, f_lb, f_ub, int(feval_budget), cf=cf)

        elif r < 1 - epsilon_P:
            X = np.random.uniform(low=f_lb, high=f_ub, size=(feval_budget, f_lb.size))
            res = np.zeros(feval_budget)
            for i, x in enumerate(X):
                if got_cf and (not cf(x)):
                    res[i] = np.inf
                else:
                    res[i] = gp_sample(model, x, n_samples=1)
            best_idx = np.argmin(res)
            xj = X[best_idx, :]

        else:
            # calculate the pareto front and randomly select a location on it
            X_front, musigma_front = NSGA2_pygmo(model, feval_budget,
                                                 f_lb, f_ub, cf)
            xj = X_front[np.random.choice(X_front.shape[0]), :]

        Xnew.append(xj)

    Xnew = np.array(Xnew)

    return Xnew
