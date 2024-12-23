from topsis import topsis
from .nsga2_pareto_front import NSGA2_pygmo
import re
import pygmo as pg
from POEE.test_problems.synthetic_problems import *


# POEE$_\mathrm{r\_PF}$
def POEE_unchanged_PF(model, f_lb, f_ub, feval_budget, q, cf):
    pop_X, pop_Y = NSGA2_pygmo(model, feval_budget, f_lb, f_ub, cf)

    # indices non-dominated points across the entire NSGA-II run
    front_inds = pg.non_dominated_front_2d(pop_Y)

    X_front = pop_X[front_inds, :]

    Xnew = X_front[np.random.choice(X_front.shape[0], q), :]

    return Xnew


# POEE$_\mathrm{r\_uPF}$
def POEE_changed_PF(model, f_lb, f_ub, feval_budget, q, cf):
    pop_X, pop_Y = NSGA2_pygmo(model, feval_budget, f_lb, f_ub, cf)

    # indices non-dominated points across the entire NSGA-II run
    front_inds = pg.non_dominated_front_2d(pop_Y)

    X_front = pop_X[front_inds, :]

    Xnew = []
    for i in range(q):
        _x = X_front[np.random.choice(X_front.shape[0]), :]
        Xnew.append(_x)

        mu_best_x, _ = model.predict(np.atleast_2d(_x), full_cov=False)

        Xtr = np.concatenate((model.X, np.atleast_2d(_x)))
        Ytr = np.concatenate((model.Y, np.atleast_2d(mu_best_x)))
        model.set_XY(Xtr, Ytr)
        mu, sigma = model.predict(pop_X)
        # indices non-dominated points across the entire NSGA-II run
        front_inds = pg.non_dominated_front_2d(points=np.concatenate((mu, -sigma), axis=1))
        X_front = pop_X[front_inds, :]

    return np.array(Xnew)


# POEE$_\mathrm{no\_TOPSIS}$
def POEE_changed_e_PF(model, f_lb, f_ub, feval_budget, q, cf):
    pop_X, pop_Y = NSGA2_pygmo(model, feval_budget, f_lb, f_ub, cf)

    # indices non-dominated points across the entire NSGA-II run
    front_inds = pg.non_dominated_front_2d(pop_Y)

    X_front = pop_X[front_inds, :]

    Xnew = []
    for i in range(q):
        if i == 0:
            # best mu, cannot pick the same points
            _x = X_front[0]
        else:
            _x = X_front[np.random.choice(X_front.shape[0]), :]

        Xnew.append(_x)

        mu_best_x, _ = model.predict(np.atleast_2d(_x), full_cov=False)

        Xtr = np.concatenate((model.X, np.atleast_2d(_x)))
        Ytr = np.concatenate((model.Y, np.atleast_2d(mu_best_x)))
        model.set_XY(Xtr, Ytr)
        mu, sigma = model.predict(pop_X)
        # indices non-dominated points across the entire NSGA-II run
        front_inds = pg.non_dominated_front_2d(points=np.concatenate((mu, -sigma), axis=1))
        X_front = pop_X[front_inds, :]

    return np.array(Xnew)


# POEE$_\mathrm{no\_exploit}$
def POEE_without_e(model, f_lb, f_ub, feval_budget, q, cf, weights):
    pop_X, pop_Y = NSGA2_pygmo(model, feval_budget, f_lb, f_ub, cf)

    # indices non-dominated points across the entire NSGA-II run
    front_inds = pg.non_dominated_front_2d(pop_Y)

    X_front = pop_X[front_inds, :]
    musigma_front = pop_Y[front_inds, :]

    Xnew = []
    for i in range(q):
        f1 = musigma_front[:, 0]
        f2 = -musigma_front[:, 1]
        data = np.stack((f1, f2), axis=-1)
        # topsis first w for mu, second for sigma
        w = weights
        I = [0, 1]  # cost (0) benefit (1)
        decision = topsis(data, w, I)
        topsis_idx = int(re.findall(r'\d+', str(decision))[0])
        best_x = X_front[topsis_idx]
        Xnew.append(best_x)

        mu_best_x, _ = model.predict(np.atleast_2d(best_x), full_cov=False)

        Xtr = np.concatenate((model.X, np.atleast_2d(best_x)))
        Ytr = np.concatenate((model.Y, np.atleast_2d(mu_best_x)))
        model.set_XY(Xtr, Ytr)
        mu, sigma = model.predict(pop_X)
        # indices non-dominated points across the entire NSGA-II run
        front_inds = pg.non_dominated_front_2d(points=np.concatenate((mu, -sigma), axis=1))
        X_front = pop_X[front_inds, :]
        musigma_front = pop_Y[front_inds, :]

    return np.array(Xnew)


# POEE
def POEE(model, f_lb, f_ub, feval_budget, q, cf, weights):
    pop_X, pop_Y = NSGA2_pygmo(model, feval_budget, f_lb, f_ub, cf)

    # indices non-dominated points across the entire NSGA-II run
    front_inds = pg.non_dominated_front_2d(pop_Y)

    X_front = pop_X[front_inds, :]
    musigma_front = pop_Y[front_inds, :]

    Xnew = []
    for i in range(q):
        f1 = musigma_front[:, 0]
        f2 = -musigma_front[:, 1]
        data = np.stack((f1, f2), axis=-1)

        if i == 0:
            # best mu
            best_x = X_front[0]
            Xnew.append(best_x)
            print(best_x, f1.min())
        else:
            # topsis first w for mu, second for sigma
            w = weights
            I = [0, 1]  # cost (0) benefit (1)
            decision = topsis(data, w, I)
            topsis_idx = int(re.findall(r'\d+', str(decision))[0])
            best_x = X_front[topsis_idx]
            Xnew.append(best_x)

        mu_best_x, _ = model.predict(np.atleast_2d(best_x), full_cov=False)

        Xtr = np.concatenate((model.X, np.atleast_2d(best_x)))
        Ytr = np.concatenate((model.Y, np.atleast_2d(mu_best_x)))
        model.set_XY(Xtr, Ytr)
        mu, sigma = model.predict(pop_X)
        # indices non-dominated points across the entire NSGA-II run
        front_inds = pg.non_dominated_front_2d(points=np.concatenate((mu, -sigma), axis=1))
        X_front = pop_X[front_inds, :]
        musigma_front = pop_Y[front_inds, :]

    return np.array(Xnew)
