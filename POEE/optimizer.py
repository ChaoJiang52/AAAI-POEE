import os
import numpy as np
import GPy as gp
import torch
from torch.optim import LBFGS
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.utils.errors import NotPSDError
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.optim import optimize_acqf
from copy import deepcopy
import gpytorch.settings as settings
from POEE import test_problems, batch_methods


def build_and_fit_GP(Xtr, Ytr):
    # create a gp model with the training data and fit it
    kernel = gp.kern.Matern52(input_dim=Xtr.shape[1], ARD=False)
    model = gp.models.GPRegression(Xtr, Ytr, kernel, normalizer=True)

    model.constrain_positive('')
    (kern_variance, kern_lengthscale,
     gaussian_noise) = model.parameter_names()

    model[kern_variance].constrain_bounded(1e-6, 1e6, warning=False)
    model[kern_lengthscale].constrain_bounded(1e-6, 1e6, warning=False)
    model[gaussian_noise].constrain_fixed(1e-6, warning=False)

    model.optimize_restarts(optimizer='lbfgs',
                            num_restarts=10,
                            num_processes=1,
                            verbose=False)

    return model


def build_and_fit_GP_torch(train_X, train_Y):
    restarts = 10
    best_model = None
    best_loss = float('inf')

    for restart in range(restarts):
        print(f"Restart {restart + 1}/{restarts}")

        # Define the GP model
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=MaternKernel(nu=2.5),
            outcome_transform=Standardize(m=1),
        )

        # Manually initialize hyperparameters to avoid extreme values
        model.covar_module.lengthscale = torch.tensor([1e-1]).double()
        model.likelihood.noise_covar.noise = torch.tensor([1e-1]).double()

        # Define the Marginal Log Likelihood
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        model.train()

        # Define the LBFGS optimizer
        optimizer = LBFGS([{"params": model.parameters()}], lr=0.1, max_iter=200)

        # Define the closure function
        def closure():
            optimizer.zero_grad()  # Clear previous gradients
            output = model(train_X)  # Forward pass
            loss = -mll(output, model.train_targets)  # Compute the negative marginal log likelihood
            loss.backward()  # backprop gradients
            return loss

        try:
            # Increase jitter setting during model training
            with settings.cholesky_jitter(1e-2):
                optimizer.step(closure)
            # optimizer.step(closure)
        except NotPSDError as e:
            print("Non-positive definite matrix encountered. Skipping this restart.")
            continue

        # Check the final loss and update the best model if this one is better
        model.eval()
        with torch.no_grad():
            final_output = model(train_X)
            final_loss = -mll(final_output, model.train_targets).item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_model = model

    if best_loss == np.inf:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit_gpytorch_mll by default uses L-BFGS
        fit_gpytorch_mll(mll)
        return model
    else:
        return best_model  # Return the best model after all restarts


def optimize(problem_name, run_no, batch_method, batch_method_args,
             batch_size, budget, noise=True, overwrite_existing=False):
    # filename to save to and path to training data
    save_file = f'../results/{problem_name:}_{run_no:}'
    if noise:
        save_file += f'_noise'
    save_file += f'_{batch_size:d}_{budget:}_{batch_method:}'
    for _, val in batch_method_args.items():
        save_file += f'_{val:}'
    save_file += '.npz'

    data_file = f'../training_data/{problem_name:}_{run_no:}.npz'

    # check to see if the save file exists
    if os.path.exists(save_file):
        if not overwrite_existing:
            print('Save file already exists:', save_file)
            print('Set overwrite_existing to True to overwrite the run.')
            return

    # load the function's additional arguments, if there are any
    with np.load(data_file, allow_pickle=True) as data:
        if 'arr_2' in data:
            f_arguments = data['arr_2'].item()
        else:
            f_arguments = {}
    # get the test problem class and instantiate it
    f_class = getattr(test_problems, problem_name)
    f = f_class(**f_arguments)

    # map it to reside in [0, 1]^d
    f = test_problems.util.uniform_problem_wrapper(f)

    # problem characteristics
    f_lb, f_ub, f_dim, f_cf = f.lb, f.ub, f.dim, f.cf

    # load the training data - it resides in full (not unit) space
    with np.load(data_file, allow_pickle=True) as data:
        Xtr = data['arr_0']
        Ytr = data['arr_1']

        # map it down to unit space
        Xtr = (Xtr - f.real_lb) / (f.real_ub - f.real_lb)

    n_train = Ytr.size

    # try to resume the run
    if os.path.exists(save_file):
        with np.load(save_file, allow_pickle=True) as data:
            Xcontinue = data['Xtr']
            Ycontinue = data['Ytr']

        # check if it has finished
        if Ycontinue.size >= n_train + budget:
            print('Run already finished:', save_file)
            return

        # if not, do some sanity checking
        n_already_completed = Ycontinue.size - n_train

        if n_already_completed % batch_size != 0:
            print('Completed batches do not match batch size:', save_file)
            print('Number of training data:', n_train)
            print('Saved evaluations size:', Ycontinue.size)
            print(n_already_completed, 'is not completely divisible by',
                  batch_size)
            return

        # we're safe to resume the run
        Xtr = Xcontinue
        Ytr = Ycontinue
        print('Resuming the run from:', save_file)
        print('Xtr shape:', Xtr.shape)

    # GP budget
    feval_budget = 10000 * f_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while Xtr.shape[0] < budget + n_train:

        if batch_method in ["qEI_bt", "GIBBON"]:
            train_Y = torch.from_numpy(-Ytr).to(device)
            if noise:
                train_Y = train_Y + 0.1 * torch.randn_like(train_Y)

            gp = build_and_fit_GP_torch(torch.from_numpy(Xtr).to(device), train_Y)

            if batch_method == "qEI_bt":
                acf = qExpectedImprovement(model=gp, best_f=(-Ytr).max())
                isSequential = False
            elif batch_method == "GIBBON":
                candidate_set_size = 1000
                candidate_set = torch.rand(candidate_set_size, f_dim, dtype=torch.double, device=device)
                acf = qLowerBoundMaxValueEntropy(gp, candidate_set)
                isSequential = True

            bounds = torch.stack([torch.from_numpy(f.lb), torch.from_numpy(f.ub)]).to(torch.double).to(device)

            # number of optimisation runs and *estimated* number of L-BFGS-B
            # function evaluations per run; note this was calculate empirically and
            # may not be true for all functions.
            N_opt_runs = 10
            fevals_assumed_per_run = 100

            # (10000D - 1000) samples
            samples = feval_budget - (N_opt_runs * fevals_assumed_per_run)

            # optimize_acqf by default uses L-BFGS
            Xnew, acq_value = optimize_acqf(
                acf, bounds=bounds, q=batch_size, num_restarts=N_opt_runs, raw_samples=samples, sequential=isSequential
            )
            print(Xnew)
            Xnew = Xnew.cpu().detach().numpy()
        else:
            # get the batch method class
            batch_f = getattr(batch_methods, batch_method)
            Y_train = deepcopy(Ytr)
            if noise:
                Y_train += 0.1 * np.random.randn(*Y_train.shape)
            model = build_and_fit_GP(Xtr, Y_train)
            Xnew = batch_f(model, f_lb, f_ub, feval_budget,
                       batch_size, f_cf, **batch_method_args)

        Ynew = np.zeros((batch_size, 1))
        for i in range(batch_size):
            # try to evaluate the solutions
            while True:
                try:
                    Ynew[i] = f(Xnew[i, :])
                    break

                # if this fails, try to generate a new one - note that this
                # will only occur on the PitzDaily test problem as there are
                # times when the CFD mesh fails to converge.
                except:
                    while True:
                        Xnew[i, :] = np.random.uniform(f_lb, f_ub)
                        if (f_cf is None) or f_cf(Xnew[i, :]):
                            break

        Xtr = np.concatenate((Xtr, np.atleast_2d(Xnew)))
        Ytr = np.concatenate((Ytr, Ynew))

        print(Xnew, Ynew)

        batch_no = int((Xtr.shape[0] - n_train) / batch_size)

        s = 'P {} AF {} Batch {: >3d}: fmin -> {:g}'.format(problem_name, batch_method, batch_no, np.min(Ytr))
        print(save_file)
        print(s)

        # save results
        np.savez(save_file, Xtr=Xtr, Ytr=Ytr, budget=budget,
                 batch_method=batch_method,
                 batch_method_args=batch_method_args,
                 batch_size=batch_size)


if __name__ == "__main__":
    budget = 300  # exclusive of training data
    overwrite_existing = True

    method_names = ["POEE", "TS", "qEI_bt", "hallu", "eShotgun", "PLAyBOOK", "LP", "MACE", "AEGiS", "GIBBON"
                    "POEE_unchanged_PF", "POEE_changed_PF", "POEE_changed_e_PF", "POEE_without_e"]

    problem_names = ["Ackley_2", "Ackley_10", "Branin", "BraninForrester", "Eggholder", "GoldsteinPrice", "GRIEWANK_2",
                     "GRIEWANK_10", "GSobol", "Hartmann6", "SixHumpCamel", "WangFreitas", "push4", "push8"]

    for batch_method in method_names:
        if batch_method in ["POEE", "POEE_without_e"]:
            batch_method_args = {'weights': [0.4, 0.6]}
        elif batch_method == "eShotgun":
            batch_method_args = {'epsilon': 0.1, 'pf': True}
        else:
            batch_method_args = {}
        for problem_name in problem_names:
            for run_no in range(1, 31):
                for batch_size in [5, 10, 20]:
                    optimize(problem_name, run_no, batch_method, batch_method_args,
                             batch_size, budget, noise=False, overwrite_existing=overwrite_existing)
