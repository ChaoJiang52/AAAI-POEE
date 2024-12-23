# Trading off Quality and Uncertainty through Multi-Objective Optimisation in Batch Bayesian Optimisation

This repository contains the Python3 code for the proposed POEE method. The code of POEE can be found in the `POEE\batch_methods\multi_objectivisation.py` file.


## Reproduction of experiments

The python file `optimizer.py` provides a convenient way to reproduce all 
experimental evaluations carried out the paper. 

```bash
> python optimizer.py
```

## Training data

The initial training locations for each of the 30 sets of
[Latin hypercube](https://www.jstor.org/stable/1268522) samples are located in
the `training_data` directory in this repository with the filename structure
`ProblemName_number`, e.g. the first set of training locations for the Branin
problem is stored in `Branin_1.npz`. 
To load and inspect these values use the following instructions:

```python
>>> import numpy as np
>>> with np.load('training_data/Branin_1.npz') as data:
        Xtr = data['arr_0']
        Ytr = data['arr_1']
>>> Xtr.shape, Ytr.shape
((4, 2), (4, 1))
```

## Optimisation results

The results of all optimisation runs can be found in the `results` directory. The filenames have the following structure:
`ProblemName_Run_Batchsize_TotalBudget_Method.npz`.

The following example loads the first optimisation run on the Branin test
problem with the POEE method with weights = [0.4, 0.6] using a batch size of 5:

```python
>>> import numpy as np
>>> with np.load('results_paper/Branin_1_5_300_POEE_[0.4, 0.6].npz', allow_pickle=True) as data:
        Xtr = data['Xtr']
        Ytr = data['Ytr']
>>> Xtr.shape, Ytr.shape
((300, 2), (300, 1))
```

## Supplementary material

The supplementary material can be found in the `supplementary material` directory. 