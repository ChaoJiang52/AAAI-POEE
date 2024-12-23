"""This script generates all training data for the optimisation runs of each
test problem evaluated in the paper.
"""
from POEE.util import generate_training_data as gtd
# from .util import generate_training_data as gtd

# double-check the user wants to overwrite any existing training data
value = input(
    'Enter "Y" to confirm you wish to generate all training data'
    + " (any existing training data will be overwritten): "
)
if value not in ["Y", "y"]:
    import sys

    sys.exit(0)

# parameters matching the paper - 51 total experiments per optimisation method
start_exp_no = 1
end_exp_no = 31
N_exps = end_exp_no - start_exp_no + 1

# synthetic test problems - LHS 2 * D samples
problem_names = ['WangFreitas', 'BraninForrester', 'Branin', 'EGGHOLDER', 'GoldsteinPrice', 'SixHumpCamel',
                 'Hartmann6', 'ACKLEY_2', 'GRIEWANK_2', 'ACKLEY_10', 'GRIEWANK_10', 'GSobol']

for name in problem_names:
    gtd.generate_training_data_LHS(name, start_exp_no, end_exp_no)

# robot pushing problems - LHS sample with generated arguments for each problem
T1_x, T1_y = gtd.generate_push4_targets(N_exps)
push4_targets = {"t1_x": T1_x, "t1_y": T1_y}
gtd.generate_training_data_LHS(
    "push4", start_exp_no, end_exp_no, optional_arguments=push4_targets
)

T1_x, T1_y, T2_x, T2_y = gtd.generate_push8_targets(N_exps)
push8_targets = {"t1_x": T1_x, "t1_y": T1_y, "t2_x": T2_x, "t2_y": T2_y}
gtd.generate_training_data_LHS(
    "push8", start_exp_no, end_exp_no, optional_arguments=push8_targets
)
