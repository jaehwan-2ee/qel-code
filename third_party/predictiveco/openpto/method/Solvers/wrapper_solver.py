# Modified from PredictiveCO-Benchmark f885426f; the changes are listed in UPSTREAM.json.
from openpto.method.Solvers.cvxpy.cp_bmatching import BmatchingSolver
from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.method.Solvers.neural.BudgetallocSolver import budgetallocSolver


################################# Wrappers ################################################
def solver_wrapper(args, conf, problem):
    prob_solver_dict = {
        "budgetalloc": {"neural": budgetallocSolver},
        "bipartitematching": {"cvxpy": BmatchingSolver},
        "knapsack": {
            "gurobi": KPGrbSolver,
        },
    }
    solve_dict = {**problem.init_API(), **conf["solver"][args.solver]}
    return prob_solver_dict[args.problem][args.solver](**solve_dict)
