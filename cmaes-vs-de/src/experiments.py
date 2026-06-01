import numpy as np
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from src.callbacks import History


def run_many_runs(n, algo_fn, n_runs=10, seed=1):

    all_curves = []

    for i in range(n_runs):

        problem = get_problem("rastrigin", n_var=n)

        cb = History()
        algo = algo_fn(n)

        minimize(
            problem,
            algo,
            termination=("n_eval", 2000),
            callback=cb,
            seed=seed + i,
            verbose=False
        )

        all_curves.append(cb.data["best"])

    return all_curves