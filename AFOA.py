import numpy as np
import time


def AFOA(X, obj_func, lb, ub, max_iter):
    start_time = time.time()
    pop_size, dim = X.shape
    fitness_values = np.zeros(pop_size)

    # Initial fitness calculation
    for i in range(pop_size):
        fitness_values[i] = obj_func(X[i, :])

    f_best = np.min(fitness_values)
    best_idx = np.argmin(fitness_values)
    X_best = X[best_idx, :]
    Convergence_curve = np.zeros(max_iter)

    for t in range(max_iter):
        # Calculate S and Q for current iteration
        S = 2 * np.sin((3 * t / (2 * max_iter)) * np.pi)
        Q = np.cos(-((max_iter - t) / max_iter) * np.pi / 2)

        # Update population based on S and Q
        if np.abs(S) < 0.5:
            A = np.cos(np.pi * np.random.rand(X.shape[0])) * (np.random.rand(X.shape[0], X.shape[1]) * 10 - 5)
            X = X + A * ((ub - lb) * (2 * np.random.rand(X.shape[0], X.shape[1]) - 1))
            X = np.clip(X, lb, ub)
        else:
            r = np.random.rand()
            if Q >= 0.6:
                R4 = 2 * np.random.rand(X.shape[0], X.shape[1])
                R5 = 2 * np.random.rand(X.shape[0], X.shape[1])
                X = 2 * (X_best - X) * R4 + np.exp(R5 * Q ** 5) * np.cos(Q * 2 * np.pi)
            else:
                if r >= 0.6:
                    R6 = -0.01 + 0.02 * np.random.rand(X.shape[0], X.shape[1])
                    R7 = 2 * np.random.rand(X.shape[0], X.shape[1])
                    J = (X_best - X) * R6
                    X = X_best - J * R7
                elif r > 0.1 and r < 0.6:
                    MM = np.cos(fitness_values / (f_best + 1e-8))  # Avoid div by zero
                    R8 = np.random.rand(X.shape[0], X.shape[1])
                    MM = np.tile(MM, (X.shape[1], 1)).T
                    X = X + np.sin(2 * np.pi * R8) * R8 * MM * (X - X_best)
                else:
                    fitness_vals = np.zeros(X.shape[0])
                    for i in range(X.shape[0]):
                        fitness_vals[i] = obj_func(X[i, :])
                    worst_idx = np.argmax(fitness_vals)
                    X = lb + np.random.rand(1, X.shape[1]) * (ub - lb)

        # Clip boundaries
        X = np.clip(X, lb, ub)

        # Evaluate fitness
        for i in range(pop_size):
            fitness_values[i] = obj_func(X[i, :])

        # Update best solution found so far
        current_best = np.min(fitness_values)
        best_idx = np.argmin(fitness_values)

        if current_best < f_best:
            X_best = X[best_idx, :]
            f_best = current_best

        # Track the convergence
        Convergence_curve[t] = f_best

    exec_time = time.time() - start_time
    return f_best, Convergence_curve, X_best, exec_time
