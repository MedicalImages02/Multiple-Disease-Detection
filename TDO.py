import time
import numpy as np


# Tasmanian Devil Optimization (TDO)
def TDO(tasmanian_devils, fobj, lb, ub, Max_iter):
    num_tasmanian_devils, num_variables = tasmanian_devils.shape[0], tasmanian_devils.shape[1]
    best_solution = None
    best_fitness = float('inf')
    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    # Main optimization loop
    for iteration in range(Max_iter):
        # Evaluate the fitness of each tasmanian devil
        fitness_values = fobj(tasmanian_devils[:])

        # Update the best solution found so far
        min_fitness_index = np.argmin(fitness_values)
        if fitness_values[min_fitness_index] < best_fitness:
            best_fitness = fitness_values[min_fitness_index]
            best_solution = tasmanian_devils[min_fitness_index]

        # Perform movement of tasmanian devils
        for i in range(num_tasmanian_devils):
            # Generate a random index different from i
            j = np.random.randint(0, num_tasmanian_devils)
            while j == i:
                j = np.random.randint(0, num_tasmanian_devils)

            # Perform the movement
            delta = np.random.uniform(-1, 1, num_variables)
            tasmanian_devils[i] += delta * (tasmanian_devils[i] - tasmanian_devils[j])

            # Bound the tasmanian devils to the search space
            tasmanian_devils = np.clip(tasmanian_devils, lb, ub)
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct



