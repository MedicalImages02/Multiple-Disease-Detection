import numpy as np
import time


def WOA(population, fobj, VRmin, VRmax, T):
    """
    Wombat Optimization Algorithm (WOA)
    """

    start_time = time.time()

    population = np.array(population)
    N, dim = population.shape

    lb = VRmin[0]
    ub = VRmax[0]

    # Initial fitness evaluation
    fitness = np.array([fobj(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = population[best_idx].copy()

    Convergence_curve = np.zeros((T, 1))

    for t in range(T):
        # Foraging Phase
        for i in range(N):
            # Flatten population similar to MATLAB population(:)
            CFP = population.copy()

            if CFP.size > 0:
                rand_idx = np.random.randint(0, N)
                SFP = CFP[rand_idx]

                r = np.random.rand(dim)
                new_position = population[i] + r * (SFP - population[i])

                # Apply bounds
                new_position = np.clip(new_position, lb, ub)

                new_fitness = fobj(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

        # Escape Phase
        for i in range(N):
            r = np.random.rand(dim)
            new_position = population[i] + (1 - 2 * r) * (ub - lb) / t

            # Apply bounds
            new_position = np.clip(new_position, lb, ub)

            new_fitness = fobj(new_position)
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

        # Update Global Best
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx].copy()

        Convergence_curve[t - 1] = best_fitness

    ct = time.time() - start_time
    return best_fitness, Convergence_curve, best_solution, ct
