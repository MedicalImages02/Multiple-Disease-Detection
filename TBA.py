import time

import numpy as np


def TBA(TB, objective_func, L, U, MaxT):
    """
    Tiger Beetle Optimization (TBO)
    """
    N, Dim = TB.shape[0], TB.shape[1]
    fitness = np.array([objective_func(tb) for tb in TB])
    Hm = 5
    p = 0.9
    SD0 = 1.0
    It = 1
    BestSol = np.zeros((Dim, 1))
    Convergence_curve = np.zeros((MaxT, 1))
    t = 0
    ct = time.time()
    # Main Loop
    while It <= MaxT:

        # Best and Worst TB
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)

        B = TB[best_idx].copy()
        W = TB[worst_idx].copy()
        fW = fitness[worst_idx]

        # SD(t) (Eq. 5)
        SD = SD0 - abs((p / np.pi) * np.arctan((p / np.pi) * It))
        # Alpha & Beta (Eq. 10, 11)
        alpha = 1 - (It / MaxT) ** 2
        beta = 1 - alpha

        # Hole digging ability (Eq. 4)
        H = np.zeros(N, dtype=int)
        for i in range(N):
            H[i] = int(max(0, (1 - np.exp(fitness[i] / fW)) * Hm))

        # Dig holes & generate larvae (Eq. 6)
        holes = []
        for i in range(N):
            for _ in range(H[i]):
                R = 1 + np.random.rand()  # R ∈ [1,2]
                rand_vec = np.random.rand(Dim)
                if np.random.rand() > 0.5:
                    TBnew = TB[i] + R * SD * rand_vec
                else:
                    TBnew = TB[i] - R * SD * rand_vec

                TBnew = np.clip(TBnew, L, U)
                holes.append(TBnew)

        if holes:
            holes = np.array(holes)
            TB = np.vstack((TB, holes))
            fitness = np.append(
                fitness,
                [objective_func(h) for h in holes]
            )

        # Survival probability (Eq. 7 & 8)
        prob = 1 - (fitness / fW) / np.sum(fitness / fW)

        survivors = []
        for i in range(len(TB)):
            if np.random.rand() <= prob[i]:
                survivors.append(TB[i])

        TB = np.array(survivors)
        fitness = np.array([objective_func(tb) for tb in TB])
        N = len(TB)

        # Average position B̅
        B_bar = np.mean(TB, axis=0)

        # Search insect-prone regions (Eq. 9)
        for i in range(N):
            TB[i] = alpha * TB[i] + beta * (B - B_bar) * np.random.rand(Dim)
            TB[i] = np.clip(TB[i], L, U)

        # Mating process (Eq. 12)
        for i in range(N):
            k, l = np.random.choice(N, 2, replace=False)
            TB[i] = TB[i] + (TB[k] - TB[l]) * np.random.rand(Dim)
            TB[i] = np.clip(TB[i], L, U)

        fitness = np.array([objective_func(tb) for tb in TB])
        It += 1
        best_idx = np.argmin(fitness)
        BestSol = TB[best_idx]
        BestFit = fitness[best_idx]
        Convergence_curve[t] = BestFit
    BestFit = Convergence_curve[MaxT - 1][0]
    ct = time.time() - ct
    return BestFit, Convergence_curve, BestSol, ct
