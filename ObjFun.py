import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_VAutoEn import Model_VAutoEn


def objfun(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_VAutoEn(data, Tar, sol)
            Eval = evaluation(Tar, pred)
            Fitn[i] = (1 / Eval[13]) + Eval[9]
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_VAutoEn(data, Tar, sol)
        Eval = evaluation(Tar, pred)
        Fitn = (1 / Eval[13]) + Eval[9]
        return Fitn
