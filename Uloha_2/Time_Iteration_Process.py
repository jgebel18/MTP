import numpy as np

import scipy.sparse.linalg as spla

class GenerationofTime:

    @staticmethod
    def Solve_time_equation(T_init, Global_k, Global_m, Time_steps, F):
        T_init = np.full(F.size,T_init)
        Temperatures_in_time= np.zeros((Time_steps.size,F.size))

        dt= Time_steps[1]-Time_steps[0]
        T_k=T_init
        A = ((1 / dt) * Global_m + Global_k)
        for i in range(len(Time_steps)):
            B= (1/dt)*(Global_m@T_k)+F
            T_k_new= spla.spsolve(A, B)
            Temperatures_in_time[i]= T_k_new
            T_k = T_k_new
        return Temperatures_in_time
