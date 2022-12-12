import numpy as np
import multiprocessing
from sim import sim


def run_sim(p):
    d = 100                                   # Number of input units
    k = 4                                     # Number of teacher hidden units
    kappa = 0                                 # Hidden layer exponent
    delta = 0                                 # Learning rate exponent
    noise = 1e-3                              # Noise level
    t_max = 1e5                               # Maximum t on the respective time scaling
    tol = 0                                   # Stops if abs(pop_risk_t - pop_risk_{t-1}) <= tol. If zero, stops on alpha_max
    save_folder = 'results/run_svda_train/'   # Folder to save
    gamma_0 = p                               # Gamma zero

    simul = sim(d,p,k, train_a= True, svd_only=True)   # Initialize the class
    Wt, at = simul.set_teacher(orthWt= True)           # orthWt: whether the teacher is delta_{rs} or not
    W0, a0 = simul.set_Theta0()                        # Gaussian uninformed initialization

    t, eg, Wf, _ = simul.fit(alpha_max= t_max,
                             delta=delta,
                             kappa=kappa,
                             lrW0= gamma_0,
                             lra0= gamma_0,
                             noise= noise,
                             lin_teacher= True,
                             tol= tol,
                             save_log= True,
                             save_folder= save_folder)

n_hidden = [ 4, 6, 8, 10, 12]

pool_obj = multiprocessing.Pool(len(n_hidden))
pool_obj.map(run_sim, n_hidden)
