import time
import numpy as np
from tqdm import tqdm

class CCA:
    pass

class PCCA:
    pass



def EM_for_PCCA(X : int, d_A : int, d_B : int, nb_components : int, W_0 : np.ndarray, phi_0 : np.ndarray, max_iter : int, plot_time : bool = False):
    
    if plot_time:
        time_start = time.time()

    N, d = X.shape

    X_bis = X.copy()
    X_bis -= np.mean(X_bis, axis=0)

    sigma = np.cov(X_bis.T)

    W, phi = W_0, phi_0


    for i in tqdm(range(max_iter)):
        #inv_phi = np.linalg.solve(phi, np.eye(d))
        inv_phi = np.linalg.inv(phi)
        M = np.linalg.solve(np.eye(nb_components) + W.T @ inv_phi @ W, np.eye(nb_components))



        W_next = sigma @ inv_phi @ W @ M @ np.linalg.inv(M + M @ W.T @ inv_phi @ sigma @ inv_phi @ W @ M)
        phi_next = sigma - sigma @ inv_phi @ W @ M @ W_next.T
        phi_next[:d_A, d_A:] = 0
        phi_next[d_A:, :d_A] = 0

        W, phi = W_next, phi_next

    if plot_time:
        time_total = time.time() - time_start
        return W, phi, M, time_total
    return W, phi, M