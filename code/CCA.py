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

def estimate_latent_and_missing_values(x, M, W, muA, muB, UA, UB):
    d_A = muA.shape[0]
    q = W.shape[1]

    xA = x[:d_A]
    xB = x[d_A:]

    # Find indices of observed and missing values
    obs_idx = ~np.isnan(x)
    obs_idx_A, obs_idx_B = obs_idx[:d_A], obs_idx[d_A:]
    missing_idx = np.isnan(x)
    missing_idx_A, missing_idx_B = missing_idx[:d_A], missing_idx[d_A:]

    # Extract observed data and corresponding W, mu
    xA_obs, xB_obs = xA[obs_idx_A], xB[obs_idx_B]
    # WA_obs, WB_obs = WA[obs_idx, :], WB[obs_idx, :]
    muA_obs, muB_obs = muA[obs_idx_A], muB[obs_idx_B]
    UA_obs, UB_obs = UA[obs_idx_A, :], UB[obs_idx_B, :]

    zA = M.T @ UA_obs[:, :q].T @ (xA_obs - muA_obs)
    zB = M.T @ UB_obs[:, :q].T @ (xB_obs - muB_obs)

    # Predict missing values using z
    WA_miss, WB_miss = W[:d_A][missing_idx_A], W[d_A:][missing_idx_B]
    muA_miss, muB_miss = muA[missing_idx_A], muB[missing_idx_B]
    x_filled = np.copy(x)

    # print('A:')
    # print(WA_miss.shape)
    # print(zA.shape)
    # print(muA_miss.shape)

    # print('\nB:')
    # print(WB_miss.shape)
    # print(zB.shape)
    # print(muB_miss.shape)

    if np.any(missing_idx_A):
        x_filled[:d_A][missing_idx_A] = WA_miss @ zA.T + muA_miss  # Only fill missing indices
       
    if np.any(missing_idx_B):
        x_filled[d_A:][missing_idx_B] = WB_miss @ zB.T + muB_miss  # Only fill missing indices

    return zA, zB, x_filled

def EM_for_PCCA_missing(X : int, d_A : int, d_B : int, nb_components : int, W_0 : np.ndarray, phi_0 : np.ndarray, max_iter : int, plot_time : bool = False):
    
    if plot_time:
        time_start = time.time()

    W, phi = W_0, phi_0

    XA = X[:, :d_A]
    XB = X[:, d_A:]
    n = X.shape[0]

    muA = np.nanmean(XA, axis=0)
    muB = np.nanmean(XB, axis=0)

    sigmaA = np.eye(d_A)
    sigmaB = np.eye(d_B)

    eps = 1e-10 # numerical stability
    eigvalA, eigvecA = np.linalg.eigh(sigmaA)
    eigvalB, eigvecB = np.linalg.eigh(sigmaB)
    UA = np.diag(1/np.sqrt(eigvalA + eps)) @ eigvecA
    UB = np.diag(1/np.sqrt(eigvalB + eps)) @ eigvecB
    M = np.eye(nb_components)

    for i in tqdm(range(max_iter)):
        
        # FILL IN MISSING VALUES
        X_filled = X.copy()

        for i in range(n):
            _, _, x_full = estimate_latent_and_missing_values(X[i, :],M, W, muA, muB, UA, UB) 
            # X and not X_filled to go back to the original values with missing entries
            X_filled[i, :] = x_full
        
        # UPDATE COVARIANCE MATRIX
        sigma = np.cov(X_filled.T)

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