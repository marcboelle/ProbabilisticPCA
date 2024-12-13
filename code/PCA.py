import numpy as np
from scipy.linalg import eigh, orth


from typing import Union
import time
class PCA:
    
    def __init__(self, nb_components):
        self.nb_components = nb_components
        self.r2 = None
        self.mean = None
        self.components = None
    
    def fit(self, X):
        #X is assumed to be of shape (n_samples, d) with d dimension of the samples
        d = X.shape[1]
        self.mean = np.mean(X, axis = 0)
        
        X_bis = X.copy()
        X_bis -= self.mean

        covariance = np.cov(X_bis.T)

        eigenvalues, eigenvectors = eigh(covariance)
        decr_eigenvalues, decr_eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

        self.components = decr_eigenvectors[:, :self.nb_components]
        self.r2 = np.sum(decr_eigenvalues[:self.nb_components])/np.sum(decr_eigenvalues)

    def transform(self, X):

        return (X - self.mean) @ self.components

    
class PPCA:
    
    def __init__(self, nb_components : int, sigma2 : Union[float, None] = None, R : Union[np.ndarray, None] = None):
        self.nb_components = nb_components
        self.mean = None
        self.W = None
        self.components = None
        self.sigma2 = sigma2
        self.inv_M = None
        self.R = R 
    
    def fit(self, X):
        #X is assumed to be of shape (n_samples, d) with d dimension of the samples
        d = X.shape[1]
        self.mean = np.mean(X, axis = 0)
        
        X_bis = X.copy()
        X_bis -= self.mean

        covariance = np.cov(X_bis.T)

        eigenvalues, eigenvectors = eigh(covariance)
        decr_eigenvalues, decr_eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

        if self.sigma2 is None:
            self.sigma2 = 1/(d-self.nb_components) * np.sum(decr_eigenvalues[self.nb_components:])

        diag = np.diag(np.sqrt(decr_eigenvalues[:self.nb_components] - self.sigma2))

        #Add optional rotation, else identity
        if self.R is None:
            self.R = np.eye(self.nb_components)

        self.W = decr_eigenvectors[:, :self.nb_components] @ diag @ self.R
        self.inv_M = np.linalg.inv(self.W.T @ self.W + self.sigma2*np.eye(self.nb_components))

        self.components = decr_eigenvectors[:, :self.nb_components]

    def transform(self, X):

        return (X - self.mean) @ self.W @ self.inv_M.T
    




def EM_for_PPCA(X, nb_components : int, W_0 : np.ndarray, sigma2_0 : int, max_iter : int, plot_time : bool = False):
    
    if plot_time:
        time_start = time.time()

    N, d = X.shape

    W, sigma2 = W_0, sigma2_0

    X_centered = X - np.mean(X, axis=0)
    tr_S = 1/N * np.sum(X_centered**2)

    for i in range(max_iter):
        if i%100000 == 0:
            print(f"Epoch {i}")
            print(W)
            print(sigma2)
        XW = X_centered @ W
        SW = 1/N * X_centered.T @ XW

        M = W.T @ W + sigma2 * np.eye(nb_components)
        #inv_M = np.linalg.inv(M)
        inv_M = np.linalg.solve(M, np.eye(nb_components))
        #E step:
        W = SW @ np.linalg.inv(sigma2*np.eye(nb_components) + inv_M @ W.T @ SW)

        #M step:
        sigma2 = 1/d * (tr_S - np.trace(SW @ inv_M @ W.T))
    
    if plot_time:
        time_total = time.time() - time_start
        return W, sigma2, time_total
    return W, sigma2
    




def ppca_missing_values(Y, d, dia):
    """
    Implements probabilistic PCA for data with missing values,
    using a factorizing distribution over hidden states and hidden observations.

    Args:
        Y:   (N by D ) input numpy ndarray of data vectors
        d:   (  int  ) dimension of latent space
        dia: (boolean) if True: print objective each step

    Returns:
        C:  (D by d ) C*C' + I*ss is covariance model, C has scaled principal directions as cols
        ss: ( float ) isotropic variance outside subspace
        M:  (D by 1 ) data mean
        X:  (N by d ) expected states
        Ye: (N by D ) expected complete observations (differs from Y if data is missing)

        Based on MATLAB code from J.J. VerBeek, 2006. http://lear.inrialpes.fr/~verbeek
    """
    N, D = Y.shape  # N observations in D dimensions (i.e. D is number of features, N is samples)
    threshold = 1E-4  # minimal relative change in objective function to continue
    hidden = np.isnan(Y)
    missing = hidden.sum()

    if missing > 0:
        M = np.nanmean(Y, axis=0)
    else:
        M = np.mean(Y, axis=0)

    Ye = Y - np.matlib.repmat(M, N, 1)

    if missing > 0:
        Ye[hidden] = 0

    # initialize
    C = np.random.normal(loc=0.0, scale=1.0, size=(D, d))
    CtC = C.T @ C
    X = Ye @ C @ np.linalg.inv(CtC)
    recon = X @ C.T
    recon[hidden] = 0
    ss = np.sum((recon - Ye) ** 2) / (N * D - missing)

    count = 1
    old = np.inf

    # EM Iterations
    while (count):
        Sx = np.linalg.inv(np.eye(d) + CtC / ss)  # E-step, covariances
        ss_old = ss
        if missing > 0:
            proj = X @ C.T
            Ye[hidden] = proj[hidden]

        X = Ye @ C @ Sx / ss  # E-step: expected values

        SumXtX = X.T @ X  # M-step
        C = Ye.T @ X @ (SumXtX + N * Sx).T @ np.linalg.inv(((SumXtX + N * Sx) @ (SumXtX + N * Sx).T))
        CtC = C.T @ C
        ss = (np.sum((X @ C.T - Ye) ** 2) + N * np.sum(CtC * Sx) + missing * ss_old) / (N * D)
        # transform Sx determinant into numpy longdouble in order to deal with high dimensionality
        Sx_det = np.min(Sx).astype(np.longdouble) ** Sx.shape[0] * np.linalg.det(Sx / np.min(Sx))
        objective = N * D + N * (D * np.log(ss) + np.trace(Sx) - np.log(Sx_det)) + np.trace(SumXtX) - missing * np.log(ss_old)

        rel_ch = np.abs(1 - objective / old)
        old = objective

        count = count + 1
        if rel_ch < threshold and count > 5:
            count = 0
        if dia:
            print(f"Objective: {objective:.2f}, Relative Change {rel_ch:.5f}")

    C = orth(C)
    covM = np.cov((Ye @ C).T)
    vals, vecs = np.linalg.eig(covM)
    ordr = np.argsort(vals)[::-1]
    vecs = vecs[:, ordr]

    C = C @ vecs
    X = Ye @ C

    # add data mean to expected complete data
    Ye = Ye + np.matlib.repmat(M, N, 1)

    return C, ss, M, X, Ye

