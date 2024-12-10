import numpy as np
from scipy.linalg import eigh


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
    

