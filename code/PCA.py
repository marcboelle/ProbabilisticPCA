import numpy as np
from scipy.linalg import eigh

from typing import Union

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
    




def EM_for_PPCA(X, nb_components : int, W_0 : np.ndarray, sigma2_0 : int, epsilon : int, max_iter : int):
    
    tr_S = np.sum()

    W, sigma2 = W_0, sigma2_0
    
    for i in range(max_iter):
        SW = np.sum(X[:, ])