import numpy as np


def get_likelihood(X, W, sigma2):
    n, d = X.shape
    C = W @ W.T + sigma2 * np.eye(d)
    X_centered = X-np.mean(X, axis=0)
    S = np.cov(X_centered.T)
    L = -n/2 * (d*np.log(2*np.pi) + np.log(np.linalg.det(C)) + np.trace(np.linalg.inv(C) @ S))
    return L

