from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh


def _estimate_epsilon_from_matlab_rule(distances: np.ndarray, n_modes: int) -> float:
    """
    MATLAB Del0.m heuristic:
        k = 1 + ceil(log(n))
        epsilon = mink(d, k)
        epsilon = mean(epsilon(k,:))
    where d contains Euclidean pairwise distances including zeros on the diagonal.
    """
    k = int(1 + np.ceil(np.log(n_modes)))
    kth_smallest = np.partition(distances, k - 1, axis=0)[k - 1, :]
    return float(np.mean(kth_smallest))


def del0(x: np.ndarray, n: int, epsilon: float | None = None):
    """
    Python translation of Berry's MATLAB SEC/Del0.m.

    Parameters
    ----------
    x : array, shape (m, N)
        N points in R^m, stored columnwise as in MATLAB.
    n : int
        Number of 0-Laplacian eigenfunctions to compute.
    epsilon : float, optional
        Kernel bandwidth. If omitted, uses the MATLAB heuristic.

    Returns
    -------
    u : array, shape (N, n)
        Eigenfunctions evaluated on the data: u[i, j] = phi_j(x_i).
    l : array, shape (n,)
        Diffusion-map/Laplace-Beltrami eigenvalues.
    D : array, shape (N, N)
        Diagonal inner-product matrix such that u.T @ D @ u ≈ I.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (m, N)")

    # MATLAB: d = pdist2(x',x');
    d = cdist(x.T, x.T, metric="euclidean")

    if epsilon is None:
        epsilon = _estimate_epsilon_from_matlab_rule(d, n)

    # MATLAB: d = exp(-d.^2/epsilon^2/4);
    K = np.exp(-(d ** 2) / (epsilon ** 2) / 4.0)
    K = 0.5 * (K + K.T)

    # alpha=1 normalization as in the MATLAB code
    D_left = np.diag(1.0 / np.sum(K, axis=1))
    K_tilde = D_left @ K @ D_left
    D = np.diag(np.sum(K_tilde, axis=1))
    K_tilde = 0.5 * (K_tilde + K_tilde.T)

    # MATLAB: [u,l] = eigs(d,D,n,'LM');
    # Since K_tilde is symmetric and D is diagonal positive, use dense generalized eigh.
    evals, evecs = eigh(K_tilde, D)
    order = np.argsort(np.real(evals))[::-1]
    evals = np.real(evals[order][:n])
    evecs = np.real(evecs[:, order][:, :n])

    # MATLAB:
    #   l = -log(l)/epsilon^2;
    lap_eigs = -np.log(np.clip(evals, 1e-15, None)) / (epsilon ** 2)

    return evecs, lap_eigs, D
