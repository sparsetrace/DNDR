from __future__ import annotations

import numpy as np
from scipy.linalg import eig


def del1(u: np.ndarray, l: np.ndarray, D: np.ndarray, n1: int | None = None, n2: int | None = None):
    """
    Python translation of SEC/Del1.m (non-antisymmetric frame).
    """
    u = np.asarray(u, dtype=float)
    l = np.asarray(l, dtype=float).reshape(-1)
    D = np.asarray(D, dtype=float)

    _, n0 = u.shape
    if n1 is None:
        n1 = n0
    if n2 is None:
        n2 = n1

    Du = D @ u

    # MATLAB uses mean here, not sum.
    tmp = (
        u[:, :n1][:, :, None, None]
        * u[:, :n1][:, None, :, None]
        * Du[:, None, None, :]
    )
    cijk = np.mean(tmp, axis=0)

    l1 = np.transpose(np.tile(l.reshape(-1, 1, 1, 1, 1), (1, n1, n1, n2, n2)), (1, 3, 0, 2, 4))

    cijkl = np.squeeze(
        np.sum(
            np.tile(cijk[:, :, :, None, None], (1, 1, 1, n2, n2))
            * np.transpose(np.tile(cijk[:n2, :n2, :, None, None], (1, 1, 1, n1, n1)), (3, 4, 2, 0, 1)),
            axis=2,
        )
    )

    cikjls = np.squeeze(
        np.sum(
            np.transpose(np.tile(cijk[:, :, :, None, None], (1, 1, 1, n2, n2)), (0, 3, 2, 1, 4))
            * l1
            * np.transpose(np.tile(cijk[:n2, :n2, :, None, None], (1, 1, 1, n1, n1)), (3, 0, 2, 4, 1)),
            axis=2,
        )
    )

    l2 = np.tile(l[:n1].reshape(-1, 1, 1, 1), (1, n1, n2, n2))
    l3 = np.tile(l[:n2].reshape(-1, 1, 1, 1), (1, n2, n1, n1))

    G = 0.5 * (
        (np.transpose(l3, (2, 0, 3, 1)) + np.transpose(l3, (2, 1, 3, 0))) * np.transpose(cijkl, (0, 2, 1, 3))
        - cikjls
    )

    ciljks = np.squeeze(
        np.sum(
            np.transpose(np.tile(cijk[:, :n2, :, None, None], (1, 1, 1, n1, n2)), (0, 4, 2, 3, 1))
            * l1
            * np.transpose(np.tile(cijk[:, :n2, :, None, None], (1, 1, 1, n1, n2)), (3, 1, 2, 0, 4)),
            axis=2,
        )
    )

    D1 = 0.25 * (
        np.transpose(l2, (0, 2, 1, 3))
        + np.transpose(l3, (2, 0, 3, 1))
        + np.transpose(l2, (1, 2, 0, 3))
        + np.transpose(l3, (2, 1, 3, 0))
    ) * (ciljks - cikjls)

    cijkls = np.squeeze(
        np.sum(
            np.tile(cijk[:, :n2, :, None, None], (1, 1, 1, n1, n2))
            * l1
            * np.transpose(np.tile(cijk[:, :n2, :, None, None], (1, 1, 1, n1, n2)), (3, 4, 2, 0, 1)),
            axis=2,
        )
    )

    D1 = D1 + 0.25 * (
        -np.transpose(l2, (0, 2, 1, 3))
        + np.transpose(l3, (2, 0, 3, 1))
        - np.transpose(l2, (1, 2, 0, 3))
        + np.transpose(l3, (2, 1, 3, 0))
    ) * cijkls

    cikjls2 = np.squeeze(
        np.sum(
            np.transpose(np.tile(cijk[:, :, :, None, None], (1, 1, 1, n2, n2)), (0, 3, 2, 1, 4))
            * (l1 ** 2)
            * np.transpose(np.tile(cijk[:n2, :n2, :, None, None], (1, 1, 1, n1, n1)), (3, 0, 2, 4, 1)),
            axis=2,
        )
    )
    D1 = D1 + 0.25 * cikjls2

    cijkls2 = np.squeeze(
        np.sum(
            np.tile(cijk[:, :n2, :, None, None], (1, 1, 1, n1, n2))
            * (l1 ** 2)
            * np.transpose(np.tile(cijk[:, :n2, :, None, None], (1, 1, 1, n1, n2)), (3, 4, 2, 0, 1)),
            axis=2,
        )
    )
    D1 = D1 + 0.25 * cijkls2

    ciljks2 = np.squeeze(
        np.sum(
            np.transpose(np.tile(cijk[:, :n2, :, None, None], (1, 1, 1, n1, n2)), (0, 4, 2, 3, 1))
            * (l1 ** 2)
            * np.transpose(np.tile(cijk[:, :n2, :, None, None], (1, 1, 1, n1, n2)), (3, 1, 2, 0, 4)),
            axis=2,
        )
    )
    D1 = D1 - 0.25 * ciljks2

    D1 = D1.reshape(n1 * n2, n1 * n2)
    G = G.reshape(n1 * n2, n1 * n2)

    D1 = 0.5 * (D1 + D1.T)
    G = 0.5 * (G + G.T)

    Ut, St, _ = np.linalg.svd(D1 + G, full_matrices=False)
    below = np.where(St / St[0] < 1e-3)[0]
    NN = int(below[0]) if below.size else len(St)

    D1proj = Ut[:, :NN].T @ D1 @ Ut[:, :NN]
    D1proj = 0.5 * (D1proj + D1proj.T)

    Gproj = Ut[:, :NN].T @ G @ Ut[:, :NN]
    Gproj = 0.5 * (Gproj + Gproj.T)

    vals, vecs = eig(D1proj, Gproj)
    order = np.argsort(np.abs(vals))
    vals = np.real(vals[order])
    vecs = np.real(vecs[:, order])

    U = Ut[:, :NN] @ vecs
    L = vals
    return U, L, D1, G, cijk
