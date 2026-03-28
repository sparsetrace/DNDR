## berry_oos_sec.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

from scipy.spatial.distance import cdist
import plotly.graph_objects as go

from .del0 import del0
from .del1 import del1
from .del1as import del1as


# ----------------------------
# Analytic torus helpers
# ----------------------------

def torus_embed(theta: np.ndarray, phi: np.ndarray, R: float = 2.0, r: float = 0.7) -> np.ndarray:
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    return np.stack(
        [
            (R + r * np.cos(phi)) * np.cos(theta),
            (R + r * np.cos(phi)) * np.sin(theta),
            r * np.sin(phi),
        ],
        axis=-1,
    )


def torus_normal(theta: float, phi: float) -> np.ndarray:
    n = np.array(
        [
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi),
        ],
        dtype=float,
    )
    return n / np.linalg.norm(n)


def torus_tangent_basis(theta: float, phi: float, R: float = 2.0, r: float = 0.7) -> np.ndarray:
    d_theta = np.array(
        [
            -(R + r * np.cos(phi)) * np.sin(theta),
            (R + r * np.cos(phi)) * np.cos(theta),
            0.0,
        ],
        dtype=float,
    )
    d_phi = np.array(
        [
            -r * np.sin(phi) * np.cos(theta),
            -r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ],
        dtype=float,
    )

    e1 = d_theta / np.linalg.norm(d_theta)
    d_phi = d_phi - e1 * np.dot(e1, d_phi)
    e2 = d_phi / np.linalg.norm(d_phi)
    return np.column_stack([e1, e2])


def sample_torus_grid(n_theta: int = 20, n_phi: int = 20, R: float = 2.0, r: float = 0.7):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    X = torus_embed(TH.ravel(), PH.ravel(), R=R, r=r)
    return X, TH.ravel(), PH.ravel()


def sample_random_torus_queries(
    n: int,
    R: float = 2.0,
    r: float = 0.7,
    normal_offset: float | tuple[float, float] = 0.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

    true_points = torus_embed(theta, phi, R=R, r=r)
    normals = np.stack([torus_normal(t, p) for t, p in zip(theta, phi)], axis=0)

    if np.isscalar(normal_offset):
        offsets = np.full(n, float(normal_offset))
    else:
        a, b = normal_offset
        offsets = rng.uniform(a, b, size=n)

    queries = true_points + offsets[:, None] * normals
    return {
        "theta": theta,
        "phi": phi,
        "true_points": true_points,
        "normals": normals,
        "offsets": offsets,
        "queries": queries,
    }


# ----------------------------
# Linear-algebra helpers
# ----------------------------

def principal_angles_deg(B_est: np.ndarray, B_true: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(B_est.T @ B_true, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def projection_matrix(B: np.ndarray) -> np.ndarray:
    return B @ B.T


def subspace_frob_error(B_est: np.ndarray, B_true: np.ndarray) -> float:
    return np.linalg.norm(projection_matrix(B_est) - projection_matrix(B_true), ord="fro")


# ----------------------------
# Result containers
# ----------------------------

@dataclass
class BerryOOSSECResult:
    query_point: np.ndarray
    projected_point: np.ndarray
    tangent_basis: np.ndarray
    singular_values: np.ndarray
    raw_field_vectors: np.ndarray
    phi_query: np.ndarray


# ----------------------------
# Berry-style OOS SEC wrapper
# ----------------------------

class BerryOOSSEC:
    """
    DMAP-only OOS SEC wrapper built on top of Berry's SEC pieces.

    Training-side:
      - Del0 gives scalar modes u and eigenvalues l
      - Del1AS / Del1 gives SEC eigenfields U plus H (or G fallback)
      - xhat = X D u(:,1:n1) gives coordinate Fourier coefficients

    OOS-side:
      - Nyström extend only the scalar eigenfunction values phi(alpha)
      - Nyström-project the query to the manifold via xhat0 @ phi0
      - Evaluate each SEC field as an ambient arrow:
            v_ell(alpha) = xhat1 @ Uop_ell.T @ phi1(alpha)
      - Stack several arrows and SVD them into a local tangent basis
    """

    def __init__(
        self,
        X: np.ndarray,                      # shape (N, ambient_dim)
        n0: int = 60,                      # number of scalar DMAP modes
        n1: int = 20,                      # SEC frame truncation
        n_fields: int = 8,                 # number of smooth SEC fields to stack at query
        epsilon: Optional[float] = None,
        use_antisymmetric: bool = True,
    ):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("X must have shape (N, ambient_dim).")
        self.N, self.ambient_dim = self.X.shape

        self.x = self.X.T                  # Berry/MATLAB convention: shape (ambient_dim, N)
        self.n0 = min(int(n0), self.N)
        self.n1 = min(int(n1), self.n0)
        self.n_fields = int(n_fields)
        self.use_antisymmetric = bool(use_antisymmetric)

        # Match the MATLAB Del0 kernel scale if epsilon not provided.
        self.epsilon = float(epsilon) if epsilon is not None else self._estimate_epsilon_matlab_rule()

        # Training DMAP/0-Laplacian
        self.u, self.l, self.D = del0(self.x, self.n0, self.epsilon)

        # Recompute the quantities needed for Nyström extension.
        d = cdist(self.X, self.X, metric="euclidean")
        self.K = np.exp(-(d ** 2) / (self.epsilon ** 2) / 4.0)
        self.q = np.sum(self.K, axis=1)

        # Recover the diffusion-map eigenvalues lambda_j from l_j = -log(lambda_j) / epsilon^2
        self.lambda0 = np.exp(-self.l * (self.epsilon ** 2))
        self.lambda0 = np.clip(self.lambda0, 1e-14, None)

        # Training SEC/1-Laplacian
        if self.use_antisymmetric:
            self.U, self.L, self.D1, self.G, self.H, self.cijk = del1as(self.u, self.l, self.D, self.n1)
        else:
            self.U, self.L, self.D1, self.G, self.cijk = del1(self.u, self.l, self.D, self.n1)
            self.H = self.G

        # Coordinate Fourier coefficients in Berry's pushforward formula.
        self.xhat0 = self.x @ self.D @ self.u[:, : self.n0]   # ambient_dim x n0
        self.xhat1 = self.x @ self.D @ self.u[:, : self.n1]   # ambient_dim x n1

        # Convert smooth SEC fields to operator matrices, Berry-style.
        num_available_fields = self.U.shape[1]
        self.num_fields = min(self.n_fields, num_available_fields)
        self.operator_mats = []
        for ell in range(self.num_fields):
            # MATLAB reshape is column-major -> use order="F"
            Uop = (self.H.T @ self.U[:, ell]).reshape((self.n1, self.n1), order="F")
            self.operator_mats.append(Uop)

    def _estimate_epsilon_matlab_rule(self) -> float:
        """
        Berry's Del0.m heuristic:
            k = 1 + ceil(log(n))
            epsilon = mean(k-th smallest Euclidean distance in each column)
        """
        d = cdist(self.X, self.X, metric="euclidean")
        k = int(1 + np.ceil(np.log(self.n0)))
        kth_smallest = np.partition(d, k - 1, axis=0)[k - 1, :]
        return float(np.mean(kth_smallest))

    def nystrom_phi(self, y: np.ndarray, m: Optional[int] = None) -> np.ndarray:
        """
        Nyström extension consistent with Berry's alpha=1 Del0 normalization.

        For the sample-space right eigenvectors u_j of P = D^{-1} K_tilde:
            u_j(y) = (1 / lambda_j) * sum_i p_i(y) u_j(x_i)
        where
            p_i(y) ∝ k(y, x_i) / q_i.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != self.ambient_dim:
            raise ValueError(f"Expected query dimension {self.ambient_dim}, got {y.shape[0]}")

        m = self.n0 if m is None else min(int(m), self.n0)

        diff = self.X - y[None, :]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        k_y = np.exp(-(d ** 2) / (self.epsilon ** 2) / 4.0)

        # q_y cancels after normalization; use weights proportional to k(y, x_i) / q_i
        w = k_y / self.q
        Z = np.sum(w)
        if Z <= 0:
            raise RuntimeError("Nyström weights vanished; epsilon may be too small.")
        p = w / Z

        phi_y = (p @ self.u[:, :m]) / self.lambda0[:m]
        return phi_y

    def project_to_manifold(self, y: np.ndarray, m: Optional[int] = None) -> np.ndarray:
        """
        Nyström projection of ambient point y onto the learned manifold via
        the Nyström extension of the coordinate functions.
        """
        m = self.n0 if m is None else min(int(m), self.n0)
        phi_y = self.nystrom_phi(y, m=m)
        return self.xhat0[:, :m] @ phi_y

    def eval_field(self, y: np.ndarray, field_index: int, project_first: bool = True) -> np.ndarray:
        """
        Evaluate one SEC eigenfield as an ambient arrow at y using Berry's operator pushforward:
            v(y) = xhat1 @ Uop^T @ phi(y)
        """
        base = self.project_to_manifold(y) if project_first else np.asarray(y, dtype=float).reshape(-1)
        phi = self.nystrom_phi(base, m=self.n1)
        Uop = self.operator_mats[field_index]
        return self.xhat1 @ (Uop.T @ phi)

    def __call__(
        self,
        y: np.ndarray,
        d: int = 2,
        n_fields: Optional[int] = None,
        project_first: bool = True,
    ) -> BerryOOSSECResult:
        """
        Build a local tangent basis at y by:
          1) optionally Nyström-projecting y to the manifold,
          2) evaluating several smooth SEC fields there,
          3) SVD-cleaning the resulting ambient arrows.
        """
        d = int(d)
        n_fields = self.num_fields if n_fields is None else min(int(n_fields), self.num_fields)

        projected = self.project_to_manifold(y) if project_first else np.asarray(y, dtype=float).reshape(-1)
        phi = self.nystrom_phi(projected, m=self.n1)

        raw_fields = np.column_stack(
            [self.xhat1 @ (self.operator_mats[ell].T @ phi) for ell in range(n_fields)]
        )  # shape (ambient_dim, n_fields)

        U, S, _ = np.linalg.svd(raw_fields, full_matrices=False)
        basis = U[:, :d]

        return BerryOOSSECResult(
            query_point=np.asarray(y, dtype=float).reshape(-1),
            projected_point=projected,
            tangent_basis=basis,
            singular_values=S,
            raw_field_vectors=raw_fields,
            phi_query=phi,
        )

    def plot_query(self, result: BerryOOSSECResult, vector_scale: float = 0.6):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=self.X[:, 0],
                y=self.X[:, 1],
                z=self.X[:, 2],
                mode="markers",
                marker=dict(size=2.5, opacity=0.35),
                name="training points",
            )
        )

        q = result.query_point
        p = result.projected_point

        fig.add_trace(
            go.Scatter3d(
                x=[q[0]], y=[q[1]], z=[q[2]],
                mode="markers",
                marker=dict(size=6),
                name="query point",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]],
                mode="markers",
                marker=dict(size=7),
                name="Nyström projection",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[q[0], p[0]], y=[q[1], p[1]], z=[q[2], p[2]],
                mode="lines",
                name="projection line",
            )
        )

        for j in range(result.tangent_basis.shape[1]):
            v = vector_scale * result.tangent_basis[:, j]
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0], p[0] + v[0]],
                    y=[p[1], p[1] + v[1]],
                    z=[p[2], p[2] + v[2]],
                    mode="lines+markers",
                    name=f"estimated tangent {j+1}",
                )
            )

        fig.update_layout(
            title="Berry-style OOS SEC tangent basis",
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig


# ----------------------------
# Torus batch evaluation + Plotly compare
# ----------------------------

def evaluate_on_torus_queries(
    sec: BerryOOSSEC,
    queries: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    R: float = 2.0,
    r: float = 0.7,
    d: int = 2,
    n_fields: Optional[int] = None,
    project_first: bool = True,
) -> Dict[str, Any]:
    queries = np.asarray(queries)
    results = []
    true_bases = []
    est_bases = []
    projected_points = []
    true_points = []
    principal_angles = []
    frob_errors = []
    singular_values = []

    for i in range(len(queries)):
        res = sec(queries[i], d=d, n_fields=n_fields, project_first=project_first)
        B_true = torus_tangent_basis(theta[i], phi[i], R=R, r=r)
        B_est = res.tangent_basis
        x_true = torus_embed(theta[i], phi[i], R=R, r=r)

        results.append(res)
        true_bases.append(B_true)
        est_bases.append(B_est)
        projected_points.append(res.projected_point)
        true_points.append(x_true)
        principal_angles.append(principal_angles_deg(B_est, B_true))
        frob_errors.append(subspace_frob_error(B_est, B_true))
        singular_values.append(res.singular_values)

    true_bases = np.stack(true_bases, axis=0)
    est_bases = np.stack(est_bases, axis=0)
    projected_points = np.stack(projected_points, axis=0)
    true_points = np.stack(true_points, axis=0)
    principal_angles = np.stack(principal_angles, axis=0)
    frob_errors = np.asarray(frob_errors)

    return {
        "results": results,
        "true_bases": true_bases,
        "est_bases": est_bases,
        "projected_points": projected_points,
        "true_points": true_points,
        "principal_angles": principal_angles,
        "max_angle": principal_angles.max(axis=1),
        "mean_angle": principal_angles.mean(axis=1),
        "frob_error": frob_errors,
        "singular_values": singular_values,
    }


def summarize_eval(eval_out: Dict[str, Any]) -> None:
    max_angle = eval_out["max_angle"]
    mean_angle = eval_out["mean_angle"]
    frob = eval_out["frob_error"]

    print("Number of query points:", len(max_angle))
    print("Max principal angle:")
    print("  mean   =", float(np.mean(max_angle)))
    print("  median =", float(np.median(max_angle)))
    print("  95%    =", float(np.percentile(max_angle, 95)))
    print("  worst  =", float(np.max(max_angle)))
    print()
    print("Mean principal angle:")
    print("  mean   =", float(np.mean(mean_angle)))
    print("  median =", float(np.median(mean_angle)))
    print()
    print("Projector Frobenius error:")
    print("  mean   =", float(np.mean(frob)))
    print("  median =", float(np.median(frob)))
    print("  worst  =", float(np.max(frob)))


def plot_compare_query(
    sec: BerryOOSSEC,
    eval_out: Dict[str, Any],
    query_index: int,
    vector_scale: float = 0.5,
    show_all_queries: bool = True,
    color_by: str = "max_angle",
):
    res = eval_out["results"][query_index]
    q = res.query_point
    p_proj = res.projected_point
    p_true = eval_out["true_points"][query_index]
    B_true = eval_out["true_bases"][query_index]
    B_est = eval_out["est_bases"][query_index]

    if color_by == "mean_angle":
        colors = eval_out["mean_angle"]
        title_color = "mean angle (deg)"
    else:
        colors = eval_out["max_angle"]
        title_color = "max angle (deg)"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=sec.X[:, 0],
            y=sec.X[:, 1],
            z=sec.X[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.15),
            name="training torus",
        )
    )

    if show_all_queries:
        all_queries = np.stack([r.query_point for r in eval_out["results"]], axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=all_queries[:, 0],
                y=all_queries[:, 1],
                z=all_queries[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=colors,
                    colorscale="Viridis",
                    colorbar=dict(title=title_color),
                    opacity=0.9,
                ),
                name="novel queries",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[q[0]], y=[q[1]], z=[q[2]],
            mode="markers",
            marker=dict(size=8, symbol="diamond"),
            name="selected query",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[p_true[0]], y=[p_true[1]], z=[p_true[2]],
            mode="markers",
            marker=dict(size=7),
            name="true torus point",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[p_proj[0]], y=[p_proj[1]], z=[p_proj[2]],
            mode="markers",
            marker=dict(size=7),
            name="Nyström projection",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[q[0], p_true[0]], y=[q[1], p_true[1]], z=[q[2], p_true[2]],
            mode="lines",
            name="query → true point",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[q[0], p_proj[0]], y=[q[1], p_proj[1]], z=[q[2], p_proj[2]],
            mode="lines",
            name="query → projected point",
        )
    )

    for j in range(B_true.shape[1]):
        v = vector_scale * B_true[:, j]
        fig.add_trace(
            go.Scatter3d(
                x=[p_true[0], p_true[0] + v[0]],
                y=[p_true[1], p_true[1] + v[1]],
                z=[p_true[2], p_true[2] + v[2]],
                mode="lines+markers",
                line=dict(dash="dash"),
                name=f"true tangent {j+1}",
            )
        )

    for j in range(B_est.shape[1]):
        v = vector_scale * B_est[:, j]
        fig.add_trace(
            go.Scatter3d(
                x=[p_proj[0], p_proj[0] + v[0]],
                y=[p_proj[1], p_proj[1] + v[1]],
                z=[p_proj[2], p_proj[2] + v[2]],
                mode="lines+markers",
                name=f"estimated tangent {j+1}",
            )
        )

    ang = np.round(eval_out["principal_angles"][query_index], 3)
    fig.update_layout(
        title=f"Query {query_index}: principal angles = {ang} deg",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig
