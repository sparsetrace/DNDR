import numpy as np
import matplotlib.pyplot as plt
# (mpl_toolkits 3D is registered automatically by projection="3d" on modern matplotlib)


# ============================================================
# Arc-length geometry for an isometric swiss roll  (UNCHANGED)
# ============================================================
def _spiral_S(theta, r0=0.5, b=0.6):
    """
    Primitive of sqrt((r0 + b*theta)^2 + b^2) dtheta
    for the Archimedean spiral r(theta)=r0+b*theta.
    """
    theta = np.asarray(theta, dtype=float)
    u = r0 + b * theta
    return (u * np.sqrt(u**2 + b**2) + b**2 * np.arcsinh(u / b)) / (2.0 * b)


def _arc_length(theta0, theta1, r0=0.5, b=0.6):
    return _spiral_S(theta1, r0=r0, b=b) - _spiral_S(theta0, r0=r0, b=b)


def _theta_from_sheet_width(width, theta_min=1.5*np.pi, r0=0.5, b=0.6):
    """
    Find theta_max such that the spiral arc length from theta_min to theta_max
    equals 'width'.
    """
    if width <= 0:
        raise ValueError("width must be > 0")

    lo = theta_min
    hi = theta_min + 1.0

    while _arc_length(theta_min, hi, r0=r0, b=b) < width:
        hi = theta_min + 2.0 * (hi - theta_min)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        L = _arc_length(theta_min, mid, r0=r0, b=b)
        if L < width:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# ============================================================
# Build a roll map from a literal sheet size W x H  (UNCHANGED)
# ============================================================
def make_isometric_swiss_roll_map_from_sheet(
    width=18.0,
    height=10.0,
    theta_min=1.5*np.pi,
    r0=0.5,
    b=0.6,
    n_lookup=30000,
):
    theta_max = _theta_from_sheet_width(width, theta_min=theta_min, r0=r0, b=b)

    theta_grid = np.linspace(theta_min, theta_max, n_lookup)
    S_grid = _spiral_S(theta_grid, r0=r0, b=b)
    S0 = S_grid[0]

    return {
        "width": float(width),
        "height": float(height),
        "theta_min": float(theta_min),
        "theta_max": float(theta_max),
        "r0": float(r0),
        "b": float(b),
        "theta_grid": theta_grid,
        "S_grid": S_grid,
        "S0": float(S0),
    }


# ============================================================
# Forward / inverse maps and sampling  (UNCHANGED)
# ============================================================
def sheet_to_swiss_roll(SH, roll_map):
    SH = np.asarray(SH, dtype=float)
    if SH.ndim != 2 or SH.shape[1] != 2:
        raise ValueError("SH must have shape (N,2)")

    s = SH[:, 0]
    h = SH[:, 1]

    S_abs = roll_map["S0"] + s
    theta = np.interp(S_abs, roll_map["S_grid"], roll_map["theta_grid"])
    r = roll_map["r0"] + roll_map["b"] * theta

    x = r * np.cos(theta)
    z = r * np.sin(theta)
    y = h

    XYZ = np.column_stack([x, y, z])
    return XYZ, theta


def swiss_roll_to_sheet(XYZ, roll_map, clip=True):
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.ndim != 2 or XYZ.shape[1] != 3:
        raise ValueError("XYZ must have shape (N,3)")

    x = XYZ[:, 0]
    y = XYZ[:, 1]
    z = XYZ[:, 2]

    r = np.sqrt(x**2 + z**2)
    theta = (r - roll_map["r0"]) / roll_map["b"]

    S_abs = _spiral_S(theta, r0=roll_map["r0"], b=roll_map["b"])
    s = S_abs - roll_map["S0"]
    h = y

    SH = np.column_stack([s, h])
    if clip:
        SH[:, 0] = np.clip(SH[:, 0], 0.0, roll_map["width"])
        SH[:, 1] = np.clip(SH[:, 1], 0.0, roll_map["height"])
    return SH


def sample_sheet_random(n, width, height, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, width, size=n)
    h = rng.uniform(0.0, height, size=n)
    return np.column_stack([s, h])


def sample_sheet_grid(width, height, n_s=120, n_h=40):
    s = np.linspace(0.0, width, n_s)
    h = np.linspace(0.0, height, n_h)
    S, H = np.meshgrid(s, h, indexing="xy")
    return np.column_stack([S.ravel(), H.ravel()])


# ============================================================
# matplotlib helper: equal-aspect 3D (plotly aspectmode="data")
# ============================================================
def _set_axes_equal_3d(ax):
    """Give a 3D axes equal unit lengths on all axes (a data-aspect cube),
    the matplotlib equivalent of plotly's aspectmode='data'."""
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_mid, y_mid, z_mid = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    half = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(x_mid - half, x_mid + half)
    ax.set_ylim3d(y_mid - half, y_mid + half)
    ax.set_zlim3d(z_mid - half, z_mid + half)
    try:
        ax.set_box_aspect((1, 1, 1))   # matplotlib >= 3.3
    except Exception:
        pass


# ============================================================
# matplotlib: literal unrolled sheet, rolled sheet, inverse recovery
# ============================================================
def swiss_roll(
    width=18.0,
    height=10.0,
    theta_min=1.5*np.pi,
    r0=0.5,
    b=0.6,
    n_points=5000,
    seed=0,
    plot=True,
    save_path=None,   # optional: write the figure to a file (e.g. "swiss_roll.png")
):
    roll_map = make_isometric_swiss_roll_map_from_sheet(
        width=width, height=height, theta_min=theta_min, r0=r0, b=b,
    )

    E_ix = sample_sheet_random(n_points, width=width, height=height, seed=seed)
    XYZ, theta = sheet_to_swiss_roll(E_ix, roll_map)
    SH_back = swiss_roll_to_sheet(XYZ, roll_map)
    color = E_ix[:, 0]  # color by sheet-length coordinate

    if plot:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax3 = fig.add_subplot(1, 3, 3)

        cmap, s2d, s3d = "viridis", 6, 4

        # 1) literal unrolled sheet
        ax1.scatter(E_ix[:, 0], E_ix[:, 1], s=s2d, c=color, cmap=cmap)
        ax1.set_title(f"Literal unrolled sheet ({width:.1f} × {height:.1f})")
        ax1.set_xlabel("sheet coordinate s")
        ax1.set_ylabel("sheet coordinate h")
        ax1.set_aspect("equal", adjustable="box")   # plotly scaleanchor x/y ratio 1

        # 2) rolled sheet in 3D
        ax2.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], s=s3d, c=color, cmap=cmap)
        ax2.set_title("Rolled sheet in 3D")
        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
        _set_axes_equal_3d(ax2)                      # plotly aspectmode="data"

        # 3) inverse-unrolled recovery
        ax3.scatter(SH_back[:, 0], SH_back[:, 1], s=s2d, c=color, cmap=cmap)
        ax3.set_title("Inverse-unrolled recovery")
        ax3.set_xlabel("sheet coordinate s")
        ax3.set_ylabel("sheet coordinate h")
        ax3.set_aspect("equal", adjustable="box")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    err = np.abs(E_ix - SH_back)
    print("sheet size:", (roll_map["width"], roll_map["height"]))
    print("theta range:", (roll_map["theta_min"], roll_map["theta_max"]))
    print("max round-trip error:", err.max())
    print("mean round-trip error:", err.mean())

    return roll_map, E_ix, XYZ, SH_back, color


# Example:
# roll_map, E_ix, XYZ, SH_back, color = swiss_roll(
#     width=60.0, height=10.0, r0=0.5, b=0.6, n_points=4000, seed=1, plot=True)
# print(color.shape, E_ix.shape, XYZ.shape)
