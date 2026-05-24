# -*- coding: utf-8 -*-
"""
Config-driven experiments for Residual-Controlled ADMM (RC-ADMM).

Paper-oriented design:
  Experiment 1: finite-budget solver performance on mixed-cone CLP benchmarks,
                small/medium/hard, multiple K values.
  Experiment 2: ablation studies: controlled parameters, controller type,
                algorithm-unrolling choice, and optional safeguards.

Important choices:
  - Pure CONFIG dictionary. No command-line parser.
  - The default benchmark is a mixed-cone product set containing nonnegative,
    SOC, rotated-SOC, and PSD blocks simultaneously.  The older single-cone
    QP-lift/SOCP/SDP cases are still supported for debugging, but the paper
    experiment should use problem = "mixed".
  - Main metrics are not a weighted merit. CSV reports objective gap,
    equality violation, cone violation, solution distance, runtime, and divergence.
  - Strong baselines include oracle-grid ADMM, spectral adaptive ADMM,
    safeguarded Anderson/DRS-style acceleration, and a stabilized learned PDHG proxy.

The stabilized learned PDHG implementation is a fair experimental proxy under the
same data and residual-feature interface, not an exact reproduction of DYS-Net,
LISCO, or any other specific paper. Replace it with official implementations if
required for final comparison.
"""

import os
import time
import math
import copy
import warnings
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 0. Global config
# ============================================================

CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
    "root_out_dir": "./rc_admm_mixed_outputs_v4",
    "quick": False,
    "overwrite_data": False,

    # Experiments to run.  Edit lists to add/remove groups.
    "experiments": [
        {"name": "exp1", "enabled": True,
         "problems": ["mixed"],
         "scales": ["small", "medium", "hard"]},
        {"name": "exp2", "enabled": True,
         "problems": ["mixed"],
         "scales": ["hard"]},
    ],

    # Multiple-K evaluation for Experiment 1.
    "k_values": [5, 10, 15, 20, 40],

    # Main methods for Experiment 1.
    "exp1_methods": [
        "Fixed-ADMM",
        "OracleGrid-ADMM",
        "Spectral-AADMM",
        "DRE-Anderson-DRS",
        "Stable-Learned-PDHG",
        "RC-ADMM",
    ],

    # Ablation variants for Experiment 2.
    "exp2_parameter_variants": [
        {"name": "rho-only", "learned": ["rho"]},
        {"name": "alpha-only", "learned": ["alpha"]},
        {"name": "beta-only", "learned": ["beta"]},
        {"name": "rho-alpha", "learned": ["rho", "alpha"]},
        {"name": "rho-beta", "learned": ["rho", "beta"]},
        {"name": "alpha-beta", "learned": ["alpha", "beta"]},
        {"name": "full", "learned": ["rho", "alpha", "beta"]},
    ],
    "exp2_controller_variants": [
        {"name": "layerwise", "controller": "layerwise"},
        {"name": "mlp-current", "controller": "mlp_current"},
        {"name": "gru", "controller": "gru"},
        {"name": "lstm", "controller": "lstm"},
    ],
    "exp2_algorithm_variants": [
        {"name": "RC-ADMM", "kind": "admm"},
        {"name": "Stable-Learned-PDHG", "kind": "pdhg"},
        {"name": "RC-DRS-alpha-beta", "kind": "drs_ab"},
    ],
    "exp2_safeguard_variants": [
        {"name": "default", "envelope": True, "growth": True},
        {"name": "no-envelope", "envelope": False, "growth": True},
        {"name": "no-growth", "envelope": True, "growth": False},
        {"name": "no-envelope-no-growth", "envelope": False, "growth": False},
    ],

    # Dataset sizes.
    "n_train": 2000,
    "n_val": 400,
    "n_test": 400,
    "batch_size": 1024,

    # Solver/controller training.
    "epochs": 80,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "grad_clip": 5.0,
    "hidden_dim": 64,
    "feature_dim": 7,

    # Base parameters and action ranges.
    "rho_base": 1.0,
    "alpha_base": 1.6,
    "beta_base": 0.3,
    "rho_min": 1e-4,
    "rho_max": 1e4,
    "alpha_min": 0.2,
    "alpha_max": 1.9,
    "beta_min": 1e-5,
    "beta_max": 1e2,
    "tau_min": 1e-4,
    "tau_max": 10.0,
    "sigma_min": 1e-4,
    "sigma_max": 10.0,

    # Stability parameters for strong baselines.
    "spectral_growth": 2.0,
    "anderson_omega": 0.25,
    "anderson_accept_tol": 1.05,
    "pdhg_safety": 0.95,
    "pdhg_log_scale": 0.5,
    "pdhg_theta_max": 0.8,

    # Envelope/growth defaults.
    "use_envelope": True,
    "use_growth": True,
    "delta0": 3.0,
    "k0": 20.0,
    "p_decay": 2.0,
    "alpha_delta_scale": 0.35,
    "chi_rho": 5.0,
    "chi_beta": 5.0,

    # Training loss weights only; not used as final reported metric.
    "lam_eq": 10.0,
    "lam_cone": 10.0,
    "lam_obj": 1.0,
    "lam_move": 0.1,
    "lam_smooth": 1e-3,

    # Oracle-grid parameter choices.
    "tune_subset": 128,
    "tune_rho_grid": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    "tune_alpha_grid": [1.0, 1.3, 1.6, 1.8],
    "tune_beta_grid": [0.03, 0.1, 0.3, 1.0, 3.0],

    # Runtime measurement.
    "runtime_repeats": 3,
    "runtime_warmup": 1,
}

# ============================================================
# 1. Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def safe_mean(x: torch.Tensor) -> float:
    x = x.detach().flatten()
    mask = torch.isfinite(x)
    if not torch.any(mask):
        return float("nan")
    return float(torch.mean(x[mask]).cpu())


def safe_median(x: torch.Tensor) -> float:
    x = x.detach().flatten()
    mask = torch.isfinite(x)
    if not torch.any(mask):
        return float("nan")
    return float(torch.median(x[mask]).cpu())


def copy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)


def apply_problem_scale(cfg: Dict[str, Any], problem: str, scale: str) -> Dict[str, Any]:
    cfg = copy_cfg(cfg)
    cfg["problem"] = problem
    cfg["scale"] = scale

    if problem == "qp":
        # QP-lift uses d scalar variables x_i, plus t_i and s_i, total n=3d.
        if scale == "small":
            cfg.update({"qp_dim": 32, "m_rand": 12, "condition": 20.0, "k_train": 10})
        elif scale == "medium":
            cfg.update({"qp_dim": 64, "m_rand": 24, "condition": 80.0, "k_train": 15})
        elif scale == "hard":
            cfg.update({"qp_dim": 96, "m_rand": 40, "condition": 300.0, "k_train": 20})
        cfg["n"] = 3 * cfg["qp_dim"]
        cfg["m"] = cfg["m_rand"] + cfg["qp_dim"]  # random Ax=b plus s_i=1.

    elif problem == "socp":
        if scale == "small":
            cfg.update({"soc_blocks": 8, "soc_dim": 8, "m": 32, "condition": 20.0, "k_train": 10})
        elif scale == "medium":
            cfg.update({"soc_blocks": 16, "soc_dim": 8, "m": 64, "condition": 80.0, "k_train": 15})
        elif scale == "hard":
            cfg.update({"soc_blocks": 20, "soc_dim": 8, "m": 128, "condition": 200.0, "k_train": 10})
        cfg["n"] = cfg["soc_blocks"] * cfg["soc_dim"]

    elif problem == "sdp":
        if scale == "small":
            cfg.update({"psd_dim": 8, "m": 24, "condition": 20.0, "k_train": 10})
        elif scale == "medium":
            cfg.update({"psd_dim": 12, "m": 64, "condition": 80.0, "k_train": 15})
        elif scale == "hard":
            cfg.update({"psd_dim": 16, "m": 128, "condition": 200.0, "k_train": 20})
        cfg["n"] = cfg["psd_dim"] * cfg["psd_dim"]

    elif problem == "mixed":
        # Mixed-cone CLP: R_+^nnonneg x SOC^q x RSOC^r x S_+^p.
        # This is the default paper benchmark because realistic conic layers
        # rarely contain only one cone family.
        if scale == "small":
            cfg.update({
                "nonneg_dim": 24,
                "soc_blocks": 4, "soc_dim": 6,
                "rot_blocks": 12,
                "psd_dim": 6,
                "m": 48, "condition": 40.0, "k_train": 10,
            })
        elif scale == "medium":
            cfg.update({
                "nonneg_dim": 48,
                "soc_blocks": 8, "soc_dim": 8,
                "rot_blocks": 24,
                "psd_dim": 8,
                "m": 96, "condition": 100.0, "k_train": 15,
            })
        elif scale == "hard":
            cfg.update({
                "nonneg_dim": 72,
                "soc_blocks": 12, "soc_dim": 8,
                "rot_blocks": 36,
                "psd_dim": 10,
                "m": 160, "condition": 250.0, "k_train": 20,
            })
        else:
            raise ValueError(f"Unknown scale: {scale}")
        cfg["n"] = (
            cfg["nonneg_dim"]
            + cfg["soc_blocks"] * cfg["soc_dim"]
            + 3 * cfg["rot_blocks"]
            + cfg["psd_dim"] * cfg["psd_dim"]
        )
    else:
        raise ValueError(f"Unknown problem type: {problem}")

    if cfg.get("quick", False):
        cfg["n_train"], cfg["n_val"], cfg["n_test"] = 128, 48, 48
        cfg["epochs"] = 5
        cfg["k_values"] = [5, 10]
        if problem == "sdp" and cfg.get("psd_dim", 8) > 8:
            cfg["psd_dim"] = 8
            cfg["n"] = 64
            cfg["m"] = min(cfg["m"], 24)
        if problem == "mixed":
            cfg.update({
                "nonneg_dim": 12,
                "soc_blocks": 2, "soc_dim": 5,
                "rot_blocks": 6,
                "psd_dim": 4,
                "m": min(cfg["m"], 24),
                "condition": min(float(cfg["condition"]), 40.0),
            })
            cfg["n"] = (
                cfg["nonneg_dim"]
                + cfg["soc_blocks"] * cfg["soc_dim"]
                + 3 * cfg["rot_blocks"]
                + cfg["psd_dim"] * cfg["psd_dim"]
            )
    return cfg

# ============================================================
# 2. Linear operators and cone projections
# ============================================================

def make_A(m: int, n: int, condition: float, rng: np.random.Generator) -> np.ndarray:
    Gu = rng.standard_normal((m, m))
    U, _ = np.linalg.qr(Gu)
    Gv = rng.standard_normal((n, m))
    V, _ = np.linalg.qr(Gv)
    s = np.ones(m) if condition <= 1.5 else np.geomspace(1.0, 1.0 / condition, m)
    return (U @ np.diag(s) @ V.T).astype(np.float32)


def make_projection_matrix(A: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    M = A @ A.T + ridge * np.eye(A.shape[0], dtype=np.float32)
    return np.linalg.solve(M, A).T.astype(np.float32)  # A^T(AA^T)^-1


def proj_affine(v: torch.Tensor, A: torch.Tensor, P: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return v - (v @ A.t() - b) @ P.t()


def proj_soc_block_torch(block: torch.Tensor) -> torch.Tensor:
    # block shape (..., d), standard SOC: t >= ||v||.
    t = block[..., 0:1]
    v = block[..., 1:]
    r = torch.linalg.norm(v, dim=-1, keepdim=True)
    r_safe = torch.clamp(r, min=1e-12)
    inside = r <= t
    zero = r <= -t
    new_t = 0.5 * (r + t)
    new_v = new_t * v / r_safe
    proj = torch.cat([new_t, new_v], dim=-1)
    out = torch.where(inside.expand_as(block), block, proj)
    out = torch.where(zero.expand_as(block), torch.zeros_like(block), out)
    return out


def proj_rotated_soc_3_torch(block: torch.Tensor) -> torch.Tensor:
    # Rotated cone in R^3: 2uv >= w^2, u>=0, v>=0.
    # Orthogonal-scaled transform to SOC: y=(u+v, u-v, sqrt(2)w).
    u = block[..., 0:1]
    v = block[..., 1:2]
    w = block[..., 2:3]
    y = torch.cat([u + v, u - v, math.sqrt(2.0) * w], dim=-1)
    py = proj_soc_block_torch(y)
    y0, y1, y2 = py[..., 0:1], py[..., 1:2], py[..., 2:3]
    pu = 0.5 * (y0 + y1)
    pv = 0.5 * (y0 - y1)
    pw = y2 / math.sqrt(2.0)
    return torch.cat([pu, pv, pw], dim=-1)


def proj_cone_torch(x: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    problem = cfg["problem"]
    if problem == "qp":
        # Variables are ordered [t_1,s_1,x_1,t_2,s_2,x_2,...].
        d = cfg["qp_dim"]
        X = x.reshape(x.shape[0], d, 3)
        return proj_rotated_soc_3_torch(X).reshape(x.shape[0], 3 * d)

    if problem == "socp":
        B = x.shape[0]
        q, dim = cfg["soc_blocks"], cfg["soc_dim"]
        X = x.reshape(B, q, dim)
        return proj_soc_block_torch(X).reshape(B, q * dim)

    if problem == "sdp":
        B, p = x.shape[0], cfg["psd_dim"]
        X = x.reshape(B, p, p)
        X = 0.5 * (X + X.transpose(-1, -2))
        eig, V = torch.linalg.eigh(X)
        eig_pos = torch.clamp(eig, min=0.0)
        Y = V @ torch.diag_embed(eig_pos) @ V.transpose(-1, -2)
        Y = 0.5 * (Y + Y.transpose(-1, -2))
        return Y.reshape(B, p * p)

    if problem == "mixed":
        B = x.shape[0]
        sl = mixed_slices(cfg)
        parts = []
        # Nonnegative orthant block.
        x_nonneg = x[:, sl["nonneg"]]
        parts.append(torch.clamp(x_nonneg, min=0.0))

        # Standard SOC product block.
        q, dim = int(cfg["soc_blocks"]), int(cfg["soc_dim"])
        x_soc = x[:, sl["soc"]].reshape(B, q, dim)
        parts.append(proj_soc_block_torch(x_soc).reshape(B, q * dim))

        # Rotated SOC product block, represented by 3D rotated cones.
        r = int(cfg["rot_blocks"])
        x_rot = x[:, sl["rot"]].reshape(B, r, 3)
        parts.append(proj_rotated_soc_3_torch(x_rot).reshape(B, 3 * r))

        # PSD block.
        p = int(cfg["psd_dim"])
        X = x[:, sl["psd"]].reshape(B, p, p)
        X = 0.5 * (X + X.transpose(-1, -2))
        eig, V = torch.linalg.eigh(X)
        eig_pos = torch.clamp(eig, min=0.0)
        Y = V @ torch.diag_embed(eig_pos) @ V.transpose(-1, -2)
        Y = 0.5 * (Y + Y.transpose(-1, -2))
        parts.append(Y.reshape(B, p * p))
        return torch.cat(parts, dim=1)

    raise ValueError(problem)

def cone_distance_torch(x: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    px = proj_cone_torch(x, cfg)
    return torch.linalg.norm(x - px, dim=1) / (1.0 + torch.linalg.norm(x, dim=1))


def mixed_slices(cfg: Dict[str, Any]) -> Dict[str, slice]:
    """Return block slices for the mixed-cone variable."""
    start = 0
    n_nonneg = int(cfg["nonneg_dim"])
    s_nonneg = slice(start, start + n_nonneg)
    start += n_nonneg

    n_soc = int(cfg["soc_blocks"]) * int(cfg["soc_dim"])
    s_soc = slice(start, start + n_soc)
    start += n_soc

    n_rot = 3 * int(cfg["rot_blocks"])
    s_rot = slice(start, start + n_rot)
    start += n_rot

    n_psd = int(cfg["psd_dim"]) * int(cfg["psd_dim"])
    s_psd = slice(start, start + n_psd)
    start += n_psd

    if start != int(cfg["n"]):
        raise ValueError(f"Mixed-cone dimension mismatch: slices end at {start}, n={cfg['n']}")
    return {"nonneg": s_nonneg, "soc": s_soc, "rot": s_rot, "psd": s_psd}

# ============================================================
# 3. KKT-consistent dataset generation
# ============================================================

def sample_nonnegative_complementarity(n: int, active_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample x>=0, s>=0, and x_i s_i=0."""
    x = rng.uniform(0.1, 2.0, size=n).astype(np.float32)
    s = np.zeros(n, dtype=np.float32)
    active = rng.random(n) < active_frac
    x[active] = 0.0
    s[active] = rng.uniform(0.1, 2.0, size=int(np.sum(active))).astype(np.float32)
    return x, s


def inv_rotated_soc_transform(y: np.ndarray) -> np.ndarray:
    """Inverse of y=(u+v, u-v, sqrt(2)w)."""
    y0, y1, y2 = y[..., 0], y[..., 1], y[..., 2]
    u = 0.5 * (y0 + y1)
    v = 0.5 * (y0 - y1)
    w = y2 / math.sqrt(2.0)
    return np.stack([u, v, w], axis=-1)


def sample_rotated_soc_complementarity(r: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample complementary pairs in a product of 3D rotated SOCs.

    We sample complementary pairs in a standard SOC after the scaled orthogonal
    transform y=(u+v,u-v,sqrt(2)w), then map them back.
    """
    y, sy = sample_soc_complementarity(r, 3, hard=True, rng=rng)
    y = y.reshape(r, 3)
    sy = sy.reshape(r, 3)
    x = inv_rotated_soc_transform(y).reshape(-1).astype(np.float32)
    s = inv_rotated_soc_transform(sy).reshape(-1).astype(np.float32)
    return x, s


def sample_soc_complementarity(q: int, dim: int, hard: bool, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    xs, ss = [], []
    for _ in range(q):
        v = rng.standard_normal(dim - 1)
        nv = np.linalg.norm(v) + 1e-12
        if hard or rng.random() < 0.60:
            # Boundary point and complementary boundary slack.
            t = nv
            a = rng.uniform(0.1, 2.0)
            x = np.concatenate([[t], v])
            s = np.concatenate([[a * nv], -a * v])
        else:
            # Strict interior point with zero slack.
            x = np.concatenate([[nv + rng.uniform(0.2, 1.5)], v])
            s = np.zeros(dim)
        xs.append(x)
        ss.append(s)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ss).astype(np.float32)


def sample_rotated_qp_lift(d: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # Variables per i: (t_i, s_i, x_i), with s_i=1 and 2 t_i s_i >= x_i^2.
    # This represents t_i >= x_i^2/2.
    blocks_x, blocks_s = [], []
    x_vals = rng.normal(0.0, 1.0, size=d)
    for xi in x_vals:
        t = 0.5 * xi * xi
        s = 1.0
        # Complementary slack on rotated cone: (1, 0.5*x^2, -x) times mu.
        mu = rng.uniform(0.1, 2.0)
        z = np.array([t, s, xi], dtype=np.float32)
        slack = mu * np.array([1.0, 0.5 * xi * xi, -xi], dtype=np.float32)
        blocks_x.append(z)
        blocks_s.append(slack)
    return np.concatenate(blocks_x).astype(np.float32), np.concatenate(blocks_s).astype(np.float32)


def rand_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.standard_normal((p, p))
    U, _ = np.linalg.qr(G)
    return U


def sample_psd_complementarity(p: int, hard: bool, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    U = rand_orthogonal(p, rng)
    r = max(1, p // (3 if hard else 2))
    eig_x = np.zeros(p)
    eig_s = np.zeros(p)
    eig_x[:r] = rng.uniform(0.2, 2.0, size=r)
    eig_s[r:] = rng.uniform(0.2, 2.0, size=p - r)
    X = U @ np.diag(eig_x) @ U.T
    S = U @ np.diag(eig_s) @ U.T
    X = 0.5 * (X + X.T)
    S = 0.5 * (S + S.T)
    return X.reshape(-1).astype(np.float32), S.reshape(-1).astype(np.float32)


def sample_mixed_complementarity(cfg: Dict[str, Any], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a KKT-complementary pair for the mixed cone."""
    hard = cfg["scale"] == "hard"
    x_nonneg, s_nonneg = sample_nonnegative_complementarity(
        int(cfg["nonneg_dim"]), active_frac=0.35 if hard else 0.25, rng=rng
    )
    x_soc, s_soc = sample_soc_complementarity(
        int(cfg["soc_blocks"]), int(cfg["soc_dim"]), hard=hard, rng=rng
    )
    x_rot, s_rot = sample_rotated_soc_complementarity(int(cfg["rot_blocks"]), rng=rng)
    x_psd, s_psd = sample_psd_complementarity(int(cfg["psd_dim"]), hard=hard, rng=rng)
    x = np.concatenate([x_nonneg, x_soc, x_rot, x_psd]).astype(np.float32)
    s = np.concatenate([s_nonneg, s_soc, s_rot, s_psd]).astype(np.float32)
    return x, s


def make_problem_matrix(cfg: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    if cfg["problem"] == "qp":
        d = cfg["qp_dim"]
        A_x = make_A(cfg["m_rand"], d, cfg["condition"], rng)
        A = np.zeros((cfg["m_rand"] + d, 3 * d), dtype=np.float32)
        # x_i is the third coordinate of each rotated block.
        for i in range(d):
            A[:cfg["m_rand"], 3 * i + 2] = A_x[:, i]
            A[cfg["m_rand"] + i, 3 * i + 1] = 1.0  # s_i = 1.
        return A
    # For SOCP, SDP, and mixed-cone benchmarks, use one dense equality map.
    return make_A(cfg["m"], cfg["n"], cfg["condition"], rng)


def generate_dataset(cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    ensure_dir(cfg["out_dir"])
    path = os.path.join(cfg["out_dir"], f"data_{cfg['problem']}_{cfg['scale']}.pt")
    if os.path.exists(path) and not cfg.get("overwrite_data", False):
        print(f"[Data] load {path}")
        return torch.load(path, map_location="cpu")

    rng = np.random.default_rng(cfg["seed"] + 123)
    A = make_problem_matrix(cfg, rng)
    P = make_projection_matrix(A)

    def sample_one() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        problem = cfg["problem"]
        if problem == "qp":
            x_star, slack = sample_rotated_qp_lift(cfg["qp_dim"], rng)
        elif problem == "socp":
            x_star, slack = sample_soc_complementarity(cfg["soc_blocks"], cfg["soc_dim"], cfg["scale"] == "hard", rng)
        elif problem == "sdp":
            x_star, slack = sample_psd_complementarity(cfg["psd_dim"], cfg["scale"] == "hard", rng)
        elif problem == "mixed":
            x_star, slack = sample_mixed_complementarity(cfg, rng)
        else:
            raise ValueError(problem)

        y = rng.standard_normal(A.shape[0]).astype(np.float32)
        c_raw = slack - A.T @ y
        scale = np.linalg.norm(c_raw) + 1e-8
        c = (c_raw / scale).astype(np.float32)
        b = (A @ x_star).astype(np.float32)
        p_star = float(c @ x_star)
        return b, c, x_star.astype(np.float32), p_star

    def split(N: int):
        bs, cs, xs, ps = [], [], [], []
        for _ in range(N):
            b, c, x, p = sample_one()
            bs.append(b); cs.append(c); xs.append(x); ps.append(p)
        return np.stack(bs), np.stack(cs), np.stack(xs), np.asarray(ps, dtype=np.float32)

    btr, ctr, xtr, ptr = split(cfg["n_train"])
    bva, cva, xva, pva = split(cfg["n_val"])
    bte, cte, xte, pte = split(cfg["n_test"])

    data = {
        "A": torch.tensor(A),
        "P": torch.tensor(P),
        "b_train": torch.tensor(btr), "c_train": torch.tensor(ctr), "xstar_train": torch.tensor(xtr), "pstar_train": torch.tensor(ptr),
        "b_val": torch.tensor(bva), "c_val": torch.tensor(cva), "xstar_val": torch.tensor(xva), "pstar_val": torch.tensor(pva),
        "b_test": torch.tensor(bte), "c_test": torch.tensor(cte), "xstar_test": torch.tensor(xte), "pstar_test": torch.tensor(pte),
    }
    torch.save(data, path)
    print(f"[Data] saved {path}")
    return data


def make_loader(data: Dict[str, torch.Tensor], split: str, cfg: Dict[str, Any], shuffle: bool) -> DataLoader:
    ds = TensorDataset(data[f"b_{split}"], data[f"c_{split}"], data[f"xstar_{split}"], data[f"pstar_{split}"])
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, drop_last=False)

# ============================================================
# 4. Metrics and features
# ============================================================

def objective(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.sum(c * z, dim=1)


def compute_metrics(out: Dict[str, torch.Tensor], A, b, c, xstar, pstar, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    z = out["z"]
    obj = objective(z, c)
    denom = 1.0 + torch.abs(pstar)
    signed_gap = (obj - pstar) / denom
    eq_vio = torch.linalg.norm(z @ A.t() - b, dim=1) / (1.0 + torch.linalg.norm(b, dim=1))
    cone_vio = cone_distance_torch(z, cfg)
    sol_dist = torch.linalg.norm(z - xstar, dim=1) / (1.0 + torch.linalg.norm(xstar, dim=1))
    div = (~torch.isfinite(obj)) | (~torch.isfinite(eq_vio)) | (~torch.isfinite(cone_vio)) | (eq_vio > 1e6) | (cone_vio > 1e6)
    return {
        "obj": obj,
        "obj_gap_signed": signed_gap,
        "obj_gap_pos": torch.relu(signed_gap),
        "obj_gap_abs": torch.abs(signed_gap),
        "eq_vio": eq_vio,
        "cone_vio": cone_vio,
        "sol_dist": sol_dist,
        "div": div.float(),
    }


def feature_vector(A, b, c, x, z, z_prev, rho_prev, cfg: Dict[str, Any]) -> torch.Tensor:
    eq = torch.linalg.norm(z @ A.t() - b, dim=1) / (1.0 + torch.linalg.norm(b, dim=1))
    cone = cone_distance_torch(z, cfg)
    con = torch.linalg.norm(x - z, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
    dz = torch.linalg.norm(z - z_prev, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
    mov = rho_prev * dz
    obj_abs = torch.abs(objective(z, c)) / (1.0 + torch.linalg.norm(c, dim=1) * torch.linalg.norm(z, dim=1))
    c_norm = torch.linalg.norm(c, dim=1)
    x_norm = torch.linalg.norm(x, dim=1)
    return torch.stack([
        torch.log1p(con), torch.log1p(eq), torch.log1p(cone), torch.log1p(mov),
        torch.log1p(dz), torch.log1p(obj_abs), torch.log1p(c_norm / (1.0 + x_norm))
    ], dim=1)


def training_loss(out, A, b, c, cfg: Dict[str, Any]) -> torch.Tensor:
    z = out["z"]
    z_prev = out.get("z_prev", z)
    obj = objective(z, c)
    eq = torch.linalg.norm(z @ A.t() - b, dim=1) / (1.0 + torch.linalg.norm(b, dim=1))
    cone = cone_distance_torch(z, cfg)
    move = torch.linalg.norm(z - z_prev, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
    obj_scaled = obj / (1.0 + torch.mean(torch.abs(obj.detach())))
    loss = cfg["lam_obj"] * torch.mean(obj_scaled) + cfg["lam_eq"] * torch.mean(eq ** 2) + cfg["lam_cone"] * torch.mean(cone ** 2) + cfg["lam_move"] * torch.mean(move ** 2)

    if "params" in out and len(out["params"]) >= 2:
        smooth = 0.0
        for key in out["params"][0].keys():
            vals = torch.stack([p[key] for p in out["params"]], dim=0)
            if torch.all(vals > 0):
                smooth = smooth + torch.mean((torch.log(vals[1:] + 1e-12) - torch.log(vals[:-1] + 1e-12)) ** 2)
            else:
                smooth = smooth + torch.mean((vals[1:] - vals[:-1]) ** 2)
        loss = loss + cfg["lam_smooth"] * smooth
    return loss

# ============================================================
# 5. Policies and solver models
# ============================================================

class CausalPolicy(nn.Module):
    def __init__(self, cfg: Dict[str, Any], controller: str, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.controller = controller
        f, h = cfg["feature_dim"], cfg["hidden_dim"]
        if controller == "layerwise":
            self.raw = nn.Parameter(torch.zeros(512, action_dim))
        elif controller == "mlp_current":
            self.net = nn.Sequential(nn.Linear(f, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, action_dim))
        elif controller == "gru":
            self.cell = nn.GRUCell(f, h)
            self.head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, action_dim))
        elif controller == "lstm":
            self.cell = nn.LSTMCell(f, h)
            self.head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, action_dim))
        else:
            raise ValueError(controller)

    def init_state(self, batch_size: int, device):
        h = self.cfg["hidden_dim"]
        if self.controller == "gru":
            return torch.zeros(batch_size, h, device=device)
        if self.controller == "lstm":
            return (torch.zeros(batch_size, h, device=device), torch.zeros(batch_size, h, device=device))
        return None

    def forward(self, phi: torch.Tensor, k: int, state):
        B = phi.shape[0]
        if self.controller == "layerwise":
            raw = self.raw[min(k, self.raw.shape[0] - 1)].unsqueeze(0).expand(B, -1)
            return raw, state
        if self.controller == "mlp_current":
            return self.net(phi), state
        if self.controller == "gru":
            h = self.cell(phi, state)
            return self.head(h), h
        if self.controller == "lstm":
            h, c = self.cell(phi, state)
            return self.head(h), (h, c)
        raise RuntimeError


def map_admm_params(raw, k: int, rho_prev, beta_prev, cfg, learned, envelope: bool, growth: bool):
    """
    Map controller raw outputs to admissible ADMM parameters.

    If a parameter is not learned, it is still returned as a batch Tensor.
    This is necessary for variants such as Learned-DRS, alpha/beta-only,
    and other partial-parameter ablations.
    """
    B, device = raw.shape[0], raw.device
    dtype = raw.dtype

    learn_rho = "rho" in learned
    learn_alpha = "alpha" in learned
    learn_beta = "beta" in learned

    rho_base = torch.full((B,), float(cfg["rho_base"]), device=device, dtype=dtype)
    alpha_base = torch.full((B,), float(cfg["alpha_base"]), device=device, dtype=dtype)
    beta_base = torch.full((B,), float(cfg["beta_base"]), device=device, dtype=dtype)

    if envelope:
        q = torch.tanh(raw)
        delta = float(cfg["delta0"]) / (
            1.0 + (float(k) / float(cfg["k0"])) ** float(cfg["p_decay"])
        )

        if learn_rho:
            rho = torch.exp(torch.log(rho_base) + delta * q[:, 0])
        else:
            rho = rho_base

        if learn_alpha:
            alpha = alpha_base + float(cfg["alpha_delta_scale"]) * delta * q[:, 1]
        else:
            alpha = alpha_base

        if learn_beta:
            beta = torch.exp(torch.log(beta_base) + delta * q[:, 2])
        else:
            beta = beta_base

    else:
        sig = torch.sigmoid(raw)

        rho_box = torch.exp(
            math.log(float(cfg["rho_min"]))
            + sig[:, 0] * (
                math.log(float(cfg["rho_max"]))
                - math.log(float(cfg["rho_min"]))
            )
        )

        alpha_box = (
            float(cfg["alpha_min"])
            + sig[:, 1] * (
                float(cfg["alpha_max"])
                - float(cfg["alpha_min"])
            )
        )

        beta_box = torch.exp(
            math.log(float(cfg["beta_min"]))
            + sig[:, 2] * (
                math.log(float(cfg["beta_max"]))
                - math.log(float(cfg["beta_min"]))
            )
        )

        rho = rho_box if learn_rho else rho_base
        alpha = alpha_box if learn_alpha else alpha_base
        beta = beta_box if learn_beta else beta_base

    rho = torch.clamp(rho, float(cfg["rho_min"]), float(cfg["rho_max"]))
    alpha = torch.clamp(alpha, float(cfg["alpha_min"]), float(cfg["alpha_max"]))
    beta = torch.clamp(beta, float(cfg["beta_min"]), float(cfg["beta_max"]))

    if growth:
        rho = torch.minimum(
            torch.maximum(rho, rho_prev / float(cfg["chi_rho"])),
            rho_prev * float(cfg["chi_rho"])
        )
        beta = torch.minimum(
            torch.maximum(beta, beta_prev / float(cfg["chi_beta"])),
            beta_prev * float(cfg["chi_beta"])
        )

        rho = torch.clamp(rho, float(cfg["rho_min"]), float(cfg["rho_max"]))
        beta = torch.clamp(beta, float(cfg["beta_min"]), float(cfg["beta_max"]))

    return rho, alpha, beta


class RCADMM(nn.Module):
    def __init__(self, cfg: Dict[str, Any], controller="gru", learned=("rho", "alpha", "beta"), envelope=True, growth=True):
        super().__init__()
        self.cfg = cfg
        self.learned = list(learned)
        self.envelope = envelope
        self.growth = growth
        self.policy = CausalPolicy(cfg, controller, 3)

    def forward(self, A, P, b, c, K: int):
        cfg = self.cfg
        B, n = c.shape
        device = c.device
        z = torch.zeros(B, n, device=device)
        u = torch.zeros(B, n, device=device)
        x = proj_affine(z - u, A, P, b)
        z_prev = z.clone()
        rho_prev = torch.full((B,), cfg["rho_base"], device=device)
        beta_prev = torch.full((B,), cfg["beta_base"], device=device)
        cbar = c / (torch.linalg.norm(c, dim=1, keepdim=True) + 1e-8)
        state = self.policy.init_state(B, device)
        params = []
        for k in range(K):
            phi = feature_vector(A, b, c, x, z, z_prev, rho_prev, cfg)
            raw, state = self.policy(phi, k, state)
            rho, alpha, beta = map_admm_params(raw, k, rho_prev, beta_prev, cfg, self.learned, self.envelope, self.growth)
            w = z - u - beta.unsqueeze(1) * cbar
            x_next = proj_affine(w, A, P, b)
            x_bar = alpha.unsqueeze(1) * x_next + (1.0 - alpha).unsqueeze(1) * z
            z_next = proj_cone_torch(x_bar + u, cfg)
            u_next = u + x_bar - z_next
            z_prev = z
            x, z, u = x_next, z_next, u_next
            rho_prev, beta_prev = rho, beta
            params.append({"rho": rho, "alpha": alpha, "beta": beta})
        return {"x": x, "z": z, "u": u, "z_prev": z_prev, "params": params}


class FixedADMM(nn.Module):
    def __init__(self, cfg: Dict[str, Any], rho=None, alpha=None, beta=None):
        super().__init__()
        self.cfg = cfg
        self.rho = cfg["rho_base"] if rho is None else float(rho)
        self.alpha = cfg["alpha_base"] if alpha is None else float(alpha)
        self.beta = cfg["beta_base"] if beta is None else float(beta)

    def forward(self, A, P, b, c, K: int):
        cfg = self.cfg
        B, n = c.shape
        device = c.device
        z = torch.zeros(B, n, device=device)
        u = torch.zeros(B, n, device=device)
        x = proj_affine(z - u, A, P, b)
        z_prev = z.clone()
        cbar = c / (torch.linalg.norm(c, dim=1, keepdim=True) + 1e-8)
        alpha = torch.full((B,), self.alpha, device=device)
        beta = torch.full((B,), self.beta, device=device)
        for _ in range(K):
            x_next = proj_affine(z - u - beta.unsqueeze(1) * cbar, A, P, b)
            x_bar = alpha.unsqueeze(1) * x_next + (1.0 - alpha).unsqueeze(1) * z
            z_next = proj_cone_torch(x_bar + u, cfg)
            u_next = u + x_bar - z_next
            z_prev = z
            x, z, u = x_next, z_next, u_next
        return {"x": x, "z": z, "u": u, "z_prev": z_prev}


class SpectralAADMM(FixedADMM):
    """Safeguarded spectral adaptive ADMM.

    The baseline updates rho by a Barzilai--Borwein style estimate, but the
    beta-parameterized implementation must also let rho affect the effective
    objective-drive step.  We therefore use beta = beta_ref * rho_ref / rho.
    A growth cap prevents large one-step rho jumps.
    """
    def forward(self, A, P, b, c, K: int):
        cfg = self.cfg
        B, n = c.shape
        device = c.device
        z = torch.zeros(B, n, device=device)
        u = torch.zeros(B, n, device=device)
        x = proj_affine(z - u, A, P, b)
        z_prev = z.clone()
        cbar = c / (torch.linalg.norm(c, dim=1, keepdim=True) + 1e-8)
        rho = torch.full((B,), self.rho, device=device)
        rho_ref = torch.full((B,), self.rho, device=device)
        beta_ref = torch.full((B,), self.beta, device=device)
        alpha = torch.full((B,), self.alpha, device=device)
        r_prev = None
        growth = float(cfg.get("spectral_growth", 2.0))
        for _ in range(K):
            beta = beta_ref * rho_ref / torch.clamp(rho, min=1e-8)
            beta = torch.clamp(beta, float(cfg["beta_min"]), float(cfg["beta_max"]))
            x_next = proj_affine(z - u - beta.unsqueeze(1) * cbar, A, P, b)
            x_bar = alpha.unsqueeze(1) * x_next + (1.0 - alpha).unsqueeze(1) * z
            z_next = proj_cone_torch(x_bar + u, cfg)
            u_next = u + x_bar - z_next
            r = x_next - z_next
            s = z_next - z
            if r_prev is not None:
                dr = r - r_prev
                ds = s
                num = torch.sum(dr * dr, dim=1) + 1e-8
                den = torch.abs(torch.sum(dr * ds, dim=1)) + 1e-8
                rho_bb = torch.clamp(num / den, float(cfg["rho_min"]), float(cfg["rho_max"]))
                rho_new = torch.sqrt(torch.clamp(rho * rho_bb, float(cfg["rho_min"]), float(cfg["rho_max"])))
                rho_new = torch.minimum(torch.maximum(rho_new, rho / growth), rho * growth)
                rho = torch.clamp(rho_new, float(cfg["rho_min"]), float(cfg["rho_max"]))
            r_prev = r.detach()
            z_prev = z
            x, z, u = x_next, z_next, u_next
        return {"x": x, "z": z, "u": u, "z_prev": z_prev}


class AndersonDRS(FixedADMM):
    """Safeguarded DRS/Anderson-style extrapolation on the ADMM state.

    The extrapolated step is accepted only if a simple residual monitor does not
    deteriorate relative to the base non-extrapolated step.  This avoids the
    catastrophic divergence often seen by unconstrained Anderson/inertial DRS on
    nonsmooth cone projections.
    """
    def __init__(self, cfg: Dict[str, Any], rho=None, alpha=None, beta=None, omega=None):
        super().__init__(cfg, rho, alpha, beta)
        self.omega = float(cfg.get("anderson_omega", 0.25) if omega is None else omega)

    def _step(self, A, P, b, cbar, z, u, alpha, beta, cfg):
        x_next = proj_affine(z - u - beta.unsqueeze(1) * cbar, A, P, b)
        x_bar = alpha.unsqueeze(1) * x_next + (1.0 - alpha).unsqueeze(1) * z
        z_next = proj_cone_torch(x_bar + u, cfg)
        u_next = u + x_bar - z_next
        return x_next, z_next, u_next

    def _monitor(self, A, b, z_new, z_old, cfg):
        eq = torch.linalg.norm(z_new @ A.t() - b, dim=1) / (1.0 + torch.linalg.norm(b, dim=1))
        cone = cone_distance_torch(z_new, cfg)
        step = torch.linalg.norm(z_new - z_old, dim=1) / (1.0 + torch.linalg.norm(z_new, dim=1))
        return eq + cone + 0.1 * step

    def forward(self, A, P, b, c, K: int):
        cfg = self.cfg
        B, n = c.shape
        device = c.device
        z = torch.zeros(B, n, device=device)
        u = torch.zeros(B, n, device=device)
        x = proj_affine(z - u, A, P, b)
        z_prev = z.clone()
        z_old, u_old = z.clone(), u.clone()
        cbar = c / (torch.linalg.norm(c, dim=1, keepdim=True) + 1e-8)
        alpha = torch.full((B,), self.alpha, device=device)
        beta = torch.full((B,), self.beta, device=device)
        accept_tol = float(cfg.get("anderson_accept_tol", 1.05))
        for k in range(K):
            # Base step.
            x_base, z_base, u_base = self._step(A, P, b, cbar, z, u, alpha, beta, cfg)
            if k > 0 and self.omega > 0:
                z_in = z + self.omega * (z - z_old)
                u_in = u + self.omega * (u - u_old)
                x_acc, z_acc, u_acc = self._step(A, P, b, cbar, z_in, u_in, alpha, beta, cfg)
                mon_base = self._monitor(A, b, z_base, z, cfg)
                mon_acc = self._monitor(A, b, z_acc, z, cfg)
                ok = (mon_acc <= accept_tol * mon_base) & torch.isfinite(mon_acc)
                mask = ok.unsqueeze(1)
                x_next = torch.where(mask, x_acc, x_base)
                z_next = torch.where(mask, z_acc, z_base)
                u_next = torch.where(mask, u_acc, u_base)
            else:
                x_next, z_next, u_next = x_base, z_base, u_base
            z_prev = z
            z_old, u_old = z, u
            x, z, u = x_next, z_next, u_next
        return {"x": x, "z": z, "u": u, "z_prev": z_prev}


class StableLearnedPDHG(nn.Module):
    """Stabilized learned PDHG proxy.

    The learned step sizes are constrained to satisfy
        tau_k * sigma_k * ||A||_2^2 <= pdhg_safety < 1,
    which prevents the NaN/Inf failures caused by unconstrained learned PDHG.
    """
    def __init__(self, cfg: Dict[str, Any], controller="gru"):
        super().__init__()
        self.cfg = cfg
        self.policy = CausalPolicy(cfg, controller, 3)

    def forward(self, A, P, b, c, K: int):
        cfg = self.cfg
        B, n = c.shape
        device = c.device
        x = torch.zeros(B, n, device=device)
        x_prev = x.clone()
        y = torch.zeros(B, A.shape[0], device=device)
        rho_prev = torch.full((B,), cfg["rho_base"], device=device)
        state = self.policy.init_state(B, device)
        params = []
        # Matrix norm is small enough to compute directly for these benchmarks.
        norm_A = torch.linalg.matrix_norm(A, ord=2).detach()
        norm_A_sq = torch.clamp(norm_A * norm_A, min=1e-8)
        tau_base = math.sqrt(float(cfg.get("pdhg_safety", 0.95))) / torch.clamp(norm_A, min=1e-8)
        log_scale = float(cfg.get("pdhg_log_scale", 0.5))
        theta_max = float(cfg.get("pdhg_theta_max", 0.8))
        for k in range(K):
            phi = feature_vector(A, b, c, x, x, x_prev, rho_prev, cfg)
            raw, state = self.policy(phi, k, state)
            q = torch.tanh(raw)
            tau = tau_base * torch.exp(log_scale * q[:, 0])
            # Strictly enforce the PDHG stability condition sample-wise.
            sigma_cap = float(cfg.get("pdhg_safety", 0.95)) / (torch.clamp(tau, min=1e-8) * norm_A_sq)
            sigma = sigma_cap * (0.05 + 0.95 * torch.sigmoid(raw[:, 1]))
            theta = theta_max * torch.sigmoid(raw[:, 2])
            x_bar = x + theta.unsqueeze(1) * (x - x_prev)
            y = y + sigma.unsqueeze(1) * (x_bar @ A.t() - b)
            grad = c + y @ A
            x_next = proj_cone_torch(x - tau.unsqueeze(1) * grad, cfg)
            x_prev, x = x, x_next
            rho_prev = 1.0 / torch.clamp(tau, min=1e-8)
            params.append({"tau": tau, "sigma": sigma, "theta": theta})
        return {"x": x, "z": x, "u": y, "z_prev": x_prev, "params": params}

# Backward-compatible alias for old configs.
LearnedPDHG = StableLearnedPDHG


class RCDRSAlphaBeta(RCADMM):
    """Internal ablation: residual-controlled alpha/beta DRS view without rho control."""
    def __init__(self, cfg: Dict[str, Any], controller="gru"):
        super().__init__(cfg, controller=controller, learned=("alpha", "beta"), envelope=True, growth=True)

# Backward-compatible alias. This is an internal ablation, not an external baseline.
LearnedDRS = RCDRSAlphaBeta


class KKTPolish(nn.Module):
    def __init__(self, cfg: Dict[str, Any], base: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.base = base
        self.net = nn.Sequential(nn.Linear(cfg["feature_dim"], cfg["hidden_dim"]), nn.SiLU(), nn.Linear(cfg["hidden_dim"], cfg["n"]))

    def forward(self, A, P, b, c, K: int):
        out = self.base(A, P, b, c, K)
        z = out["z"]
        phi = feature_vector(A, b, c, out.get("x", z), z, out.get("z_prev", z), torch.ones(z.shape[0], device=z.device), self.cfg)
        dz = 0.05 * torch.tanh(self.net(phi))
        z1 = proj_cone_torch(z + dz, self.cfg)
        z2 = proj_affine(z1, A, P, b)
        z3 = proj_cone_torch(z2, self.cfg)
        return {"x": z2, "z": z3, "u": out.get("u", torch.zeros_like(z)), "z_prev": z}

# ============================================================
# 6. Training, tuning, evaluation
# ============================================================

def to_device(batch, device):
    return tuple(t.to(device) for t in batch)


@torch.no_grad()
def evaluate(name: str, model: nn.Module, loader: DataLoader, A, P, cfg: Dict[str, Any], K: int, runtime=True) -> Dict[str, Any]:
    model.eval()
    device = torch.device(cfg["device"])
    vals = {"obj_gap_pos": [], "obj_gap_signed": [], "obj_gap_abs": [], "eq_vio": [], "cone_vio": [], "sol_dist": [], "div": []}
    n_inst = 0
    if runtime:
        for i, batch in enumerate(loader):
            if i >= cfg["runtime_warmup"]:
                break
            b, c, xstar, pstar = to_device(batch, device)
            _ = model(A, P, b, c, K)
            if device.type == "cuda":
                torch.cuda.synchronize()
        start = now_ms()
    else:
        start = None
    repeats = cfg["runtime_repeats"] if runtime else 1
    for rep in range(repeats):
        for batch in loader:
            b, c, xstar, pstar = to_device(batch, device)
            out = model(A, P, b, c, K)
            if runtime and device.type == "cuda":
                torch.cuda.synchronize()
            mets = compute_metrics(out, A, b, c, xstar, pstar, cfg)
            if rep == 0:
                for key in vals:
                    vals[key].append(mets[key].detach().cpu())
            n_inst += b.shape[0]
    elapsed = (now_ms() - start) if runtime else None
    row = {"method": name, "K": K}
    for key, chunks in vals.items():
        arr = torch.cat(chunks, dim=0)
        row[f"{key}_mean"] = safe_mean(arr)
        row[f"{key}_median"] = safe_median(arr)
    row["runtime_ms"] = elapsed / max(n_inst, 1) if elapsed is not None else float("nan")
    row["n_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return row


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, A, P, cfg: Dict[str, Any], K: int, name: str) -> nn.Module:
    device = torch.device(cfg["device"])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_state, best_score = None, float("inf")
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for batch in train_loader:
            b, c, xstar, pstar = to_device(batch, device)
            out = model(A, P, b, c, K)
            loss = training_loss(out, A, b, c, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep == 1 or ep % max(1, cfg["epochs"] // 4) == 0:
            val = evaluate(name, model, val_loader, A, P, cfg, K, runtime=False)
            score = val["obj_gap_pos_mean"] + 10.0 * val["eq_vio_mean"] + 10.0 * val["cone_vio_mean"]
            print(f"[Train:{name}] ep={ep:03d} loss={np.mean(losses):.3e} gap={val['obj_gap_pos_mean']:.3e} eq={val['eq_vio_mean']:.3e} cone={val['cone_vio_mean']:.3e}")
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def tune_oracle_grid(val_loader: DataLoader, A, P, cfg: Dict[str, Any], K: int) -> Tuple[float, float, float]:
    device = torch.device(cfg["device"])
    best, best_score = None, float("inf")
    count = 0
    for rho in cfg["tune_rho_grid"]:
        for alpha in cfg["tune_alpha_grid"]:
            for beta in cfg["tune_beta_grid"]:
                model = FixedADMM(cfg, rho, alpha, beta).to(device)
                # Score on at most tune_subset samples.
                vals = []
                seen = 0
                for batch in val_loader:
                    b, c, xstar, pstar = to_device(batch, device)
                    out = model(A, P, b, c, K)
                    mets = compute_metrics(out, A, b, c, xstar, pstar, cfg)
                    score_vec = mets["obj_gap_pos"] + 10.0 * mets["eq_vio"] + 10.0 * mets["cone_vio"]
                    vals.append(score_vec.detach())
                    seen += b.shape[0]
                    if seen >= cfg["tune_subset"]:
                        break
                score = float(torch.mean(torch.cat(vals)).cpu())
                count += 1
                if score < best_score:
                    best_score = score
                    best = (float(rho), float(alpha), float(beta))
    print(f"[OracleGrid] K={K}, best={best}, score={best_score:.3e}, trials={count}")
    return best

# ============================================================
# 7. Experiment runners
# ============================================================

def prepare_suite(base_cfg: Dict[str, Any], problem: str, scale: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, DataLoader, DataLoader, DataLoader]:
    cfg = apply_problem_scale(base_cfg, problem, scale)
    cfg["out_dir"] = os.path.join(cfg["root_out_dir"], f"{problem}_{scale}")
    ensure_dir(cfg["out_dir"])
    set_seed(cfg["seed"])
    data = generate_dataset(cfg)
    device = torch.device(cfg["device"])
    A = data["A"].to(device)
    P = data["P"].to(device)
    train_loader = make_loader(data, "train", cfg, shuffle=True)
    val_loader = make_loader(data, "val", cfg, shuffle=False)
    test_loader = make_loader(data, "test", cfg, shuffle=False)
    return cfg, data, A, P, train_loader, val_loader, test_loader


def build_learned_model(method_name: str, cfg: Dict[str, Any]) -> Optional[nn.Module]:
    if method_name == "RC-ADMM":
        return RCADMM(cfg, controller="gru", learned=("rho", "alpha", "beta"), envelope=cfg["use_envelope"], growth=cfg["use_growth"])
    if method_name in ["Stable-Learned-PDHG", "Learned-PDHG"]:
        return StableLearnedPDHG(cfg, controller="gru")
    # RC-DRS-alpha-beta is intentionally not used in Exp. 1, because it is an
    # internal ablation of RC-ADMM rather than an independent external baseline.
    if method_name in ["RC-DRS-alpha-beta", "Learned-DRS"]:
        return RCDRSAlphaBeta(cfg, controller="gru")
    return None


def run_exp1_suite(base_cfg: Dict[str, Any], problem: str, scale: str) -> pd.DataFrame:
    cfg, data, A, P, train_loader, val_loader, test_loader = prepare_suite(base_cfg, problem, scale)
    device = torch.device(cfg["device"])
    rows: List[Dict[str, Any]] = []

    # Train each learned method once at k_train, evaluate at all K.
    learned_models: Dict[str, nn.Module] = {}
    for method in cfg["exp1_methods"]:
        model = build_learned_model(method, cfg)
        if model is not None:
            print(f"\n[Train learned] {problem}-{scale}: {method}")
            learned_models[method] = train_model(model.to(device), train_loader, val_loader, A, P, cfg, cfg["k_train"], method)

    oracle_cache: Dict[int, Tuple[float, float, float]] = {}
    for K in cfg["k_values"]:
        if "Fixed-ADMM" in cfg["exp1_methods"]:
            rows.append(evaluate("Fixed-ADMM", FixedADMM(cfg).to(device), test_loader, A, P, cfg, K))

        if any(m in cfg["exp1_methods"] for m in ["OracleGrid-ADMM", "Spectral-AADMM", "DRE-Anderson-DRS"]):
            oracle_cache[K] = tune_oracle_grid(val_loader, A, P, cfg, K)
            rho, alpha, beta = oracle_cache[K]
        else:
            rho, alpha, beta = cfg["rho_base"], cfg["alpha_base"], cfg["beta_base"]

        if "OracleGrid-ADMM" in cfg["exp1_methods"]:
            row = evaluate("OracleGrid-ADMM", FixedADMM(cfg, rho, alpha, beta).to(device), test_loader, A, P, cfg, K)
            row.update({"rho": rho, "alpha": alpha, "beta": beta})
            rows.append(row)
        if "Spectral-AADMM" in cfg["exp1_methods"]:
            rows.append(evaluate("Spectral-AADMM", SpectralAADMM(cfg, rho, alpha, beta).to(device), test_loader, A, P, cfg, K))
        if "DRE-Anderson-DRS" in cfg["exp1_methods"]:
            rows.append(evaluate("DRE-Anderson-DRS", AndersonDRS(cfg, rho, alpha, beta).to(device), test_loader, A, P, cfg, K))

        for method, model in learned_models.items():
            rows.append(evaluate(method, model, test_loader, A, P, cfg, K))

    df = pd.DataFrame(rows)
    df["problem"], df["scale"], df["exp"] = problem, scale, "exp1"
    out = os.path.join(cfg["out_dir"], f"exp1_{problem}_{scale}.csv")
    df.to_csv(out, index=False)
    print(f"[Saved] {out}")
    print(df[["problem", "scale", "method", "K", "obj_gap_pos_mean", "eq_vio_mean", "cone_vio_mean", "sol_dist_mean", "runtime_ms"]])
    return df


def run_exp2_suite(base_cfg: Dict[str, Any], problem: str, scale: str) -> pd.DataFrame:
    cfg, data, A, P, train_loader, val_loader, test_loader = prepare_suite(base_cfg, problem, scale)
    device = torch.device(cfg["device"])
    K = cfg["k_train"]
    rows: List[Dict[str, Any]] = []

    for spec in cfg["exp2_parameter_variants"]:
        name = f"Param-{spec['name']}"
        model = RCADMM(cfg, controller="gru", learned=tuple(spec["learned"]), envelope=cfg["use_envelope"], growth=cfg["use_growth"]).to(device)
        train_model(model, train_loader, val_loader, A, P, cfg, K, name)
        row = evaluate(name, model, test_loader, A, P, cfg, K)
        row["ablation_group"] = "parameter"
        rows.append(row)

    for spec in cfg["exp2_controller_variants"]:
        name = f"Ctrl-{spec['name']}"
        model = RCADMM(cfg, controller=spec["controller"], learned=("rho", "alpha", "beta"), envelope=cfg["use_envelope"], growth=cfg["use_growth"]).to(device)
        train_model(model, train_loader, val_loader, A, P, cfg, K, name)
        row = evaluate(name, model, test_loader, A, P, cfg, K)
        row["ablation_group"] = "controller"
        rows.append(row)

    for spec in cfg["exp2_algorithm_variants"]:
        name = f"Alg-{spec['name']}"
        if spec["kind"] == "admm":
            model = RCADMM(cfg, controller="gru", learned=("rho", "alpha", "beta"), envelope=cfg["use_envelope"], growth=cfg["use_growth"])
        elif spec["kind"] == "pdhg":
            model = StableLearnedPDHG(cfg, controller="gru")
        elif spec["kind"] in ["drs", "drs_ab"]:
            model = RCDRSAlphaBeta(cfg, controller="gru")
        else:
            raise ValueError(spec["kind"])
        model.to(device)
        train_model(model, train_loader, val_loader, A, P, cfg, K, name)
        row = evaluate(name, model, test_loader, A, P, cfg, K)
        row["ablation_group"] = "algorithm"
        rows.append(row)

    for spec in cfg["exp2_safeguard_variants"]:
        name = f"Safe-{spec['name']}"
        model = RCADMM(cfg, controller="gru", learned=("rho", "alpha", "beta"), envelope=spec["envelope"], growth=spec["growth"]).to(device)
        train_model(model, train_loader, val_loader, A, P, cfg, K, name)
        row = evaluate(name, model, test_loader, A, P, cfg, K)
        row["ablation_group"] = "safeguard"
        rows.append(row)

    df = pd.DataFrame(rows)
    df["problem"], df["scale"], df["exp"] = problem, scale, "exp2"
    out = os.path.join(cfg["out_dir"], f"exp2_{problem}_{scale}.csv")
    df.to_csv(out, index=False)
    print(f"[Saved] {out}")
    print(df[["problem", "scale", "ablation_group", "method", "K", "obj_gap_pos_mean", "eq_vio_mean", "cone_vio_mean", "sol_dist_mean", "runtime_ms"]])
    return df


def run_all(config: Dict[str, Any]) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dir(config["root_out_dir"])
    all_dfs = []
    for exp in config["experiments"]:
        if not exp.get("enabled", True):
            continue
        for problem in exp["problems"]:
            for scale in exp["scales"]:
                if config.get("quick", False) and not (problem == "mixed" and scale == "small"):
                    continue
                print("\n" + "#" * 100)
                print(f"[RUN] exp={exp['name']} | problem={problem} | scale={scale} | device={config['device']}")
                print("#" * 100)
                if exp["name"] == "exp1":
                    all_dfs.append(run_exp1_suite(config, problem, scale))
                elif exp["name"] == "exp2":
                    all_dfs.append(run_exp2_suite(config, problem, scale))
                else:
                    raise ValueError(exp["name"])
    if all_dfs:
        out_df = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(config["root_out_dir"], "all_results.csv")
        out_df.to_csv(out_path, index=False)
        print(f"[All saved] {out_path}")
        return out_df
    return pd.DataFrame()


if __name__ == "__main__":
    run_all(CONFIG)
