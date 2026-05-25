"""
Fair per-K DFL experiment for RC-ADMM as a differentiable mixed-cone solver layer.

Protocol follows the solver-level Experiment 1:
  for each seed and prescribed depth K, tune the oracle ADMM core on validation data,
  train each learned decision layer at the same K, and evaluate at the same K.

Compared layers include MSE-PTO, unrolled ADMM/DRS/PDHG layers, RC-ADMM with
oracle-centered envelope/no-envelope variants, and an optional CVXPYLayer baseline.
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    CVXPYLAYERS_AVAILABLE = True
except Exception:
    cp = None
    CvxpyLayer = None
    CVXPYLAYERS_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # Multi-seed / per-K protocol
    "seeds": [0, 1, 2],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float32",

    # Output
    "out_dir": "./dfl_mixed_perK_outputs",

    # Quick smoke test. Set False for full experiment.
    "quick": False,

    # Data sizes. These are overridden in quick mode.
    "n_train": 5000,
    "n_val": 1500,
    "n_test": 1500,
    "batch_size": 2500,

    # Feature-to-cost data generation
    "feature_dim": 20,
    "feature_hidden": 64,
    "cost_noise": 0.05,
    "cost_scale": 1.0,
    "cost_pred_clip": 20.0,

    # Mixed cone dimensions. These are overridden in quick mode.
    "mixed_dims": {
        "nonneg_dim": 8,
        "soc_blocks": 3,
        "soc_dim": 5,
        "rot_blocks": 2,
        "rot_dim": 5,
        "psd_dim": 4,
        "m_eq": 18,
        "condition": 80.0,
    },

    # Fixed-depth DFL budgets. Each learned method is trained separately at each K.
    "k_values": [5, 10, 15, 20],

    # Predictor
    "predictor_hidden": 128,
    "predictor_layers": 2,

    # Training
    "epochs_predictor": 50,
    "epochs_solver_pretrain": 80,
    "lr_predictor": 1e-3,
    "lr_solver": 1e-3,
    "weight_decay": 1e-5,
    "lambda_eq_dfl": 20.0,
    "lambda_mse_aux": 0.0,
    "grad_clip": 5.0,

    # Optional RC solver pretraining before DFL.
    "pretrain_rc_solver": True,

    # ADMM / RC base parameters. alpha_base and beta_base are overwritten by
    # the per-K oracle-tuned core before each training run.
    "rho_base": 1.0,
    "alpha_base": 1.0,
    "beta_base": 1.0,
    "alpha_min": 0.2,
    "alpha_max": 1.8,
    "beta_min": 1e-3,
    "beta_max": 10.0,
    "rho_min": 1e-3,
    "rho_max": 1e3,

    # RC controller
    "rc_hidden": 64,
    "rc_controller": "gru",
    "envelope": True,
    "growth": True,
    "delta0": 2.0,
    "k0": 80.0,
    "p_decay": 1.2,
    "alpha_delta_scale": 0.25,
    "chi_rho": 10.0,
    "chi_beta": 10.0,

    # Oracle fixed ADMM tuning. This is performed separately at each K.
    "oracle_grid": {
        "alphas": [0.8, 1.0, 1.3, 1.6, 1.8],
        "betas": [0.1, 0.3, 1.0, 3.0, 6.0],
    },

    # DRE-Anderson-DRS
    "anderson_omega": 0.25,
    "anderson_accept_tol": 1.05,

    # Stable PDHG
    "pdhg_safety": 0.95,
    "pdhg_theta": 0.8,

    # Optional CVXPYLayer baseline. It is automatically skipped when the package
    # is unavailable. It can be slow on PSD mixed-cone problems.
    "enable_cvxpy_layer": True,
    "cvx_solver_args": {"eps": 1e-4, "max_iters": 2000},
    "max_train_batches_cvx": None,

    # Methods. CVXPYLayer entries are skipped automatically if unavailable.
    "methods": [
        "MSE-PTO-OracleADMM",
        "MSE-PTO-CVXPYLayer",
        "DFL-CVXPYLayer",
        "DFL-Fixed-ADMM",
        "DFL-OracleGrid-ADMM",
        "DFL-DRE-Anderson-DRS",
        "DFL-Stable-Learned-PDHG",
        "DFL-RC-Env-frozen",
        "DFL-RC-Env-joint",
        "DFL-RC-NoEnv-frozen",
        "DFL-RC-NoEnv-joint",
    ],
}


if CONFIG["quick"]:
    CONFIG.update({
        "seeds": [0],
        "n_train": 256,
        "n_val": 96,
        "n_test": 96,
        "batch_size": 64,
        "epochs_predictor": 3,
        "epochs_solver_pretrain": 3,
        "k_values": [5, 10],
        "enable_cvxpy_layer": False,
    })
    CONFIG["mixed_dims"] = {
        "nonneg_dim": 4,
        "soc_blocks": 2,
        "soc_dim": 4,
        "rot_blocks": 1,
        "rot_dim": 4,
        "psd_dim": 3,
        "m_eq": 8,
        "condition": 20.0,
    }


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(cfg) -> torch.dtype:
    return torch.float64 if cfg.get("dtype") == "float64" else torch.float32


def to_device_batch(batch, device):
    return tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)


def batch_matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # A: [m,n], x: [B,n] -> [B,m]
    return x @ A.t()


def objective(c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (c * z).sum(dim=1)


def safe_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim).clamp_min(eps)


# ============================================================
# Mixed cone structure and projections
# ============================================================

@dataclass
class ConeSlices:
    nonneg: slice
    soc: List[slice]
    rot: List[slice]
    psd: slice
    n: int
    psd_dim: int


def make_cone_slices(cfg) -> ConeSlices:
    d = cfg["mixed_dims"]
    idx = 0
    nonneg = slice(idx, idx + d["nonneg_dim"])
    idx += d["nonneg_dim"]

    soc_slices = []
    for _ in range(d["soc_blocks"]):
        soc_slices.append(slice(idx, idx + d["soc_dim"]))
        idx += d["soc_dim"]

    rot_slices = []
    for _ in range(d["rot_blocks"]):
        rot_slices.append(slice(idx, idx + d["rot_dim"]))
        idx += d["rot_dim"]

    psd_len = d["psd_dim"] * d["psd_dim"]
    psd = slice(idx, idx + psd_len)
    idx += psd_len

    return ConeSlices(nonneg=nonneg, soc=soc_slices, rot=rot_slices,
                      psd=psd, n=idx, psd_dim=d["psd_dim"])


def proj_soc_block(x: torch.Tensor) -> torch.Tensor:
    # x: [B,d], cone {t >= ||v||}
    t = x[:, :1]
    v = x[:, 1:]
    nv = torch.linalg.norm(v, dim=1, keepdim=True)

    inside = nv <= t
    negative = nv <= -t
    scale = 0.5 * (1.0 + t / nv.clamp_min(1e-12))
    proj_mid = torch.cat([0.5 * (nv + t), scale * v], dim=1)
    zeros = torch.zeros_like(x)
    return torch.where(inside, x, torch.where(negative, zeros, proj_mid))


def proj_rotated_soc_block(x: torch.Tensor) -> torch.Tensor:
    # Rotated cone: u >= 0, v >= 0, 2uv >= ||w||^2.
    # Transform to SOC:
    # y0=(u+v)/sqrt(2), y1=(u-v)/sqrt(2), y_rest=w.
    sqrt2 = math.sqrt(2.0)
    u = x[:, :1]
    v = x[:, 1:2]
    w = x[:, 2:]
    y = torch.cat([(u + v) / sqrt2, (u - v) / sqrt2, w], dim=1)
    py = proj_soc_block(y)
    y0 = py[:, :1]
    y1 = py[:, 1:2]
    pw = py[:, 2:]
    pu = (y0 + y1) / sqrt2
    pv = (y0 - y1) / sqrt2
    return torch.cat([pu, pv, pw], dim=1)


def proj_psd_block(x: torch.Tensor, p: int) -> torch.Tensor:
    # x: [B,p*p]. Symmetrize then eigenvalue clipping.
    # Numerical guards are important during end-to-end DFL training, where
    # early predictors may generate large cost vectors and extrapolated solver
    # states can become ill-conditioned.
    B = x.shape[0]
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    X = x.reshape(B, p, p)
    X = 0.5 * (X + X.transpose(-1, -2))
    try:
        eigvals, eigvecs = torch.linalg.eigh(X)
    except RuntimeError:
        # Add a tiny diagonal perturbation and retry. This should almost never
        # be used, but it prevents a whole DFL run from crashing in early epochs.
        eye = torch.eye(p, device=x.device, dtype=x.dtype).unsqueeze(0)
        eigvals, eigvecs = torch.linalg.eigh(X + 1e-6 * eye)
    eigvals_pos = eigvals.clamp_min(0.0)
    Xp = eigvecs @ torch.diag_embed(eigvals_pos) @ eigvecs.transpose(-1, -2)
    return Xp.reshape(B, p * p)


def proj_mixed_cone(x: torch.Tensor, cs: ConeSlices) -> torch.Tensor:
    parts = []
    parts.append(x[:, cs.nonneg].clamp_min(0.0))
    for sl in cs.soc:
        parts.append(proj_soc_block(x[:, sl]))
    for sl in cs.rot:
        parts.append(proj_rotated_soc_block(x[:, sl]))
    parts.append(proj_psd_block(x[:, cs.psd], cs.psd_dim))
    return torch.cat(parts, dim=1)


def cone_violation(x: torch.Tensor, cs: ConeSlices) -> torch.Tensor:
    px = proj_mixed_cone(x, cs)
    return torch.linalg.norm(x - px, dim=1) / (1.0 + torch.linalg.norm(x, dim=1))


# ============================================================
# KKT-consistent data generation
# ============================================================

def random_orthogonal_matrix(n: int, device, dtype) -> torch.Tensor:
    G = torch.randn(n, n, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(G)
    return Q


def generate_A(m: int, n: int, condition: float, device, dtype) -> torch.Tensor:
    # Generate full-row-rank A with controlled singular values.
    G1 = torch.randn(m, m, device=device, dtype=dtype)
    U, _ = torch.linalg.qr(G1)
    G2 = torch.randn(n, m, device=device, dtype=dtype)
    V, _ = torch.linalg.qr(G2)
    s = torch.logspace(0.0, math.log10(condition), steps=m, device=device, dtype=dtype)
    # normalize to avoid huge b
    s = s / s.max()
    A = U @ torch.diag(s) @ V.t()
    return A


def sample_primal_dual_cone_pair(B: int, cs: ConeSlices, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample x in K and s in K* with approximate complementarity."""
    x_parts, s_parts = [], []

    # Nonnegative cone: complementary sparse pattern.
    dn = cs.nonneg.stop - cs.nonneg.start
    mask = (torch.rand(B, dn, device=device) > 0.35).to(dtype)
    x_non = mask * (0.2 + torch.rand(B, dn, device=device, dtype=dtype))
    s_non = (1.0 - mask) * (0.2 + torch.rand(B, dn, device=device, dtype=dtype))
    x_parts.append(x_non)
    s_parts.append(s_non)

    # SOC blocks: boundary complementary pair.
    for sl in cs.soc:
        d = sl.stop - sl.start
        v = torch.randn(B, d - 1, device=device, dtype=dtype)
        nv = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
        t = nv
        x_soc = torch.cat([t, v], dim=1)
        mu = 0.2 + torch.rand(B, 1, device=device, dtype=dtype)
        s_soc = mu * torch.cat([t, -v], dim=1)
        x_parts.append(x_soc)
        s_parts.append(s_soc)

    # Rotated SOC blocks: use SOC transform to generate complementary pairs.
    sqrt2 = math.sqrt(2.0)
    for sl in cs.rot:
        d = sl.stop - sl.start
        v_soc = torch.randn(B, d - 1, device=device, dtype=dtype)
        nv = torch.linalg.norm(v_soc, dim=1, keepdim=True).clamp_min(1e-8)
        y = torch.cat([nv, v_soc], dim=1)  # SOC boundary
        mu = 0.2 + torch.rand(B, 1, device=device, dtype=dtype)
        sy = mu * torch.cat([nv, -v_soc], dim=1)

        y0, y1, yw = y[:, :1], y[:, 1:2], y[:, 2:]
        u = (y0 + y1) / sqrt2
        vv = (y0 - y1) / sqrt2
        x_rot = torch.cat([u, vv, yw], dim=1)

        sy0, sy1, syw = sy[:, :1], sy[:, 1:2], sy[:, 2:]
        su = (sy0 + sy1) / sqrt2
        sv = (sy0 - sy1) / sqrt2
        s_rot = torch.cat([su, sv, syw], dim=1)

        # Numerical safety: project both.
        x_parts.append(proj_rotated_soc_block(x_rot))
        s_parts.append(proj_rotated_soc_block(s_rot))

    # PSD block: X and S with orthogonal eigenspaces.
    p = cs.psd_dim
    X_list, S_list = [], []
    for _ in range(B):
        Q = random_orthogonal_matrix(p, device, dtype)
        rank = max(1, p // 2)
        vals_x = torch.zeros(p, device=device, dtype=dtype)
        vals_s = torch.zeros(p, device=device, dtype=dtype)
        vals_x[:rank] = 0.2 + torch.rand(rank, device=device, dtype=dtype)
        vals_s[rank:] = 0.2 + torch.rand(p - rank, device=device, dtype=dtype)
        X = Q @ torch.diag(vals_x) @ Q.t()
        S = Q @ torch.diag(vals_s) @ Q.t()
        X_list.append(X.reshape(-1))
        S_list.append(S.reshape(-1))
    x_parts.append(torch.stack(X_list, dim=0))
    s_parts.append(torch.stack(S_list, dim=0))

    return torch.cat(x_parts, dim=1), torch.cat(s_parts, dim=1)



def make_feature_kkt_params(cfg, cs: ConeSlices, device, dtype):
    """Random fixed nonlinear map from features to KKT-consistent CLP instances.

    The generated map is:
        s -> hidden h(s) -> (x_star(s), slack_star(s), y(s))
        b(s) = A x_star(s), c(s) = slack_star(s) - A^T y(s)

    Hence x_star is exactly KKT-consistent for the generated (A,b,c,K).
    """
    d_s = cfg["feature_dim"]
    h = cfg["feature_hidden"]
    params = {
        "W1": torch.randn(d_s, h, device=device, dtype=dtype) / math.sqrt(d_s),
        "b1": 0.1 * torch.randn(h, device=device, dtype=dtype),
    }

    def lin(name, out_dim, scale=1.0):
        params[f"W_{name}"] = scale * torch.randn(h, out_dim, device=device, dtype=dtype) / math.sqrt(h)
        params[f"b_{name}"] = 0.1 * torch.randn(out_dim, device=device, dtype=dtype)

    dn = cs.nonneg.stop - cs.nonneg.start
    lin("non_mask", dn)
    lin("non_x", dn)
    lin("non_s", dn)

    soc_v_dim = sum((sl.stop - sl.start - 1) for sl in cs.soc)
    lin("soc_v", soc_v_dim)
    lin("soc_mu", len(cs.soc))

    rot_v_dim = sum((sl.stop - sl.start - 1) for sl in cs.rot)
    lin("rot_v", rot_v_dim)
    lin("rot_mu", len(cs.rot))

    p = cs.psd_dim
    r = max(1, p // 2)
    lin("psd_xeig", r)
    lin("psd_seig", p - r)
    # Fixed orthogonal basis for PSD block. This keeps the map learnable but still nontrivial.
    Q = random_orthogonal_matrix(p, device, dtype)
    params["psd_Q"] = Q
    params["psd_rank"] = r

    # equality multiplier map, dimension filled later by generate_dfl_dataset
    return params


def hidden_from_features(S: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.tanh(S @ params["W1"] + params["b1"])


def apply_linear(H: torch.Tensor, params: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return H @ params[f"W_{name}"] + params[f"b_{name}"]


def feature_to_cone_pair(S: torch.Tensor, params: Dict[str, torch.Tensor], cs: ConeSlices):
    """Generate complementary pair (x_star, slack_star) from features."""
    H = hidden_from_features(S, params)
    B, device, dtype = S.shape[0], S.device, S.dtype
    x_parts, s_parts = [], []

    # Nonnegative cone with feature-dependent active set.
    mask_logits = apply_linear(H, params, "non_mask")
    x_raw = apply_linear(H, params, "non_x")
    s_raw = apply_linear(H, params, "non_s")
    mask = (mask_logits > 0.0).to(dtype)
    x_non = mask * (0.2 + F.softplus(x_raw))
    s_non = (1.0 - mask) * (0.2 + F.softplus(s_raw))
    x_parts.append(x_non)
    s_parts.append(s_non)

    # SOC blocks: x=(||v||,v), s=mu(||v||,-v).
    soc_v_all = apply_linear(H, params, "soc_v")
    soc_mu_all = 0.2 + F.softplus(apply_linear(H, params, "soc_mu"))
    v_cursor = 0
    for j, sl in enumerate(cs.soc):
        d = sl.stop - sl.start
        vd = d - 1
        v = soc_v_all[:, v_cursor:v_cursor + vd] / math.sqrt(max(vd, 1))
        v_cursor += vd
        nv = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-6)
        mu = soc_mu_all[:, j:j+1]
        x_soc = torch.cat([nv, v], dim=1)
        s_soc = mu * torch.cat([nv, -v], dim=1)
        x_parts.append(x_soc)
        s_parts.append(s_soc)

    # Rotated SOC blocks through SOC isomorphism.
    sqrt2 = math.sqrt(2.0)
    rot_v_all = apply_linear(H, params, "rot_v")
    rot_mu_all = 0.2 + F.softplus(apply_linear(H, params, "rot_mu"))
    v_cursor = 0
    for j, sl in enumerate(cs.rot):
        d = sl.stop - sl.start
        vd = d - 1
        v_soc = rot_v_all[:, v_cursor:v_cursor + vd] / math.sqrt(max(vd, 1))
        v_cursor += vd
        nv = torch.linalg.norm(v_soc, dim=1, keepdim=True).clamp_min(1e-6)
        mu = rot_mu_all[:, j:j+1]
        y = torch.cat([nv, v_soc], dim=1)
        sy = mu * torch.cat([nv, -v_soc], dim=1)

        y0, y1, yw = y[:, :1], y[:, 1:2], y[:, 2:]
        u = (y0 + y1) / sqrt2
        vv = (y0 - y1) / sqrt2
        x_rot = torch.cat([u, vv, yw], dim=1)

        sy0, sy1, syw = sy[:, :1], sy[:, 1:2], sy[:, 2:]
        su = (sy0 + sy1) / sqrt2
        sv = (sy0 - sy1) / sqrt2
        s_rot = torch.cat([su, sv, syw], dim=1)

        x_parts.append(proj_rotated_soc_block(x_rot))
        s_parts.append(proj_rotated_soc_block(s_rot))

    # PSD block with complementary eigenspaces.
    p = cs.psd_dim
    r = int(params["psd_rank"])
    Q = params["psd_Q"]
    xeig_raw = apply_linear(H, params, "psd_xeig")
    seig_raw = apply_linear(H, params, "psd_seig")
    xeig = torch.zeros(B, p, device=device, dtype=dtype)
    seig = torch.zeros(B, p, device=device, dtype=dtype)
    xeig[:, :r] = 0.2 + F.softplus(xeig_raw)
    if p - r > 0:
        seig[:, r:] = 0.2 + F.softplus(seig_raw)

    X = Q.unsqueeze(0) @ torch.diag_embed(xeig) @ Q.t().unsqueeze(0)
    Spsd = Q.unsqueeze(0) @ torch.diag_embed(seig) @ Q.t().unsqueeze(0)
    x_parts.append(X.reshape(B, p * p))
    s_parts.append(Spsd.reshape(B, p * p))

    x_star = torch.cat(x_parts, dim=1)
    slack = torch.cat(s_parts, dim=1)
    return x_star, slack, H


def generate_dfl_dataset(cfg, A: torch.Tensor, cs: ConeSlices, N: int, kkt_params):
    device = A.device
    dtype = A.dtype
    d_s = cfg["feature_dim"]

    S_feat = torch.randn(N, d_s, device=device, dtype=dtype)
    x_star, slack, H = feature_to_cone_pair(S_feat, kkt_params, cs)

    # Feature-dependent equality multiplier.
    if "W_y" not in kkt_params:
        h = H.shape[1]
        m = A.shape[0]
        kkt_params["W_y"] = 0.3 * torch.randn(h, m, device=device, dtype=dtype) / math.sqrt(h)
        kkt_params["b_y"] = 0.05 * torch.randn(m, device=device, dtype=dtype)

    y = H @ kkt_params["W_y"] + kkt_params["b_y"]

    b = batch_matvec(A, x_star)
    c_true = slack - y @ A

    # Per-instance positive normalization preserves the optimizer and improves training scale.
    c_true = c_true / (torch.linalg.norm(c_true, dim=1, keepdim=True) + 1e-8)
    opt_val = objective(c_true, x_star)

    return TensorDatasetDFL(S_feat, b, c_true, x_star, opt_val)


class TensorDatasetDFL(torch.utils.data.Dataset):
    def __init__(self, S, b, c, x_star, opt_val):
        self.S = S.detach().cpu()
        self.b = b.detach().cpu()
        self.c = c.detach().cpu()
        self.x_star = x_star.detach().cpu()
        self.opt_val = opt_val.detach().cpu()

    def __len__(self):
        return self.S.shape[0]

    def __getitem__(self, idx):
        return self.S[idx], self.b[idx], self.c[idx], self.x_star[idx], self.opt_val[idx]


def make_loaders(cfg, A, cs):
    device, dtype = A.device, A.dtype
    params = make_feature_kkt_params(cfg, cs, device, dtype)
    train = generate_dfl_dataset(cfg, A, cs, cfg["n_train"], params)
    val = generate_dfl_dataset(cfg, A, cs, cfg["n_val"], params)
    test = generate_dfl_dataset(cfg, A, cs, cfg["n_test"], params)

    def loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=shuffle, drop_last=False
        )
    return loader(train, True), loader(val, False), loader(test, False)


# ============================================================
# Affine projection
# ============================================================

@dataclass
class AffineProjector:
    A: torch.Tensor
    M: torch.Tensor  # inverse/pinv of AA^T

    @classmethod
    def build(cls, A: torch.Tensor):
        AA = A @ A.t()
        # pinv is safer for generated ill-conditioned cases
        M = torch.linalg.pinv(AA)
        return cls(A=A, M=M)

    def proj(self, v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        Av_minus_b = batch_matvec(self.A, v) - b
        correction = (Av_minus_b @ self.M.t()) @ self.A
        return v - correction


# ============================================================
# Solver layers
# ============================================================

class BaseSolver(nn.Module):
    name: str = "BaseSolver"

    def forward(self, b: torch.Tensor, c: torch.Tensor, K: int) -> torch.Tensor:
        raise NotImplementedError


class ADMMFixedSolver(BaseSolver):
    def __init__(self, projector: AffineProjector, cs: ConeSlices,
                 alpha: float = 1.0, beta: float = 1.0, name: str = "Fixed-ADMM"):
        super().__init__()
        self.projector = projector
        self.cs = cs
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.name = name

    def admm_step(self, z, u, b, c, alpha, beta):
        cbar = c / (torch.linalg.norm(c, dim=1, keepdim=True) + 1e-8)
        w = z - u - beta * cbar
        x = self.projector.proj(w, b)
        xbar = alpha * x + (1.0 - alpha) * z
        z_new = proj_mixed_cone(xbar + u, self.cs)
        u_new = u + xbar - z_new
        return x, z_new, u_new

    def forward(self, b, c, K: int):
        B, n = c.shape
        z = torch.zeros(B, n, device=c.device, dtype=c.dtype)
        u = torch.zeros_like(z)
        for _ in range(K):
            _, z, u = self.admm_step(z, u, b, c, self.alpha, self.beta)
        return z


class DREAndersonDRSSolver(ADMMFixedSolver):
    def __init__(self, projector: AffineProjector, cs: ConeSlices,
                 alpha: float = 1.0, beta: float = 1.0, omega: float = 0.25,
                 accept_tol: float = 1.05):
        super().__init__(projector, cs, alpha=alpha, beta=beta, name="DRE-Anderson-DRS")
        self.omega = float(omega)
        self.accept_tol = float(accept_tol)

    def monitor(self, z, u, b):
        eq = batch_matvec(self.projector.A, z) - b
        return torch.linalg.norm(eq, dim=1) + 0.1 * torch.linalg.norm(u, dim=1)

    def forward(self, b, c, K: int):
        B, n = c.shape
        z = torch.zeros(B, n, device=c.device, dtype=c.dtype)
        u = torch.zeros_like(z)
        z_prev = z.clone()
        u_prev = u.clone()

        for k in range(K):
            _, z_base, u_base = self.admm_step(z, u, b, c, self.alpha, self.beta)

            if k == 0:
                z_prev, u_prev = z, u
                z, u = z_base, u_base
                continue

            z_acc = z_base + self.omega * (z_base - z_prev)
            u_acc = u_base + self.omega * (u_base - u_prev)
            z_acc = torch.nan_to_num(z_acc, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
            u_acc = torch.nan_to_num(u_acc, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
            # Ensure cone feasibility after acceleration.
            z_acc = proj_mixed_cone(z_acc, self.cs)

            mon_base = self.monitor(z_base, u_base, b)
            mon_acc = self.monitor(z_acc, u_acc, b)
            accept = (mon_acc <= self.accept_tol * mon_base).float().view(-1, 1)

            z_next = accept * z_acc + (1.0 - accept) * z_base
            u_next = accept * u_acc + (1.0 - accept) * u_base

            z_prev, u_prev = z, u
            z = torch.nan_to_num(z_next, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
            u = torch.nan_to_num(u_next, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

        return z


class StablePDHGSolver(BaseSolver):
    name = "Stable-Learned-PDHG"

    def __init__(self, A: torch.Tensor, cs: ConeSlices, theta: float = 0.8, safety: float = 0.95):
        super().__init__()
        self.A = A
        self.cs = cs
        self.theta = float(theta)
        self.safety = float(safety)
        # Spectral norm for stable step sizing
        with torch.no_grad():
            norm_A = torch.linalg.matrix_norm(A, ord=2).item()
        self.norm_A_sq = max(norm_A * norm_A, 1e-8)
        self.tau = 0.9 / math.sqrt(self.norm_A_sq)
        self.sigma = self.safety / (self.tau * self.norm_A_sq)

    def forward(self, b, c, K: int):
        B, n = c.shape
        m = self.A.shape[0]
        x = torch.zeros(B, n, device=c.device, dtype=c.dtype)
        xbar = x.clone()
        y = torch.zeros(B, m, device=c.device, dtype=c.dtype)

        tau = self.tau
        sigma = self.sigma
        theta = self.theta

        for _ in range(K):
            y = y + sigma * (batch_matvec(self.A, xbar) - b)
            x_old = x
            grad = y @ self.A + c
            x = proj_mixed_cone(x - tau * grad, self.cs)
            xbar = x + theta * (x - x_old)

        return x



class CVXPYLayerSolver(BaseSolver):
    name = "CVXPYLayer"

    def __init__(self, A: torch.Tensor, cs: ConeSlices, cfg):
        super().__init__()
        if not CVXPYLAYERS_AVAILABLE:
            raise ImportError("cvxpylayers is not installed.")
        self.A = A
        self.cs = cs
        self.cfg = cfg
        self.layer = self._build_layer(A.detach().cpu().double().numpy(), cs)

    def _build_layer(self, A_np, cs: ConeSlices):
        n = cs.n
        m = A_np.shape[0]
        x = cp.Variable(n)
        b_param = cp.Parameter(m)
        c_param = cp.Parameter(n)
        cons = [A_np @ x == b_param]

        # Nonnegative block
        cons += [x[cs.nonneg.start:cs.nonneg.stop] >= 0]

        # SOC blocks
        for sl in cs.soc:
            cons += [cp.SOC(x[sl.start], x[sl.start + 1:sl.stop])]

        # Rotated SOC blocks represented as a standard SOC:
        # 2uv >= ||w||^2, u,v>=0 iff u+v >= ||[u-v, sqrt(2)w]||.
        for sl in cs.rot:
            u = x[sl.start]
            v = x[sl.start + 1]
            w = x[sl.start + 2:sl.stop]
            cons += [u >= 0, v >= 0]
            cons += [cp.SOC(u + v, cp.hstack([u - v, math.sqrt(2.0) * w]))]

        # PSD block, represented by a full vectorized matrix plus symmetry.
        p = cs.psd_dim
        X = cp.reshape(x[cs.psd.start:cs.psd.stop], (p, p), order="C")
        cons += [X == X.T, X >> 0]

        prob = cp.Problem(cp.Minimize(c_param @ x), cons)
        return CvxpyLayer(prob, parameters=[b_param, c_param], variables=[x])

    def forward(self, b, c, K: int):
        solver_args = self.cfg.get("cvx_solver_args", {"eps": 1e-4, "max_iters": 2000})
        z, = self.layer(b, c, solver_args=solver_args)
        return z


def residual_features(A, x, z, z_prev, b, c, rho_prev, alpha_prev, beta_prev, k: int, K: int, cfg):
    """Current 10-dimensional residual-action-time feature used in the paper.

    Cone feasibility is not included because z is produced by exact cone projection.
    rho is an auxiliary feedback-scale action and does not enter the ADMM transition.
    """
    z_norm = torch.linalg.norm(z, dim=1)
    con = torch.linalg.norm(x - z, dim=1) / (1.0 + z_norm)
    eq = torch.linalg.norm(batch_matvec(A, z) - b, dim=1) / (1.0 + torch.linalg.norm(b, dim=1))
    dz = torch.linalg.norm(z - z_prev, dim=1) / (1.0 + z_norm)
    obj_val = objective(c, z)
    obj_prev = objective(c, z_prev)
    obj_scale = torch.abs(obj_val) / (1.0 + torch.linalg.norm(c, dim=1) * z_norm)
    delta_obj = torch.asinh((obj_val - obj_prev) / (1.0 + torch.abs(obj_prev)))

    rho_base = float(cfg.get("rho_base", 1.0))
    beta_base = float(cfg.get("beta_base", 1.0))
    a_rho = torch.log(torch.clamp(rho_prev, min=1e-12) / max(rho_base, 1e-12))
    a_alpha = alpha_prev
    a_beta = torch.log(torch.clamp(beta_prev, min=1e-12) / max(beta_base, 1e-12))
    t = torch.full_like(con, float(k) / float(max(K, 1)))
    trem = torch.full_like(con, float(K - k) / float(max(K, 1)))

    return torch.stack([
        torch.log1p(con),
        torch.log1p(eq),
        torch.log1p(dz),
        torch.log1p(obj_scale),
        delta_obj,
        a_rho,
        a_alpha,
        a_beta,
        t,
        trem,
    ], dim=1)

class RCADMMController(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg["rc_hidden"]
        self.kind = cfg.get("rc_controller", "gru").lower()
        if self.kind == "lstm":
            self.rnn = nn.LSTM(input_size=10, hidden_size=hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=10, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, feat_seq):
        out, _ = self.rnn(feat_seq)
        return self.head(out[:, -1, :])


class RCADMMSolver(BaseSolver):
    name = "RC-ADMM"

    def __init__(self, cfg, projector: AffineProjector, cs: ConeSlices):
        super().__init__()
        self.cfg = cfg
        self.projector = projector
        self.cs = cs
        self.ctrl = RCADMMController(cfg)

    def map_params(self, raw, k, rho_prev, beta_prev):
        cfg = self.cfg
        B = raw.shape[0]
        device, dtype = raw.device, raw.dtype

        rho_base = torch.full((B,), float(cfg["rho_base"]), device=device, dtype=dtype)
        alpha_base = torch.full((B,), float(cfg["alpha_base"]), device=device, dtype=dtype)
        beta_base = torch.full((B,), float(cfg["beta_base"]), device=device, dtype=dtype)

        if cfg["envelope"]:
            q = torch.tanh(raw)
            delta = float(cfg["delta0"]) / (1.0 + (float(k) / float(cfg["k0"])) ** float(cfg["p_decay"]))
            rho = torch.exp(torch.log(rho_base) + delta * q[:, 0])
            alpha = alpha_base + float(cfg["alpha_delta_scale"]) * delta * q[:, 1]
            beta = torch.exp(torch.log(beta_base) + delta * q[:, 2])
        else:
            sig = torch.sigmoid(raw)
            rho = torch.exp(math.log(cfg["rho_min"]) + sig[:, 0] * (math.log(cfg["rho_max"]) - math.log(cfg["rho_min"])))
            alpha = cfg["alpha_min"] + sig[:, 1] * (cfg["alpha_max"] - cfg["alpha_min"])
            beta = torch.exp(math.log(cfg["beta_min"]) + sig[:, 2] * (math.log(cfg["beta_max"]) - math.log(cfg["beta_min"])))

        rho = rho.clamp(cfg["rho_min"], cfg["rho_max"])
        alpha = alpha.clamp(cfg["alpha_min"], cfg["alpha_max"])
        beta = beta.clamp(cfg["beta_min"], cfg["beta_max"])

        if cfg["growth"]:
            rho = torch.minimum(torch.maximum(rho, rho_prev / cfg["chi_rho"]), rho_prev * cfg["chi_rho"])
            beta = torch.minimum(torch.maximum(beta, beta_prev / cfg["chi_beta"]), beta_prev * cfg["chi_beta"])
            rho = rho.clamp(cfg["rho_min"], cfg["rho_max"])
            beta = beta.clamp(cfg["beta_min"], cfg["beta_max"])

        return rho, alpha, beta

    def forward(self, b, c, K: int):
        A = self.projector.A
        B, n = c.shape
        z = torch.zeros(B, n, device=c.device, dtype=c.dtype)
        u = torch.zeros_like(z)
        z_prev = z.clone()
        x = self.projector.proj(z - u, b)

        rho_prev = torch.full((B,), float(self.cfg["rho_base"]), device=c.device, dtype=c.dtype)
        alpha_prev = torch.full((B,), float(self.cfg["alpha_base"]), device=c.device, dtype=c.dtype)
        beta_prev = torch.full((B,), float(self.cfg["beta_base"]), device=c.device, dtype=c.dtype)

        feats = []
        for k in range(K):
            feat = residual_features(A, x, z, z_prev, b, c, rho_prev, alpha_prev, beta_prev, k, K, self.cfg).unsqueeze(1)
            feats.append(feat)
            raw = self.ctrl(torch.cat(feats, dim=1))
            rho, alpha, beta = self.map_params(raw, k, rho_prev, beta_prev)

            cbar = c / (torch.linalg.norm(c, dim=1, keepdim=True) + 1e-8)
            w = z - u - beta.view(-1, 1) * cbar
            x = self.projector.proj(w, b)
            xbar = alpha.view(-1, 1) * x + (1.0 - alpha.view(-1, 1)) * z
            z_new = proj_mixed_cone(xbar + u, self.cs)
            u_new = u + xbar - z_new

            z_prev = z
            z, u = z_new, u_new
            rho_prev = rho
            alpha_prev = alpha
            beta_prev = beta

        return z


# ============================================================
# Predictor
# ============================================================

class CostPredictor(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        mods = []
        d = d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*mods)

    def forward(self, s):
        return self.net(s)


def predict_cost(predictor: nn.Module, S: torch.Tensor, cfg) -> torch.Tensor:
    raw = predictor(S)
    raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    clip = cfg.get("cost_pred_clip", None)
    if clip is not None and clip > 0:
        raw = clip * torch.tanh(raw / clip)
    return raw


# ============================================================
# Training and evaluation
# ============================================================

def terminal_metrics(A, cs, z, b, c, x_star, opt_val):
    z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    c = torch.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    obj = objective(c, z)
    gap_signed = (obj - opt_val) / (1.0 + torch.abs(opt_val))
    gap_pos = torch.relu(gap_signed)
    eq = torch.linalg.norm(batch_matvec(A, z) - b, dim=1) / (1.0 + torch.linalg.norm(b, dim=1))
    cone = cone_violation(z, cs)
    dist = torch.linalg.norm(z - x_star, dim=1) / (1.0 + torch.linalg.norm(x_star, dim=1))
    return {
        "regret_pos": gap_pos,
        "regret_signed": gap_signed,
        "eq_vio": eq,
        "cone_vio": cone,
        "sol_dist": dist,
        "obj": obj,
    }


def dfl_loss(A, cs, z, b, c_true, c_pred, x_star, opt_val, cfg):
    m = terminal_metrics(A, cs, z, b, c_true, x_star, opt_val)
    # Use positive regret and first-order feasibility penalty.
    # This prevents the predictor from exploiting infeasible objective decrease.
    loss = m["regret_pos"].mean() + cfg["lambda_eq_dfl"] * m["eq_vio"].mean()
    if cfg.get("lambda_mse_aux", 0.0) > 0:
        loss = loss + cfg["lambda_mse_aux"] * F.mse_loss(c_pred, c_true)
    return loss


def sanitize_gradients(params):
    for p in params:
        if p.grad is not None:
            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)


def pretrain_rc_solver(cfg, solver: RCADMMSolver, train_loader, val_loader, K: int):
    print("[Pretrain solver] RC-ADMM controller")
    device = next(solver.parameters()).device
    opt = torch.optim.AdamW(solver.parameters(), lr=cfg["lr_solver"], weight_decay=cfg["weight_decay"])
    A, cs = solver.projector.A, solver.cs

    best_state = None
    best_val = float("inf")

    for ep in range(cfg["epochs_solver_pretrain"]):
        solver.train()
        for S, b, c, x_star, opt_val in train_loader:
            S, b, c, x_star, opt_val = to_device_batch((S, b, c, x_star, opt_val), device)
            opt.zero_grad(set_to_none=True)
            z = solver(b, c, K)
            loss = dfl_loss(A, cs, z, b, c, c, x_star, opt_val, cfg)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            sanitize_gradients(list(solver.parameters()))
            nn.utils.clip_grad_norm_(solver.parameters(), cfg["grad_clip"])
            opt.step()

        val_score = evaluate_solver_quality(cfg, solver, val_loader, K)["regret_pos_mean"]
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in solver.state_dict().items()}
        print(f"  epoch {ep+1:03d} val_regret={val_score:.4e}")

    if best_state is not None:
        solver.load_state_dict(best_state)
    return solver


@torch.no_grad()
def evaluate_solver_quality(cfg, solver, loader, K: int) -> Dict[str, float]:
    device = next(solver.parameters()).device if any(p.requires_grad for p in solver.parameters()) else cfg["_device_obj"]
    # for non-param modules, infer from A if available
    if hasattr(solver, "projector"):
        device = solver.projector.A.device
        A = solver.projector.A
        cs = solver.cs
    else:
        device = solver.A.device
        A = solver.A
        cs = solver.cs

    solver.eval()
    all_rows = []
    for S, b, c, x_star, opt_val in loader:
        S, b, c, x_star, opt_val = to_device_batch((S, b, c, x_star, opt_val), device)
        z = solver(b, c, K)
        met = terminal_metrics(A, cs, z, b, c, x_star, opt_val)
        all_rows.append({k: v.detach().cpu() for k, v in met.items() if torch.is_tensor(v)})
    out = {}
    for key in ["regret_pos", "regret_signed", "eq_vio", "cone_vio", "sol_dist"]:
        vals = torch.cat([r[key] for r in all_rows], dim=0)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_median"] = float(vals.median())
    return out


def train_predictor_for_method(cfg, method: str, solver: BaseSolver,
                               train_loader, val_loader, K: int,
                               d_in: int, d_out: int,
                               freeze_solver: bool = True):
    device = cfg["_device_obj"]
    predictor = CostPredictor(d_in, d_out, cfg["predictor_hidden"], cfg["predictor_layers"]).to(device)

    params = list(predictor.parameters())
    if (not freeze_solver) and any(p.requires_grad for p in solver.parameters()):
        params += list(solver.parameters())

    opt = torch.optim.AdamW(params, lr=cfg["lr_predictor"], weight_decay=cfg["weight_decay"])

    if freeze_solver:
        # Important:
        # Freezing parameters is not the same as setting the solver to eval mode.
        # During DFL training, gradients must still backpropagate through the
        # solver layer to the predicted cost c_pred. For cuDNN-backed GRU/LSTM,
        # backward through an RNN is only allowed when the RNN module is in
        # training mode. Therefore we keep solver.train() but disable parameter
        # gradients.
        solver.train()
        for p in solver.parameters():
            p.requires_grad_(False)
    else:
        solver.train()

    A = solver.projector.A if hasattr(solver, "projector") else solver.A
    cs = solver.cs

    best_state = None
    best_val = float("inf")

    for ep in range(cfg["epochs_predictor"]):
        predictor.train()
        # Keep the solver in training mode even when its parameters are frozen.
        # This is required for cuDNN RNN backward through the RC controller.
        solver.train()

        max_batches = cfg.get("max_train_batches_cvx", None) if "CVXPYLayer" in method else None
        for batch_idx, (S, b, c_true, x_star, opt_val) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            S, b, c_true, x_star, opt_val = to_device_batch((S, b, c_true, x_star, opt_val), device)
            opt.zero_grad(set_to_none=True)

            c_pred = predict_cost(predictor, S, cfg)
            z = solver(b, c_pred, K)
            loss = dfl_loss(A, cs, z, b, c_true, c_pred, x_star, opt_val, cfg)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            sanitize_gradients(params)
            nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()

        val = evaluate_dfl_method(cfg, predictor, solver, val_loader, K, method, measure_time=False)
        if val["regret_pos_mean"] < best_val:
            best_val = val["regret_pos_mean"]
            best_state = {
                "predictor": {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()},
                "solver": {k: v.detach().cpu().clone() for k, v in solver.state_dict().items()},
            }
        print(f"  [{method}] epoch {ep+1:03d} val_regret={val['regret_pos_mean']:.4e}")

    if best_state is not None:
        predictor.load_state_dict(best_state["predictor"])
        # only load solver if it was trainable
        if not freeze_solver:
            solver.load_state_dict(best_state["solver"])
    return predictor, solver


def train_mse_predictor(cfg, train_loader, val_loader, d_in, d_out):
    device = cfg["_device_obj"]
    predictor = CostPredictor(d_in, d_out, cfg["predictor_hidden"], cfg["predictor_layers"]).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=cfg["lr_predictor"], weight_decay=cfg["weight_decay"])

    best_state, best_val = None, float("inf")
    for ep in range(cfg["epochs_predictor"]):
        predictor.train()
        for S, b, c_true, x_star, opt_val in train_loader:
            S, c_true = S.to(device), c_true.to(device)
            opt.zero_grad(set_to_none=True)
            c_pred = predict_cost(predictor, S, cfg)
            loss = F.mse_loss(c_pred, c_true)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            sanitize_gradients(list(predictor.parameters()))
            nn.utils.clip_grad_norm_(predictor.parameters(), cfg["grad_clip"])
            opt.step()

        predictor.eval()
        losses = []
        with torch.no_grad():
            for S, b, c_true, x_star, opt_val in val_loader:
                S, c_true = S.to(device), c_true.to(device)
                losses.append(F.mse_loss(predictor(S), c_true).detach().cpu())
        val = float(torch.stack(losses).mean())
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
        print(f"  [MSE-PTO] epoch {ep+1:03d} val_mse={val:.4e}")

    if best_state is not None:
        predictor.load_state_dict(best_state)
    return predictor


@torch.no_grad()
def evaluate_dfl_method(cfg, predictor, solver, loader, K: int, method: str, measure_time: bool = True):
    device = cfg["_device_obj"]
    predictor.eval()
    solver.eval()

    A = solver.projector.A if hasattr(solver, "projector") else solver.A
    cs = solver.cs

    metrics_acc = []
    mse_vals = []
    times = []

    for S, b, c_true, x_star, opt_val in loader:
        S, b, c_true, x_star, opt_val = to_device_batch((S, b, c_true, x_star, opt_val), device)
        c_pred = predict_cost(predictor, S, cfg)
        mse_vals.append(F.mse_loss(c_pred, c_true, reduction="none").mean(dim=1).detach().cpu())

        if measure_time:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            z = solver(b, c_pred, K)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0 / S.shape[0])
        else:
            z = solver(b, c_pred, K)

        met = terminal_metrics(A, cs, z, b, c_true, x_star, opt_val)
        metrics_acc.append({k: v.detach().cpu() for k, v in met.items() if torch.is_tensor(v)})

    out = {"method": method}
    for key in ["regret_pos", "regret_signed", "eq_vio", "cone_vio", "sol_dist"]:
        vals = torch.cat([m[key] for m in metrics_acc], dim=0)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_median"] = float(vals.median())
    mse_all = torch.cat(mse_vals, dim=0)
    out["pred_mse_mean"] = float(mse_all.mean())
    out["pred_mse_median"] = float(mse_all.median())
    out["runtime_ms"] = float(np.mean(times)) if times else float("nan")
    return out


def tune_oracle_admm(cfg, projector, cs, val_loader, K):
    print("[Tune] OracleGrid-ADMM")
    best = None
    best_score = float("inf")
    for alpha in cfg["oracle_grid"]["alphas"]:
        for beta in cfg["oracle_grid"]["betas"]:
            solver = ADMMFixedSolver(projector, cs, alpha=alpha, beta=beta, name="OracleGrid-ADMM")
            score = evaluate_solver_quality(cfg, solver, val_loader, K)
            # Use objective gap first, eq as tie-breaker
            val = score["regret_pos_mean"] + 0.1 * score["eq_vio_mean"]
            print(f"  alpha={alpha:.2f} beta={beta:.2f}: score={val:.4e}")
            if val < best_score:
                best_score = val
                best = (alpha, beta)
    print(f"  selected alpha={best[0]:.3g}, beta={best[1]:.3g}")
    return best


def make_method_cfg(cfg, **overrides):
    new_cfg = copy.deepcopy(cfg)
    new_cfg.update(overrides)
    return new_cfg


def build_solvers(cfg, projector, cs, oracle_alpha_beta):
    alpha_o, beta_o = oracle_alpha_beta
    solvers = {
        "DFL-Fixed-ADMM": ADMMFixedSolver(projector, cs, cfg["alpha_base"], cfg["beta_base"], name="Fixed-ADMM"),
        "DFL-OracleGrid-ADMM": ADMMFixedSolver(projector, cs, alpha_o, beta_o, name="OracleGrid-ADMM"),
        "DFL-DRE-Anderson-DRS": DREAndersonDRSSolver(
            projector, cs, alpha_o, beta_o,
            omega=cfg["anderson_omega"],
            accept_tol=cfg["anderson_accept_tol"],
        ),
        "DFL-Stable-Learned-PDHG": StablePDHGSolver(
            projector.A, cs, theta=cfg["pdhg_theta"], safety=cfg["pdhg_safety"]
        ),
        "DFL-RC-Env-frozen": RCADMMSolver(make_method_cfg(cfg, envelope=True), projector, cs),
        "DFL-RC-Env-joint": RCADMMSolver(make_method_cfg(cfg, envelope=True), projector, cs),
        "DFL-RC-NoEnv-frozen": RCADMMSolver(make_method_cfg(cfg, envelope=False), projector, cs),
        "DFL-RC-NoEnv-joint": RCADMMSolver(make_method_cfg(cfg, envelope=False), projector, cs),
    }
    if cfg.get("enable_cvxpy_layer", False) and CVXPYLAYERS_AVAILABLE:
        solvers["DFL-CVXPYLayer"] = CVXPYLayerSolver(projector.A, cs, cfg)
        solvers["MSE-PTO-CVXPYLayer"] = solvers["DFL-CVXPYLayer"]
    return solvers

# ============================================================
# Main run
# ============================================================

def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def add_run_metadata(row, seed, K, train_K, method, oracle_alpha_beta, cfg):
    row = dict(row)
    row.update({
        "seed": seed,
        "K": K,
        "train_K": train_K,
        "method": method,
        "oracle_alpha": float(oracle_alpha_beta[0]),
        "oracle_beta": float(oracle_alpha_beta[1]),
        "rc_center_rho": float(cfg.get("rho_base", 1.0)),
        "rc_center_alpha": float(cfg.get("alpha_base", 1.0)),
        "rc_center_beta": float(cfg.get("beta_base", 1.0)),
    })
    return row


def train_and_eval_mse_pto(cfg, method, solver, train_loader, val_loader, test_loader, K, n):
    print(f"\n[Train] {method} predictor")
    predictor = train_mse_predictor(cfg, train_loader, val_loader, cfg["feature_dim"], n)
    return evaluate_dfl_method(cfg, predictor, solver, test_loader, K, method)


def maybe_pretrain_rc_pair(cfg, solvers, train_loader, val_loader, K, prefix):
    frozen = f"DFL-RC-{prefix}-frozen"
    joint = f"DFL-RC-{prefix}-joint"
    if cfg.get("pretrain_rc_solver", True) and frozen in solvers:
        solvers[frozen].to(cfg["_device_obj"])
        solvers[frozen] = pretrain_rc_solver(cfg, solvers[frozen], train_loader, val_loader, K)
        if joint in solvers:
            solvers[joint].to(cfg["_device_obj"])
            solvers[joint].load_state_dict(solvers[frozen].state_dict())


def run_single_seed(cfg, seed: int):
    set_seed(seed)
    device = torch.device(cfg["device"])
    dtype = get_dtype(cfg)
    cfg["_device_obj"] = device

    cs = make_cone_slices(cfg)
    n = cs.n
    m = cfg["mixed_dims"]["m_eq"]
    A = generate_A(m, n, cfg["mixed_dims"]["condition"], device, dtype)
    projector = AffineProjector.build(A)

    print("=" * 80)
    print(f"Seed={seed}, device={device}, dtype={dtype}, n={n}, m={m}")
    print(f"CVXPYLayer available={CVXPYLAYERS_AVAILABLE}, enabled={cfg.get('enable_cvxpy_layer', False)}")
    print("=" * 80)

    train_loader, val_loader, test_loader = make_loaders(cfg, A, cs)
    seed_rows = []

    for K in cfg["k_values"]:
        print("\n" + "#" * 80)
        print(f"[Depth] K=train_K={K}")
        print("#" * 80)

        oracle_alpha_beta = tune_oracle_admm(cfg, projector, cs, val_loader, K)
        # Per-K oracle-centered core for RC. rho is an auxiliary feedback scale;
        # alpha and beta are the tuned projection-splitting controls.
        k_cfg = copy.deepcopy(cfg)
        k_cfg["alpha_base"] = float(oracle_alpha_beta[0])
        k_cfg["beta_base"] = float(oracle_alpha_beta[1])
        k_cfg["rho_base"] = float(cfg.get("rho_base", 1.0))
        k_cfg["_device_obj"] = device

        solvers = build_solvers(k_cfg, projector, cs, oracle_alpha_beta)

        # Pretrain RC Env/NoEnv solver controllers at this exact K.
        maybe_pretrain_rc_pair(k_cfg, solvers, train_loader, val_loader, K, "Env")
        maybe_pretrain_rc_pair(k_cfg, solvers, train_loader, val_loader, K, "NoEnv")

        rows = []
        methods = list(k_cfg["methods"])
        if (not CVXPYLAYERS_AVAILABLE) or (not k_cfg.get("enable_cvxpy_layer", False)):
            methods = [m for m in methods if "CVXPYLayer" not in m]

        for method in methods:
            if method == "MSE-PTO-OracleADMM":
                solver = solvers["DFL-OracleGrid-ADMM"].to(device)
                row = train_and_eval_mse_pto(k_cfg, method, solver, train_loader, val_loader, test_loader, K, n)
            elif method == "MSE-PTO-CVXPYLayer":
                if "MSE-PTO-CVXPYLayer" not in solvers:
                    print(f"[Skip] {method}: cvxpylayers unavailable")
                    continue
                row = train_and_eval_mse_pto(k_cfg, method, solvers["MSE-PTO-CVXPYLayer"], train_loader, val_loader, test_loader, K, n)
            else:
                if method not in solvers:
                    print(f"[Skip] Unknown method {method}")
                    continue
                print(f"\n[Train] {method}")
                solver = solvers[method].to(device)
                freeze = True
                if method.endswith("-joint"):
                    freeze = False
                predictor, trained_solver = train_predictor_for_method(
                    k_cfg, method, solver, train_loader, val_loader, K,
                    k_cfg["feature_dim"], n, freeze_solver=freeze
                )
                row = evaluate_dfl_method(k_cfg, predictor, trained_solver, test_loader, K, method)

            row = add_run_metadata(row, seed, K, K, method, oracle_alpha_beta, k_cfg)
            # Parameter count includes predictor and trainable solver parameters if available.
            row["n_solver_params"] = count_trainable_params(solvers.get(method, nn.Module())) if method in solvers else 0
            rows.append(row)
            seed_rows.append(row)
            print("[Test]", row)

        # Per-K partial save.
        out_dir = Path(k_cfg["out_dir"]) / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / f"dfl_perK_seed{seed}_K{K}.csv", index=False)
        pd.DataFrame(seed_rows).to_csv(out_dir / f"dfl_perK_seed{seed}_partial.csv", index=False)

    return pd.DataFrame(seed_rows)


def summarize_results(df: pd.DataFrame, out_dir: Path):
    metric_cols = [
        "regret_pos_mean", "regret_signed_mean", "eq_vio_mean", "cone_vio_mean",
        "sol_dist_mean", "pred_mse_mean", "runtime_ms",
    ]
    agg = df.groupby(["method", "K"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"{c}_avg": (c, "mean") for c in metric_cols},
        **{f"{c}_std": (c, "std") for c in metric_cols},
    )
    agg.to_csv(out_dir / "dfl_summary_by_K.csv", index=False)
    return agg


def run_dfl_experiment(cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in cfg["seeds"]:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["seed"] = seed
        df_seed = run_single_seed(seed_cfg, seed)
        all_rows.append(df_seed)
        pd.concat(all_rows, ignore_index=True).to_csv(out_dir / "dfl_all_results_partial.csv", index=False)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out_dir / "dfl_all_results.csv", index=False)
    summary = summarize_results(df, out_dir)
    print("\nSaved:", out_dir / "dfl_all_results.csv")
    print("Saved:", out_dir / "dfl_summary_by_K.csv")
    print(summary)
    return df, summary


if __name__ == "__main__":
    run_dfl_experiment(CONFIG)
