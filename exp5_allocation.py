"""
EXP5: Edge-AI inference resource allocation with rotated-SOC constraints.

CONFIG-driven script, no command-line arguments.

Application:
    Multiple DNN inference tasks share limited edge computing and communication
    resources. Each task has a compute workload a_i, input size d_i, deadline
    tau_i, and priority pi_i. The allocator assigns CPU/GPU compute f_i and
    bandwidth b_i, while controlling latency slack.

Convex conic model:
    t_comp_i >= a_i / f_i,
    t_comm_i >= d_i / b_i,

which is represented by rotated SOC constraints:
    (t_comp_i, f_i, sqrt(2a_i)) in Q_r,
    (t_comm_i, b_i, sqrt(2d_i)) in Q_r.

Additional affine constraints:
    t_comp_i + t_comm_i - s_i + l_i = tau_i,
    sum_i f_i + r_F = F_max,
    sum_i b_i + r_B = B_max,

with nonnegative slack/spare variables s_i, l_i, r_F, r_B.
The objective penalizes resource usage, latency, and priority-weighted
deadline slack.

Compared methods:
    Fixed-ADMM, OracleGrid-ADMM, Spectral-AADMM, DRE-Anderson-DRS,
    Stable-PDHG, and RC-ADMM.

Outputs:
    edge_inference_results.csv with cost gap, equality violation,
    cone violation, delay violation, miss rate, resource violation, and runtime.
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "seed": 23,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float32",
    "out_dir": "./edge_inference_outputs",

    # Smoke test by default. Set False for paper-level runs.
    "quick": False,

    # Workload setting
    "n_tasks": 32,
    "scenario_level": "hard",  # "medium" or "hard"

    # Dataset
    "n_train": 2048,
    "n_val": 512,
    "n_test": 512,
    "batch_size": 1024,

    # Solver depths
    "K_values": [5, 10, 15, 20],
    "K_ref": 250,

    # Resource capacities
    "F_max_medium": 16.0,
    "B_max_medium": 10.0,
    "F_max_hard": 12.0,
    "B_max_hard": 7.5,

    # Synthetic DNN workload distributions
    # Workload a and data size d are lognormal-like positive variables.
    "a_mean_medium": 0.30,
    "a_mean_hard": 0.42,
    "d_mean_medium": 0.18,
    "d_mean_hard": 0.26,
    "workload_sigma": 0.45,

    # Deadlines
    "tau_low_medium": 1.40,
    "tau_high_medium": 2.60,
    "tau_low_hard": 0.95,
    "tau_high_hard": 1.85,

    # Objective coefficients
    "c_compute": 0.025,
    "c_bandwidth": 0.025,
    "c_latency": 0.03,
    "slack_penalty_base": 6.0,

    # ADMM base parameters
    "alpha_base": 1.0,
    "beta_base": 1.0,
    "rho_base": 1.0,

    # Oracle fixed-parameter tuning
    "oracle_grid": {
        "alphas": [0.8, 1.0, 1.3, 1.6],
        "betas": [0.2, 0.5, 1.0, 2.0, 4.0],
    },

    # Spectral-AADMM
    "spectral_beta_min": 0.05,
    "spectral_beta_max": 8.0,
    "spectral_growth": 2.0,

    # DRE-Anderson
    "anderson_omega": 0.25,
    "anderson_accept_tol": 1.05,

    # PDHG
    "pdhg_safety": 0.95,
    "pdhg_theta": 0.8,
    "pdhg_dual_clip": 50.0,

    # RC controller
    "epochs_rc": 80,
    "lr_rc": 1e-3,
    "weight_decay": 1e-5,
    "grad_clip": 5.0,
    "rc_hidden": 64,
    "rc_controller": "gru",

    # RC parameterization
    "alpha_min": 0.2,
    "alpha_max": 1.8,
    "beta_min": 1e-3,
    "beta_max": 10.0,
    "rho_min": 1e-3,
    "rho_max": 1e3,
    "envelope": False,
    "growth": False,
    "delta0": 2.0,
    "k0": 10.0,
    "p_decay": 1.2,
    "alpha_delta_scale": 0.6,
    "chi_rho": 5.0,
    "chi_beta": 5.0,

    # Training and evaluation score weights
    "lambda_eq_train": 25.0,
    "lambda_cone_train": 1.0,
    "lambda_delay_train": 10.0,
    "lambda_res_train": 10.0,

    "lambda_eq_eval": 10.0,
    "lambda_delay_eval": 10.0,
    "lambda_res_eval": 10.0,

    # Miss if actual latency exceeds deadline by this tolerance.
    "miss_tol": 1e-3,

    # Methods
    "methods": [
        "Fixed-ADMM",
        "OracleGrid-ADMM",
        "Spectral-AADMM",
        "DRE-Anderson-DRS",
        "Stable-PDHG",
        "RC-ADMM",
    ],
}


if CONFIG["quick"]:
    CONFIG.update({
        "n_tasks": 12,
        "n_train": 256,
        "n_val": 96,
        "n_test": 96,
        "batch_size": 64,
        "K_values": [5, 10],
        "K_ref": 100,
        "epochs_rc": 3,
    })


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(cfg):
    return torch.float64 if cfg.get("dtype") == "float64" else torch.float32


def batch_matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x @ A.t()


# ============================================================
# Cone layout and projections
# ============================================================

@dataclass
class EdgeLayout:
    n_tasks: int
    comp: List[slice]
    comm: List[slice]
    nonneg: slice
    slack: slice
    spare: slice
    rF_idx: int
    rB_idx: int
    n: int
    m: int


def make_layout(n_tasks: int) -> EdgeLayout:
    idx = 0
    comp, comm = [], []
    for _ in range(n_tasks):
        comp.append(slice(idx, idx + 3))
        idx += 3
    for _ in range(n_tasks):
        comm.append(slice(idx, idx + 3))
        idx += 3

    nonneg_start = idx
    slack = slice(idx, idx + n_tasks)
    idx += n_tasks
    spare = slice(idx, idx + n_tasks)
    idx += n_tasks
    rF_idx = idx
    idx += 1
    rB_idx = idx
    idx += 1
    nonneg = slice(nonneg_start, idx)

    n = idx
    m = 3 * n_tasks + 2
    return EdgeLayout(
        n_tasks=n_tasks, comp=comp, comm=comm, nonneg=nonneg,
        slack=slack, spare=spare, rF_idx=rF_idx, rB_idx=rB_idx,
        n=n, m=m
    )


def proj_soc_block(x: torch.Tensor) -> torch.Tensor:
    t = x[:, :1]
    v = x[:, 1:]
    nv = torch.linalg.norm(v, dim=1, keepdim=True)
    inside = nv <= t
    negative = nv <= -t
    scale = 0.5 * (1.0 + t / nv.clamp_min(1e-12))
    mid = torch.cat([0.5 * (nv + t), scale * v], dim=1)
    zeros = torch.zeros_like(x)
    return torch.where(inside, x, torch.where(negative, zeros, mid))


def proj_rotated_soc_block(x: torch.Tensor) -> torch.Tensor:
    # Rotated SOC: u >= 0, v >= 0, 2uv >= ||w||^2.
    sqrt2 = math.sqrt(2.0)
    u = x[:, :1]
    v = x[:, 1:2]
    w = x[:, 2:]
    y = torch.cat([(u + v) / sqrt2, (u - v) / sqrt2, w], dim=1)
    py = proj_soc_block(y)
    y0, y1, yw = py[:, :1], py[:, 1:2], py[:, 2:]
    return torch.cat([(y0 + y1) / sqrt2, (y0 - y1) / sqrt2, yw], dim=1)


def proj_product_cone(x: torch.Tensor, layout: EdgeLayout) -> torch.Tensor:
    parts = []
    for sl in layout.comp:
        parts.append(proj_rotated_soc_block(x[:, sl]))
    for sl in layout.comm:
        parts.append(proj_rotated_soc_block(x[:, sl]))
    parts.append(x[:, layout.nonneg].clamp_min(0.0))
    return torch.cat(parts, dim=1)


def cone_violation(x: torch.Tensor, layout: EdgeLayout) -> torch.Tensor:
    px = proj_product_cone(x, layout)
    return torch.linalg.norm(x - px, dim=1) / (1.0 + torch.linalg.norm(x, dim=1))


# ============================================================
# Problem construction and data generation
# ============================================================

@dataclass
class EdgeProblem:
    layout: EdgeLayout
    A: torch.Tensor
    Pinv_AAt: torch.Tensor


@dataclass
class EdgeData:
    b: torch.Tensor      # [B,m]
    c: torch.Tensor      # [B,n]
    a: torch.Tensor      # compute workload [B,N]
    d: torch.Tensor      # data size [B,N]
    tau: torch.Tensor    # deadline [B,N]
    priority: torch.Tensor  # [B,N]
    Fmax: torch.Tensor   # [B]
    Bmax: torch.Tensor   # [B]


def build_edge_problem(layout: EdgeLayout, device, dtype) -> EdgeProblem:
    N = layout.n_tasks
    A = torch.zeros(layout.m, layout.n, device=device, dtype=dtype)
    row = 0

    # comp[i][2] = sqrt(2a_i)
    for i in range(N):
        A[row, layout.comp[i].start + 2] = 1.0
        row += 1

    # comm[i][2] = sqrt(2d_i)
    for i in range(N):
        A[row, layout.comm[i].start + 2] = 1.0
        row += 1

    # deadline: t_comp + t_comm - slack + spare = tau
    for i in range(N):
        A[row, layout.comp[i].start + 0] = 1.0
        A[row, layout.comm[i].start + 0] = 1.0
        A[row, layout.slack.start + i] = -1.0
        A[row, layout.spare.start + i] = 1.0
        row += 1

    # compute resource: sum f_i + rF = Fmax
    for i in range(N):
        A[row, layout.comp[i].start + 1] = 1.0
    A[row, layout.rF_idx] = 1.0
    row += 1

    # bandwidth resource: sum b_i + rB = Bmax
    for i in range(N):
        A[row, layout.comm[i].start + 1] = 1.0
    A[row, layout.rB_idx] = 1.0
    row += 1

    assert row == layout.m
    Pinv_AAt = torch.linalg.pinv(A @ A.t())
    return EdgeProblem(layout=layout, A=A, Pinv_AAt=Pinv_AAt)


def affine_project(problem: EdgeProblem, v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    Av_minus_b = batch_matvec(problem.A, v) - b
    correction = (Av_minus_b @ problem.Pinv_AAt.t()) @ problem.A
    return v - correction


def equality_violation(problem: EdgeProblem, z: torch.Tensor, data: EdgeData) -> torch.Tensor:
    return torch.linalg.norm(batch_matvec(problem.A, z) - data.b, dim=1) / (1.0 + torch.linalg.norm(data.b, dim=1))


def generate_edge_dataset(cfg, problem: EdgeProblem, M: int, device, dtype) -> EdgeData:
    layout = problem.layout
    N = layout.n_tasks
    level = cfg.get("scenario_level", "hard").lower()

    if level == "hard":
        Fmax = float(cfg["F_max_hard"])
        Bmax = float(cfg["B_max_hard"])
        a_mean = float(cfg["a_mean_hard"])
        d_mean = float(cfg["d_mean_hard"])
        tau_low = float(cfg["tau_low_hard"])
        tau_high = float(cfg["tau_high_hard"])
    else:
        Fmax = float(cfg["F_max_medium"])
        Bmax = float(cfg["B_max_medium"])
        a_mean = float(cfg["a_mean_medium"])
        d_mean = float(cfg["d_mean_medium"])
        tau_low = float(cfg["tau_low_medium"])
        tau_high = float(cfg["tau_high_medium"])

    sigma = float(cfg["workload_sigma"])

    # Lognormal with approximate mean scaling.
    a = a_mean * torch.exp(sigma * torch.randn(M, N, device=device, dtype=dtype) - 0.5 * sigma ** 2)
    d = d_mean * torch.exp(sigma * torch.randn(M, N, device=device, dtype=dtype) - 0.5 * sigma ** 2)

    # Priorities in {1,2,3}.
    priority = torch.randint(1, 4, (M, N), device=device).to(dtype)

    # Tighter deadlines for high-priority tasks.
    base_tau = tau_low + (tau_high - tau_low) * torch.rand(M, N, device=device, dtype=dtype)
    tau = base_tau / (1.0 + 0.12 * (priority - 1.0))

    F = Fmax * (0.92 + 0.16 * torch.rand(M, device=device, dtype=dtype))
    B = Bmax * (0.92 + 0.16 * torch.rand(M, device=device, dtype=dtype))

    b = torch.zeros(M, problem.layout.m, device=device, dtype=dtype)
    row = 0
    b[:, row:row+N] = torch.sqrt(2.0 * a)
    row += N
    b[:, row:row+N] = torch.sqrt(2.0 * d)
    row += N
    b[:, row:row+N] = tau
    row += N
    b[:, row] = F
    row += 1
    b[:, row] = B

    c = torch.zeros(M, problem.layout.n, device=device, dtype=dtype)
    for i in range(N):
        # latency terms
        c[:, layout.comp[i].start + 0] = float(cfg["c_latency"])
        c[:, layout.comm[i].start + 0] = float(cfg["c_latency"])
        # resource usage terms
        c[:, layout.comp[i].start + 1] = float(cfg["c_compute"])
        c[:, layout.comm[i].start + 1] = float(cfg["c_bandwidth"])
    c[:, layout.slack] = float(cfg["slack_penalty_base"]) * priority

    return EdgeData(b=b, c=c, a=a, d=d, tau=tau, priority=priority, Fmax=F, Bmax=B)


# ============================================================
# Metrics
# ============================================================

def objective(z: torch.Tensor, data: EdgeData) -> torch.Tensor:
    return (z * data.c).sum(dim=1)


def actual_delay(z: torch.Tensor, layout: EdgeLayout) -> torch.Tensor:
    B = z.shape[0]
    N = layout.n_tasks
    tc = torch.zeros(B, N, device=z.device, dtype=z.dtype)
    tm = torch.zeros_like(tc)
    for i in range(N):
        tc[:, i] = z[:, layout.comp[i].start + 0]
        tm[:, i] = z[:, layout.comm[i].start + 0]
    return tc + tm


def resource_violation(z: torch.Tensor, data: EdgeData, layout: EdgeLayout) -> torch.Tensor:
    f_sum = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    b_sum = torch.zeros_like(f_sum)
    for i in range(layout.n_tasks):
        f_sum += z[:, layout.comp[i].start + 1]
        b_sum += z[:, layout.comm[i].start + 1]
    vio = torch.relu(f_sum - data.Fmax) + torch.relu(b_sum - data.Bmax)
    denom = 1.0 + data.Fmax + data.Bmax
    return vio / denom


def edge_metrics(problem: EdgeProblem, z: torch.Tensor, data: EdgeData, ref_cost=None, cfg=CONFIG):
    layout = problem.layout
    cost = objective(z, data)
    eq = equality_violation(problem, z, data)
    cone = cone_violation(z, layout)
    delay = actual_delay(z, layout)
    delay_vio_mat = torch.relu(delay - data.tau)
    delay_vio = delay_vio_mat.mean(dim=1)
    miss = (delay_vio_mat > float(cfg["miss_tol"])).float().mean(dim=1)
    res_vio = resource_violation(z, data, layout)
    slack_sum = z[:, layout.slack].sum(dim=1)

    out = {
        "cost": cost,
        "eq": eq,
        "cone": cone,
        "delay_vio": delay_vio,
        "miss_rate": miss,
        "resource_vio": res_vio,
        "slack_sum": slack_sum,
    }
    if ref_cost is not None:
        out["cost_gap"] = torch.relu((cost - ref_cost) / (1.0 + torch.abs(ref_cost)))
    return out


# ============================================================
# Solvers
# ============================================================

class EdgeSolver(nn.Module):
    name = "base"
    def forward(self, data: EdgeData, K: int) -> torch.Tensor:
        raise NotImplementedError


class FixedADMMSolver(EdgeSolver):
    def __init__(self, cfg, problem: EdgeProblem, alpha: float, beta: float, name="Fixed-ADMM"):
        super().__init__()
        self.cfg = cfg
        self.problem = problem
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.name = name

    def step(self, z, u, data: EdgeData, alpha, beta):
        cbar = data.c / (torch.linalg.norm(data.c, dim=1, keepdim=True) + 1e-8)
        w = z - u - beta * cbar
        x = affine_project(self.problem, w, data.b)
        xbar = alpha * x + (1.0 - alpha) * z
        z_new = proj_product_cone(xbar + u, self.problem.layout)
        u_new = u + xbar - z_new
        return x, z_new, u_new

    def forward(self, data: EdgeData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
        u = torch.zeros_like(z)
        for _ in range(K):
            _, z, u = self.step(z, u, data, self.alpha, self.beta)
        return z


class SpectralAADMM(FixedADMMSolver):
    def __init__(self, cfg, problem: EdgeProblem, alpha=1.0, beta=1.0):
        super().__init__(cfg, problem, alpha, beta, name="Spectral-AADMM")

    def forward(self, data: EdgeData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
        u = torch.zeros_like(z)
        beta = torch.full((B,), self.beta, device=data.b.device, dtype=data.b.dtype)
        cbar = data.c / (torch.linalg.norm(data.c, dim=1, keepdim=True) + 1e-8)

        beta_min = float(self.cfg["spectral_beta_min"])
        beta_max = float(self.cfg["spectral_beta_max"])
        growth = float(self.cfg["spectral_growth"])

        for _ in range(K):
            w = z - u - beta.view(-1, 1) * cbar
            x = affine_project(self.problem, w, data.b)
            xbar = self.alpha * x + (1.0 - self.alpha) * z
            z_new = proj_product_cone(xbar + u, self.problem.layout)
            u_new = u + xbar - z_new

            eq = equality_violation(self.problem, z_new, data)
            mov = torch.linalg.norm(z_new - z, dim=1) / (1.0 + torch.linalg.norm(z_new, dim=1))
            ratio = torch.sqrt((eq + 1e-8) / (mov + 1e-8))
            beta = torch.clamp(torch.clamp(beta * ratio, beta / growth, beta * growth), beta_min, beta_max)

            z, u = z_new, u_new
        return z


class DREAndersonDRS(FixedADMMSolver):
    def __init__(self, cfg, problem: EdgeProblem, alpha, beta, omega=0.25, accept_tol=1.05):
        super().__init__(cfg, problem, alpha, beta, name="DRE-Anderson-DRS")
        self.omega = float(omega)
        self.accept_tol = float(accept_tol)

    def monitor(self, z, u, data):
        return equality_violation(self.problem, z, data) + 0.05 * torch.linalg.norm(u, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))

    def forward(self, data: EdgeData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
        u = torch.zeros_like(z)
        z_prev = z.clone()
        u_prev = u.clone()

        for k in range(K):
            _, z_base, u_base = self.step(z, u, data, self.alpha, self.beta)
            if k == 0:
                z_prev, u_prev = z, u
                z, u = z_base, u_base
                continue

            z_acc = proj_product_cone(z_base + self.omega * (z_base - z_prev), self.problem.layout)
            u_acc = u_base + self.omega * (u_base - u_prev)
            mon_base = self.monitor(z_base, u_base, data)
            mon_acc = self.monitor(z_acc, u_acc, data)
            accept = (mon_acc <= self.accept_tol * mon_base).float().view(-1, 1)

            z_next = accept * z_acc + (1.0 - accept) * z_base
            u_next = accept * u_acc + (1.0 - accept) * u_base
            z_prev, u_prev = z, u
            z, u = z_next, u_next
        return z


class StablePDHG(EdgeSolver):
    name = "Stable-PDHG"
    def __init__(self, cfg, problem: EdgeProblem):
        super().__init__()
        self.cfg = cfg
        self.problem = problem
        with torch.no_grad():
            norm_A = torch.linalg.matrix_norm(problem.A, ord=2).item()
        norm_A_sq = max(norm_A ** 2, 1e-8)
        self.tau = 0.5 / (norm_A_sq + 1.0)
        self.sigma = min(0.5 / norm_A_sq, float(cfg["pdhg_safety"]) / (self.tau * norm_A_sq + 1e-8))
        self.theta = float(cfg["pdhg_theta"])
        self.dual_clip = float(cfg["pdhg_dual_clip"])

    def forward(self, data: EdgeData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
        zbar = z.clone()
        y = torch.zeros(B, self.problem.layout.m, device=data.b.device, dtype=data.b.dtype)
        for _ in range(K):
            y = y + self.sigma * (batch_matvec(self.problem.A, zbar) - data.b)
            y = torch.clamp(y, -self.dual_clip, self.dual_clip)
            z_old = z
            grad = y @ self.problem.A + data.c
            z = proj_product_cone(z - self.tau * grad, self.problem.layout)
            zbar = z + self.theta * (z - z_old)
        return z


def rc_features(problem: EdgeProblem, z, z_prev, u, data, rho_prev):
    met = edge_metrics(problem, z, data, cfg=CONFIG)
    mov = rho_prev * torch.linalg.norm(z - z_prev, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
    dual = torch.linalg.norm(u, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
    obj = torch.abs(objective(z, data)) / (1.0 + torch.linalg.norm(z, dim=1))
    return torch.stack([
        torch.log1p(met["eq"]),
        torch.log1p(met["cone"]),
        torch.log1p(mov),
        torch.log1p(met["delay_vio"]),
        torch.log1p(obj + dual),
    ], dim=1)


class RCController(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg["rc_hidden"])
        if cfg.get("rc_controller", "gru").lower() == "lstm":
            self.rnn = nn.LSTM(input_size=5, hidden_size=hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=5, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, seq):
        out, _ = self.rnn(seq)
        return self.head(out[:, -1, :])


class RCADMM(EdgeSolver):
    name = "RC-ADMM"
    def __init__(self, cfg, problem: EdgeProblem):
        super().__init__()
        self.cfg = cfg
        self.problem = problem
        self.ctrl = RCController(cfg)

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
        return rho, alpha, beta

    def forward(self, data: EdgeData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
        u = torch.zeros_like(z)
        z_prev = z.clone()
        rho_prev = torch.full((B,), float(self.cfg["rho_base"]), device=data.b.device, dtype=data.b.dtype)
        beta_prev = torch.full((B,), float(self.cfg["beta_base"]), device=data.b.device, dtype=data.b.dtype)

        feats = []
        for k in range(K):
            feat = rc_features(self.problem, z, z_prev, u, data, rho_prev).unsqueeze(1)
            feats.append(feat)
            raw = self.ctrl(torch.cat(feats, dim=1))
            rho, alpha, beta = self.map_params(raw, k, rho_prev, beta_prev)

            cbar = data.c / (torch.linalg.norm(data.c, dim=1, keepdim=True) + 1e-8)
            w = z - u - beta.view(-1, 1) * cbar
            x = affine_project(self.problem, w, data.b)
            xbar = alpha.view(-1, 1) * x + (1.0 - alpha.view(-1, 1)) * z
            z_new = proj_product_cone(xbar + u, self.problem.layout)
            u_new = u + xbar - z_new

            z_prev = z
            z, u = z_new, u_new
            rho_prev = rho
            beta_prev = beta
        return z


# ============================================================
# Training, tuning, and evaluation
# ============================================================

def make_batches(data: EdgeData, batch_size: int, shuffle: bool):
    M = data.b.shape[0]
    ids = torch.randperm(M, device=data.b.device) if shuffle else torch.arange(M, device=data.b.device)
    for i in range(0, M, batch_size):
        idx = ids[i:i+batch_size]
        yield EdgeData(
            b=data.b[idx], c=data.c[idx], a=data.a[idx], d=data.d[idx],
            tau=data.tau[idx], priority=data.priority[idx],
            Fmax=data.Fmax[idx], Bmax=data.Bmax[idx]
        )


def train_loss(problem, z, data, cfg):
    met = edge_metrics(problem, z, data, cfg=cfg)
    scale = 1.0 + data.tau.sum(dim=1)
    return (
        (met["cost"] / scale).mean()
        + cfg["lambda_eq_train"] * met["eq"].mean()
        + cfg["lambda_cone_train"] * met["cone"].mean()
        + cfg["lambda_delay_train"] * met["delay_vio"].mean()
        + cfg["lambda_res_train"] * met["resource_vio"].mean()
    )


@torch.no_grad()
def eval_open_loop(problem, solver, data, K, cfg):
    solver.eval()
    z = solver(data, K)
    met = edge_metrics(problem, z, data, cfg=cfg)
    score = (
        (met["cost"] / (1.0 + data.tau.sum(dim=1))).mean()
        + cfg["lambda_eq_train"] * met["eq"].mean()
        + cfg["lambda_delay_train"] * met["delay_vio"].mean()
        + cfg["lambda_res_train"] * met["resource_vio"].mean()
    )
    return {
        "score": float(score),
        "cost": float(met["cost"].mean()),
        "eq": float(met["eq"].mean()),
        "delay": float(met["delay_vio"].mean()),
        "miss": float(met["miss_rate"].mean()),
        "res": float(met["resource_vio"].mean()),
    }


def tune_oracle(problem, cfg, val_data, K):
    print(f"[Tune] OracleGrid K={K}")
    best = (cfg["alpha_base"], cfg["beta_base"])
    best_score = float("inf")
    for alpha in cfg["oracle_grid"]["alphas"]:
        for beta in cfg["oracle_grid"]["betas"]:
            solver = FixedADMMSolver(cfg, problem, alpha, beta, name="OracleGrid-ADMM")
            s = eval_open_loop(problem, solver, val_data, K, cfg)
            print(f"  alpha={alpha:.2f}, beta={beta:.2f}: score={s['score']:.4e}, cost={s['cost']:.4e}, eq={s['eq']:.4e}, delay={s['delay']:.4e}")
            if s["score"] < best_score:
                best_score = s["score"]
                best = (alpha, beta)
    print(f"  selected alpha={best[0]:.3g}, beta={best[1]:.3g}")
    return best


def train_rc(problem, cfg, train_data, val_data, K):
    print(f"[Train] RC-ADMM K={K}")
    solver = RCADMM(cfg, problem).to(train_data.b.device)
    opt = torch.optim.AdamW(solver.parameters(), lr=cfg["lr_rc"], weight_decay=cfg["weight_decay"])
    best_state = None
    best_score = float("inf")

    for ep in range(cfg["epochs_rc"]):
        solver.train()
        for batch in make_batches(train_data, cfg["batch_size"], shuffle=True):
            opt.zero_grad(set_to_none=True)
            z = solver(batch, K)
            loss = train_loss(problem, z, batch, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(solver.parameters(), cfg["grad_clip"])
            opt.step()

        s = eval_open_loop(problem, solver, val_data, K, cfg)
        if s["score"] < best_score:
            best_score = s["score"]
            best_state = {k: v.detach().cpu().clone() for k, v in solver.state_dict().items()}
        print(f"  epoch {ep+1:03d}: score={s['score']:.4e}, cost={s['cost']:.4e}, eq={s['eq']:.4e}, delay={s['delay']:.4e}, miss={s['miss']:.3f}")

    if best_state is not None:
        solver.load_state_dict(best_state)
    return solver


@torch.no_grad()
def compute_reference(problem, ref_solver, data, K_ref):
    ref_solver.eval()
    zref = ref_solver(data, K_ref)
    ref = edge_metrics(problem, zref, data, cfg=CONFIG)
    print(
        f"[Reference] K_ref={K_ref}, cost={float(ref['cost'].mean()):.4e}, "
        f"eq={float(ref['eq'].mean()):.4e}, delay={float(ref['delay_vio'].mean()):.4e}, "
        f"miss={float(ref['miss_rate'].mean()):.3f}"
    )
    return zref, ref["cost"]


@torch.no_grad()
def evaluate_method(problem, solver, data, K, ref_cost, method, cfg):
    device = data.b.device
    solver.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    z = solver(data, K)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    met = edge_metrics(problem, z, data, ref_cost=ref_cost, cfg=cfg)
    score = (
        met["cost_gap"]
        + cfg["lambda_eq_eval"] * met["eq"]
        + cfg["lambda_delay_eval"] * met["delay_vio"]
        + cfg["lambda_res_eval"] * met["resource_vio"]
    )

    return {
        "method": method,
        "score_mean": float(score.mean()),
        "score_median": float(score.median()),
        "cost_gap_mean": float(met["cost_gap"].mean()),
        "cost_gap_median": float(met["cost_gap"].median()),
        "cost_mean": float(met["cost"].mean()),
        "eq_vio_mean": float(met["eq"].mean()),
        "cone_vio_mean": float(met["cone"].mean()),
        "delay_vio_mean": float(met["delay_vio"].mean()),
        "miss_rate_mean": float(met["miss_rate"].mean()),
        "resource_vio_mean": float(met["resource_vio"].mean()),
        "slack_sum_mean": float(met["slack_sum"].mean()),
        "runtime_ms": float((t1 - t0) * 1000.0 / data.b.shape[0]),
    }


# ============================================================
# Main
# ============================================================

def run_edge_experiment(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    dtype = get_dtype(cfg)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = make_layout(int(cfg["n_tasks"]))
    problem = build_edge_problem(layout, device, dtype)

    print("=" * 80)
    print(f"Device={device}, dtype={dtype}, quick={cfg['quick']}")
    print(f"N_tasks={layout.n_tasks}, n={layout.n}, m={layout.m}, K_values={cfg['K_values']}, level={cfg['scenario_level']}")
    print("=" * 80)

    train_data = generate_edge_dataset(cfg, problem, cfg["n_train"], device, dtype)
    val_data = generate_edge_dataset(cfg, problem, cfg["n_val"], device, dtype)
    test_data = generate_edge_dataset(cfg, problem, cfg["n_test"], device, dtype)

    rows = []
    for K in cfg["K_values"]:
        print("\n" + "#" * 80)
        print(f"# Edge inference allocation K={K}")
        print("#" * 80)

        alpha_o, beta_o = tune_oracle(problem, cfg, val_data, K)

        ref_solver = FixedADMMSolver(cfg, problem, alpha_o, beta_o, name="Long-Oracle-ADMM").to(device)
        _, ref_cost = compute_reference(problem, ref_solver, test_data, cfg["K_ref"])

        solvers = {
            "Fixed-ADMM": FixedADMMSolver(cfg, problem, cfg["alpha_base"], cfg["beta_base"], name="Fixed-ADMM").to(device),
            "OracleGrid-ADMM": FixedADMMSolver(cfg, problem, alpha_o, beta_o, name="OracleGrid-ADMM").to(device),
            "Spectral-AADMM": SpectralAADMM(cfg, problem, cfg["alpha_base"], cfg["beta_base"]).to(device),
            "DRE-Anderson-DRS": DREAndersonDRS(cfg, problem, alpha_o, beta_o, cfg["anderson_omega"], cfg["anderson_accept_tol"]).to(device),
            "Stable-PDHG": StablePDHG(cfg, problem).to(device),
        }

        if "RC-ADMM" in cfg["methods"]:
            solvers["RC-ADMM"] = train_rc(problem, cfg, train_data, val_data, K).to(device)

        for method in cfg["methods"]:
            if method not in solvers:
                continue
            print(f"[Evaluate] K={K} {method}")
            r = evaluate_method(problem, solvers[method], test_data, K, ref_cost, method, cfg)
            r["K"] = int(K)
            rows.append(r)
            print(r)

    df = pd.DataFrame(rows)
    front = ["K", "method"]
    df = df[front + [c for c in df.columns if c not in front]]
    path = out_dir / "edge_inference_results.csv"
    df.to_csv(path, index=False)

    print("\nSaved:", path)
    print(df)
    return df


if __name__ == "__main__":
    run_edge_experiment(CONFIG)
