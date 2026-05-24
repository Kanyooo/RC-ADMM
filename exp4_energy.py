"""
EXP4: Application-oriented renewable energy dispatch experiment for RC-ADMM.

CONFIG-driven script, no command-line arguments.

Task:
    24-hour single-bus renewable microgrid dispatch with generator, battery,
    renewable curtailment, load shedding, and reserve adequacy.

Variables per scenario:
    pg[t]       conventional generation
    pch[t]      battery charging power
    pdis[t]     battery discharging power
    e[t]        battery state of charge, t=0..H
    curt[t]     renewable curtailment
    shed[t]     load shedding
    sres[t]     reserve slack, enforcing pg_max - pg[t] + pdis[t] >= reserve[t]

Constraints:
    power balance:
        pg + pdis - pch - curt + shed = load - ren
    storage dynamics:
        e[t+1] = e[t] + eta_ch*pch[t] - pdis[t]/eta_dis
    initial and terminal SOC:
        e[0] = e0, e[H] = e0
    reserve:
        -pg + pdis - sres = reserve - pg_max, sres >= 0
    bounds:
        variables are projected onto scenario-dependent boxes.

Objective:
    quadratic generation cost + linear generation/charge/discharge/curtail/shed costs.

Methods:
    Fixed-ADMM
    OracleGrid-ADMM
    Spectral-AADMM
    DRE-Anderson-DRS
    Stable-PDHG
    RC-ADMM

Metrics:
    cost gap against a long-run oracle ADMM reference,
    equality violation, load shedding, curtailment, runtime.
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "seed": 11,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float32",
    "out_dir": "./dispatch_outputs",

    # Quick smoke test by default. Set False for full experiment.
    "quick": False,

    # Horizon and solver depths
    "H": 24,
    "K_values": [5, 10, 15, 20],
    "K_ref": 250,

    # Dataset
    "n_train": 2048,
    "n_val": 512,
    "n_test": 512,
    "batch_size": 1024,
    "scenario_level": "hard",  # "medium" or "hard"

    # Dispatch device limits, per-unit scale
    "pg_max": 1.35,
    "pch_max": 0.35,
    "pdis_max": 0.35,
    "e_min": 0.10,
    "e_max": 1.20,
    "e0_min": 0.35,
    "e0_max": 0.85,
    "eta_ch": 0.95,
    "eta_dis": 0.95,

    # Cost coefficients
    "a_pg": 0.08,
    "b_pg": 1.00,
    "c_ch": 0.015,
    "c_dis": 0.015,
    "c_curt": 0.10,
    "c_shed": 8.00,
    "c_res_slack": 0.00,

    # Scenario generation
    "load_base": 0.95,
    "load_amp_medium": 0.25,
    "load_amp_hard": 0.38,
    "pv_amp_medium": [0.35, 0.80],
    "pv_amp_hard": [0.15, 0.55],
    "wind_amp_medium": [0.25, 0.70],
    "wind_amp_hard": [0.10, 0.45],
    "noise_std": 0.03,
    "reserve_factor_medium": 0.10,
    "reserve_factor_hard": 0.18,

    # ADMM base parameters
    "alpha_base": 1.0,
    "beta_base": 1.0,
    "rho_base": 1.0,

    # Oracle fixed ADMM tuning
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
    "envelope": True,
    "growth": True,
    "delta0": 2.0,
    "k0": 10.0,
    "p_decay": 1.2,
    "alpha_delta_scale": 0.6,
    "chi_rho": 5.0,
    "chi_beta": 5.0,

    # Training loss weights
    "lambda_eq_train": 30.0,
    "lambda_dispatch_train": 1.0,

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
        "H": 12,
        "K_values": [5, 10],
        "K_ref": 80,
        "n_train": 256,
        "n_val": 96,
        "n_test": 96,
        "batch_size": 64,
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


def safe_norm(x, dim=1, eps=1e-12):
    return torch.linalg.norm(x, dim=dim).clamp_min(eps)


# ============================================================
# Dispatch problem construction
# ============================================================

@dataclass
class DispatchProblem:
    H: int
    n: int
    m: int
    A: torch.Tensor
    Pinv_AAt: torch.Tensor
    Hmat: torch.Tensor
    c_lin: torch.Tensor

    pg: slice
    pch: slice
    pdis: slice
    e: slice
    curt: slice
    shed: slice
    sres: slice


def build_dispatch_problem(cfg, device, dtype) -> DispatchProblem:
    H = int(cfg["H"])
    idx = 0
    pg = slice(idx, idx + H); idx += H
    pch = slice(idx, idx + H); idx += H
    pdis = slice(idx, idx + H); idx += H
    e = slice(idx, idx + H + 1); idx += H + 1
    curt = slice(idx, idx + H); idx += H
    shed = slice(idx, idx + H); idx += H
    sres = slice(idx, idx + H); idx += H
    n = idx

    # Equalities:
    # H power balance + H storage dynamics + 1 init SOC + 1 terminal SOC + H reserve.
    m = 3 * H + 2
    A = torch.zeros(m, n, device=device, dtype=dtype)

    row = 0
    # Power balance: pg + pdis - pch - curt + shed = load - ren
    for t in range(H):
        A[row, pg.start + t] = 1.0
        A[row, pdis.start + t] = 1.0
        A[row, pch.start + t] = -1.0
        A[row, curt.start + t] = -1.0
        A[row, shed.start + t] = 1.0
        row += 1

    # Storage dynamics: e[t+1]-e[t]-eta_ch*pch + pdis/eta_dis = 0
    eta_ch = float(cfg["eta_ch"])
    eta_dis = float(cfg["eta_dis"])
    for t in range(H):
        A[row, e.start + t + 1] = 1.0
        A[row, e.start + t] = -1.0
        A[row, pch.start + t] = -eta_ch
        A[row, pdis.start + t] = 1.0 / eta_dis
        row += 1

    # Initial SOC e[0]=e0
    A[row, e.start + 0] = 1.0
    row += 1

    # Terminal SOC e[H]=e0
    A[row, e.start + H] = 1.0
    row += 1

    # Reserve: -pg + pdis - sres = reserve - pg_max
    for t in range(H):
        A[row, pg.start + t] = -1.0
        A[row, pdis.start + t] = 1.0
        A[row, sres.start + t] = -1.0
        row += 1

    Pinv_AAt = torch.linalg.pinv(A @ A.t())

    # Objective: 0.5 z^T Hmat z + c_lin^T z.
    Hmat = torch.zeros(n, n, device=device, dtype=dtype)
    c_lin = torch.zeros(n, device=device, dtype=dtype)

    # 0.5 * Hii * pg^2 = a_pg * pg^2 => Hii = 2*a_pg
    for t in range(H):
        Hmat[pg.start + t, pg.start + t] = 2.0 * float(cfg["a_pg"])
        c_lin[pg.start + t] = float(cfg["b_pg"])
        c_lin[pch.start + t] = float(cfg["c_ch"])
        c_lin[pdis.start + t] = float(cfg["c_dis"])
        c_lin[curt.start + t] = float(cfg["c_curt"])
        c_lin[shed.start + t] = float(cfg["c_shed"])
        c_lin[sres.start + t] = float(cfg["c_res_slack"])

    return DispatchProblem(H, n, m, A, Pinv_AAt, Hmat, c_lin, pg, pch, pdis, e, curt, shed, sres)


def project_affine(problem: DispatchProblem, v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    Av_minus_b = batch_matvec(problem.A, v) - b
    correction = (Av_minus_b @ problem.Pinv_AAt.t()) @ problem.A
    return v - correction


def project_box(v: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(v, lb), ub)


def dispatch_cost(problem: DispatchProblem, z: torch.Tensor) -> torch.Tensor:
    q = 0.5 * (z * (z @ problem.Hmat.t())).sum(dim=1)
    l = (z * problem.c_lin.view(1, -1)).sum(dim=1)
    return q + l


def equality_violation(problem: DispatchProblem, z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(batch_matvec(problem.A, z) - b, dim=1) / (1.0 + torch.linalg.norm(b, dim=1))


def box_violation(z: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    viol = torch.relu(lb - z) + torch.relu(z - ub)
    return torch.linalg.norm(viol, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))


# ============================================================
# Scenario generation
# ============================================================

@dataclass
class DispatchDataset:
    b: torch.Tensor
    lb: torch.Tensor
    ub: torch.Tensor
    load: torch.Tensor
    ren: torch.Tensor
    reserve: torch.Tensor
    e0: torch.Tensor


def generate_profiles(cfg, N: int, device, dtype) -> DispatchDataset:
    H = int(cfg["H"])
    t = torch.arange(H, device=device, dtype=dtype).view(1, H)
    level = cfg.get("scenario_level", "hard").lower()

    if level == "hard":
        load_amp = float(cfg["load_amp_hard"])
        pv_low, pv_high = cfg["pv_amp_hard"]
        wind_low, wind_high = cfg["wind_amp_hard"]
        reserve_factor = float(cfg["reserve_factor_hard"])
    else:
        load_amp = float(cfg["load_amp_medium"])
        pv_low, pv_high = cfg["pv_amp_medium"]
        wind_low, wind_high = cfg["wind_amp_medium"]
        reserve_factor = float(cfg["reserve_factor_medium"])

    # Load profile: morning/evening pattern + noise.
    phase = 2 * math.pi * torch.rand(N, 1, device=device, dtype=dtype)
    load_base = float(cfg["load_base"]) * (0.90 + 0.25 * torch.rand(N, 1, device=device, dtype=dtype))
    load = load_base + load_amp * (0.5 + 0.5 * torch.sin(2 * math.pi * (t - 7.0) / 24.0 + phase))
    load += float(cfg["noise_std"]) * torch.randn(N, H, device=device, dtype=dtype)
    load = load.clamp_min(0.25)

    # PV profile.
    pv_amp = pv_low + (pv_high - pv_low) * torch.rand(N, 1, device=device, dtype=dtype)
    pv_shape = torch.sin(math.pi * (t - 6.0) / 12.0).clamp_min(0.0)
    pv = pv_amp * pv_shape
    pv += float(cfg["noise_std"]) * torch.randn(N, H, device=device, dtype=dtype)
    pv = pv.clamp_min(0.0)

    # Wind profile.
    wind_amp = wind_low + (wind_high - wind_low) * torch.rand(N, 1, device=device, dtype=dtype)
    wind_phase = 2 * math.pi * torch.rand(N, 1, device=device, dtype=dtype)
    wind = wind_amp * (0.55 + 0.30 * torch.sin(2 * math.pi * t / 24.0 + wind_phase))
    wind += float(cfg["noise_std"]) * torch.randn(N, H, device=device, dtype=dtype)
    wind = wind.clamp_min(0.0)

    ren = pv + wind

    # Reserve requirement as variability-aware margin.
    reserve = reserve_factor * (load + 0.5 * ren)
    reserve = reserve.clamp_min(0.02)

    e0 = float(cfg["e0_min"]) + (float(cfg["e0_max"]) - float(cfg["e0_min"])) * torch.rand(N, 1, device=device, dtype=dtype)

    return load, ren, reserve, e0


def make_dispatch_dataset(cfg, problem: DispatchProblem, N: int, device, dtype) -> DispatchDataset:
    H = problem.H
    load, ren, reserve, e0 = generate_profiles(cfg, N, device, dtype)

    b = torch.zeros(N, problem.m, device=device, dtype=dtype)
    row = 0
    # power balance rhs
    b[:, row:row+H] = load - ren
    row += H
    # dynamics rhs
    b[:, row:row+H] = 0.0
    row += H
    # init SOC
    b[:, row] = e0.squeeze(1)
    row += 1
    # terminal SOC
    b[:, row] = e0.squeeze(1)
    row += 1
    # reserve rhs: reserve - pg_max
    b[:, row:row+H] = reserve - float(cfg["pg_max"])

    # Bounds
    lb = torch.zeros(N, problem.n, device=device, dtype=dtype)
    ub = torch.full((N, problem.n), 1e6, device=device, dtype=dtype)

    ub[:, problem.pg] = float(cfg["pg_max"])
    ub[:, problem.pch] = float(cfg["pch_max"])
    ub[:, problem.pdis] = float(cfg["pdis_max"])
    lb[:, problem.e] = float(cfg["e_min"])
    ub[:, problem.e] = float(cfg["e_max"])
    ub[:, problem.curt] = ren
    ub[:, problem.shed] = load
    ub[:, problem.sres] = float(cfg["pg_max"]) + float(cfg["pdis_max"])

    return DispatchDataset(b=b, lb=lb, ub=ub, load=load, ren=ren, reserve=reserve, e0=e0)


# ============================================================
# Solvers
# ============================================================

class DispatchSolver(nn.Module):
    name = "base"
    def forward(self, data: DispatchDataset, K: int) -> torch.Tensor:
        raise NotImplementedError


class FixedADMMSolver(DispatchSolver):
    def __init__(self, problem: DispatchProblem, alpha: float, beta: float, name="Fixed-ADMM"):
        super().__init__()
        self.problem = problem
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.name = name

    def step(self, z, dual, data: DispatchDataset, alpha, beta):
        p = self.problem
        grad = z @ p.Hmat.t() + p.c_lin.view(1, -1)
        grad = grad / (torch.linalg.norm(grad, dim=1, keepdim=True) + 1e-8)
        w = z - dual - beta * grad
        x = project_affine(p, w, data.b)
        xbar = alpha * x + (1.0 - alpha) * z
        z_new = project_box(xbar + dual, data.lb, data.ub)
        dual_new = dual + xbar - z_new
        return x, z_new, dual_new

    def forward(self, data: DispatchDataset, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.n, device=data.b.device, dtype=data.b.dtype)
        dual = torch.zeros_like(z)
        for _ in range(K):
            _, z, dual = self.step(z, dual, data, self.alpha, self.beta)
        return z


class SpectralAADMM(FixedADMMSolver):
    def __init__(self, problem: DispatchProblem, alpha=1.0, beta=1.0, cfg=None):
        super().__init__(problem, alpha, beta, name="Spectral-AADMM")
        self.cfg = cfg or CONFIG

    def forward(self, data: DispatchDataset, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.n, device=data.b.device, dtype=data.b.dtype)
        dual = torch.zeros_like(z)
        beta = torch.full((B,), self.beta, device=data.b.device, dtype=data.b.dtype)

        beta_min = float(self.cfg["spectral_beta_min"])
        beta_max = float(self.cfg["spectral_beta_max"])
        growth = float(self.cfg["spectral_growth"])

        for _ in range(K):
            # Use per-batch beta by explicit step.
            p = self.problem
            grad = z @ p.Hmat.t() + p.c_lin.view(1, -1)
            grad = grad / (torch.linalg.norm(grad, dim=1, keepdim=True) + 1e-8)
            w = z - dual - beta.view(-1, 1) * grad
            x = project_affine(p, w, data.b)
            xbar = self.alpha * x + (1.0 - self.alpha) * z
            z_new = project_box(xbar + dual, data.lb, data.ub)
            dual_new = dual + xbar - z_new

            eq = equality_violation(p, z_new, data.b)
            mov = torch.linalg.norm(z_new - z, dim=1) / (1.0 + torch.linalg.norm(z_new, dim=1))
            ratio = torch.sqrt((eq + 1e-8) / (mov + 1e-8))
            beta_cand = torch.clamp(beta * ratio, beta / growth, beta * growth)
            beta = torch.clamp(beta_cand, beta_min, beta_max)

            z, dual = z_new, dual_new

        return z


class DREAndersonDRS(FixedADMMSolver):
    def __init__(self, problem: DispatchProblem, alpha: float, beta: float, omega=0.25, accept_tol=1.05):
        super().__init__(problem, alpha, beta, name="DRE-Anderson-DRS")
        self.omega = float(omega)
        self.accept_tol = float(accept_tol)

    def monitor(self, z, dual, data):
        return equality_violation(self.problem, z, data.b) + 0.05 * torch.linalg.norm(dual, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))

    def forward(self, data: DispatchDataset, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.n, device=data.b.device, dtype=data.b.dtype)
        dual = torch.zeros_like(z)
        z_prev = z.clone()
        dual_prev = dual.clone()

        for k in range(K):
            _, z_base, dual_base = self.step(z, dual, data, self.alpha, self.beta)
            if k == 0:
                z_prev, dual_prev = z, dual
                z, dual = z_base, dual_base
                continue

            z_acc = project_box(z_base + self.omega * (z_base - z_prev), data.lb, data.ub)
            dual_acc = dual_base + self.omega * (dual_base - dual_prev)

            mon_base = self.monitor(z_base, dual_base, data)
            mon_acc = self.monitor(z_acc, dual_acc, data)
            accept = (mon_acc <= self.accept_tol * mon_base).float().view(-1, 1)

            z_next = accept * z_acc + (1.0 - accept) * z_base
            dual_next = accept * dual_acc + (1.0 - accept) * dual_base

            z_prev, dual_prev = z, dual
            z, dual = z_next, dual_next

        return z


class StablePDHG(DispatchSolver):
    name = "Stable-PDHG"
    def __init__(self, problem: DispatchProblem, safety=0.95, theta=0.8, dual_clip=50.0):
        super().__init__()
        self.problem = problem
        self.safety = float(safety)
        self.theta = float(theta)
        self.dual_clip = float(dual_clip)
        with torch.no_grad():
            norm_A = torch.linalg.matrix_norm(problem.A, ord=2).item()
            norm_H = torch.linalg.matrix_norm(problem.Hmat, ord=2).item()
        norm_A_sq = max(norm_A * norm_A, 1e-8)
        norm_H = max(norm_H, 1e-8)
        self.tau = 0.5 / (norm_H + norm_A_sq + 1e-8)
        self.sigma = min(0.5 / (norm_A_sq + 1e-8), self.safety / (self.tau * norm_A_sq + 1e-8))

    def forward(self, data: DispatchDataset, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        p = self.problem
        z = torch.zeros(B, p.n, device=data.b.device, dtype=data.b.dtype)
        zbar = z.clone()
        y = torch.zeros(B, p.m, device=data.b.device, dtype=data.b.dtype)

        for _ in range(K):
            y = y + self.sigma * (batch_matvec(p.A, zbar) - data.b)
            y = torch.clamp(y, -self.dual_clip, self.dual_clip)
            z_old = z
            grad = z @ p.Hmat.t() + p.c_lin.view(1, -1) + y @ p.A
            z = project_box(z - self.tau * grad, data.lb, data.ub)
            zbar = z + self.theta * (z - z_old)

        return z


def rc_features(problem, z, z_prev, dual, data, rho_prev):
    eq = equality_violation(problem, z, data.b)
    mov = rho_prev * torch.linalg.norm(z - z_prev, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
    cons = torch.linalg.norm(dual, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
    cost = dispatch_cost(problem, z) / (1.0 + torch.linalg.norm(z, dim=1))
    grad = z @ problem.Hmat.t() + problem.c_lin.view(1, -1)
    gnorm = torch.linalg.norm(grad, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
    return torch.stack([
        torch.log1p(eq),
        torch.log1p(mov),
        torch.log1p(cons),
        torch.log1p(torch.relu(cost)),
        torch.log1p(gnorm),
    ], dim=1)


class RCController(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg["rc_hidden"])
        kind = cfg.get("rc_controller", "gru").lower()
        if kind == "lstm":
            self.rnn = nn.LSTM(input_size=5, hidden_size=hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=5, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, seq):
        out, _ = self.rnn(seq)
        return self.head(out[:, -1, :])


class RCADMM(DispatchSolver):
    name = "RC-ADMM"
    def __init__(self, cfg, problem):
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

    def forward(self, data: DispatchDataset, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.n, device=data.b.device, dtype=data.b.dtype)
        dual = torch.zeros_like(z)
        z_prev = z.clone()
        rho_prev = torch.full((B,), float(self.cfg["rho_base"]), device=data.b.device, dtype=data.b.dtype)
        beta_prev = torch.full((B,), float(self.cfg["beta_base"]), device=data.b.device, dtype=data.b.dtype)

        feats = []
        for k in range(K):
            feat = rc_features(self.problem, z, z_prev, dual, data, rho_prev).unsqueeze(1)
            feats.append(feat)
            raw = self.ctrl(torch.cat(feats, dim=1))
            rho, alpha, beta = self.map_params(raw, k, rho_prev, beta_prev)

            grad = z @ self.problem.Hmat.t() + self.problem.c_lin.view(1, -1)
            grad = grad / (torch.linalg.norm(grad, dim=1, keepdim=True) + 1e-8)
            w = z - dual - beta.view(-1, 1) * grad
            x = project_affine(self.problem, w, data.b)
            xbar = alpha.view(-1, 1) * x + (1.0 - alpha.view(-1, 1)) * z
            z_new = project_box(xbar + dual, data.lb, data.ub)
            dual_new = dual + xbar - z_new

            z_prev = z
            z, dual = z_new, dual_new
            rho_prev = rho
            beta_prev = beta

        return z


# ============================================================
# Evaluation and training
# ============================================================

def make_batches(data: DispatchDataset, batch_size: int, shuffle: bool):
    N = data.b.shape[0]
    idx = torch.randperm(N, device=data.b.device) if shuffle else torch.arange(N, device=data.b.device)
    for i in range(0, N, batch_size):
        ids = idx[i:i+batch_size]
        yield DispatchDataset(
            b=data.b[ids], lb=data.lb[ids], ub=data.ub[ids],
            load=data.load[ids], ren=data.ren[ids], reserve=data.reserve[ids], e0=data.e0[ids]
        )


def train_loss(problem, z, data, cfg):
    cost = dispatch_cost(problem, z)
    scale = 1.0 + data.load.sum(dim=1)
    eq = equality_violation(problem, z, data.b)
    return cfg["lambda_dispatch_train"] * (cost / scale).mean() + cfg["lambda_eq_train"] * eq.mean()


@torch.no_grad()
def eval_open_loop(problem, solver, data, K, cfg):
    solver.eval()
    z = solver(data, K)
    cost = dispatch_cost(problem, z)
    eq = equality_violation(problem, z, data.b)
    shed = z[:, problem.shed].sum(dim=1)
    curt = z[:, problem.curt].sum(dim=1)
    return {
        "cost_mean": float(cost.mean()),
        "eq_mean": float(eq.mean()),
        "shed_mean": float(shed.mean()),
        "curt_mean": float(curt.mean()),
        "score": float((cost / (1.0 + data.load.sum(dim=1))).mean() + cfg["lambda_eq_train"] * eq.mean()),
    }


def tune_oracle(problem, cfg, val_data, K):
    print(f"[Tune] OracleGrid K={K}")
    best = (cfg["alpha_base"], cfg["beta_base"])
    best_score = float("inf")
    for alpha in cfg["oracle_grid"]["alphas"]:
        for beta in cfg["oracle_grid"]["betas"]:
            solver = FixedADMMSolver(problem, alpha, beta, name="OracleGrid-ADMM")
            s = eval_open_loop(problem, solver, val_data, K, cfg)
            print(f"  alpha={alpha:.2f}, beta={beta:.2f}: score={s['score']:.4e}, cost={s['cost_mean']:.4e}, eq={s['eq_mean']:.4e}")
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
        print(f"  epoch {ep+1:03d}: score={s['score']:.4e}, cost={s['cost_mean']:.4e}, eq={s['eq_mean']:.4e}")

    if best_state is not None:
        solver.load_state_dict(best_state)
    return solver


@torch.no_grad()
def compute_reference(problem, ref_solver, data, K_ref):
    ref_solver.eval()
    zref = ref_solver(data, K_ref)
    cref = dispatch_cost(problem, zref)
    return zref, cref


@torch.no_grad()
def evaluate_method(problem, solver, data, K, ref_cost, method):
    device = data.b.device
    solver.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    z = solver(data, K)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    cost = dispatch_cost(problem, z)
    gap = torch.relu((cost - ref_cost) / (1.0 + torch.abs(ref_cost)))
    eq = equality_violation(problem, z, data.b)
    shed = z[:, problem.shed].sum(dim=1)
    curt = z[:, problem.curt].sum(dim=1)
    boxv = box_violation(z, data.lb, data.ub)

    return {
        "method": method,
        "cost_gap_mean": float(gap.mean()),
        "cost_gap_median": float(gap.median()),
        "cost_mean": float(cost.mean()),
        "eq_vio_mean": float(eq.mean()),
        "eq_vio_median": float(eq.median()),
        "box_vio_mean": float(boxv.mean()),
        "shed_mean": float(shed.mean()),
        "shed_median": float(shed.median()),
        "curt_mean": float(curt.mean()),
        "curt_median": float(curt.median()),
        "runtime_ms": float((t1 - t0) * 1000.0 / data.b.shape[0]),
    }


# ============================================================
# Main
# ============================================================

def run_dispatch_experiment(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    dtype = get_dtype(cfg)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = build_dispatch_problem(cfg, device, dtype)
    print("=" * 80)
    print(f"Device={device}, dtype={dtype}, quick={cfg['quick']}")
    print(f"H={cfg['H']}, n={problem.n}, m={problem.m}, K_values={cfg['K_values']}, level={cfg['scenario_level']}")
    print("=" * 80)

    train_data = make_dispatch_dataset(cfg, problem, cfg["n_train"], device, dtype)
    val_data = make_dispatch_dataset(cfg, problem, cfg["n_val"], device, dtype)
    test_data = make_dispatch_dataset(cfg, problem, cfg["n_test"], device, dtype)

    rows = []
    for K in cfg["K_values"]:
        print("\n" + "#" * 80)
        print(f"# Dispatch experiment K={K}")
        print("#" * 80)

        alpha_o, beta_o = tune_oracle(problem, cfg, val_data, K)

        # Reference: long-run oracle ADMM with the same tuned parameters.
        ref_solver = FixedADMMSolver(problem, alpha_o, beta_o, name="Long-Oracle-ADMM")
        _, ref_cost = compute_reference(problem, ref_solver, test_data, cfg["K_ref"])

        solvers = {
            "Fixed-ADMM": FixedADMMSolver(problem, cfg["alpha_base"], cfg["beta_base"], name="Fixed-ADMM").to(device),
            "OracleGrid-ADMM": FixedADMMSolver(problem, alpha_o, beta_o, name="OracleGrid-ADMM").to(device),
            "Spectral-AADMM": SpectralAADMM(problem, cfg["alpha_base"], cfg["beta_base"], cfg=cfg).to(device),
            "DRE-Anderson-DRS": DREAndersonDRS(problem, alpha_o, beta_o, cfg["anderson_omega"], cfg["anderson_accept_tol"]).to(device),
            "Stable-PDHG": StablePDHG(problem, cfg["pdhg_safety"], cfg["pdhg_theta"], cfg["pdhg_dual_clip"]).to(device),
        }

        if "RC-ADMM" in cfg["methods"]:
            solvers["RC-ADMM"] = train_rc(problem, cfg, train_data, val_data, K).to(device)

        for method in cfg["methods"]:
            if method not in solvers:
                continue
            print(f"[Evaluate] K={K} {method}")
            r = evaluate_method(problem, solvers[method], test_data, K, ref_cost, method)
            r["K"] = int(K)
            rows.append(r)
            print(r)

    df = pd.DataFrame(rows)
    front = ["K", "method"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    path = out_dir / "dispatch_results.csv"
    df.to_csv(path, index=False)
    print("\nSaved:", path)
    print(df)
    return df


if __name__ == "__main__":
    run_dispatch_experiment(CONFIG)
