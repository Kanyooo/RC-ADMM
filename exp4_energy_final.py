"""
EXP4 (final): Stress-medium forecast-aware multi-bus renewable microgrid dispatch.

CONFIG-driven script, no command-line arguments.

Task
----
A 6-bus renewable microgrid schedules generators, batteries, renewable
curtailment, load shedding, line flows, voltage angles, and reserve slack over a
24-hour horizon. The solver sees forecast load/renewable profiles and is
reported on both forecast feasibility and realized operating risk under moderate forecast
errors. The stress level is calibrated so that the long-run reference is operationally
meaningful while short-depth solvers still face a nontrivial feasibility/reliability trade-off.

Convex model
------------
The optimization problem is a box-constrained convex QP with affine equalities:
    - bus-level DC power balance,
    - line-flow equations f_l = b_l(theta_i - theta_j),
    - reference angle theta_0 = 0,
    - battery SOC dynamics with initial/terminal SOC,
    - system reserve adequacy with nonnegative reserve slack.

The box projection handles generator, battery, SOC, curtailment, shedding,
reserve slack, voltage-angle, and line-flow limits. This keeps the solver layer
in the same affine-projection + box-projection ADMM/DRS family as the main
paper, but the application is no longer a toy single-bus dispatch.

Compared methods
----------------
    Fixed-ADMM
    OracleGrid-ADMM
    Spectral-AADMM
    DRE-Anderson-DRS
    Stable-PDHG
    RC-ADMM-Env
    RC-ADMM-NoEnv

Protocol
--------
For each seed and each K in {5,10,15}:
    1. tune OracleGrid-ADMM on validation data at K;
    2. train RC-ADMM-Env and RC-ADMM-NoEnv at the same K;
    3. evaluate all solvers on held-out test instances at the same K.

Metrics
-------
    cost_gap against a long-run oracle ADMM reference,
    forecast equality violation,
    box violation,
    realized load-shedding rate,
    realized renewable-curtailment rate,
    realized reserve-shortfall rate,
    runtime per instance.

CostGap is useful only together with feasibility/reliability metrics. The script
therefore also reports a mean-threshold operational status.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float32",
    "out_dir": "./forecast_dispatch_stress_outputs",
    "quick": False,

    # Paper-level protocol requested by the user.
    "seeds": [0, 1, 2],
    "K_values": [5, 10, 15],
    "K_ref": 300,

    # Dataset.
    "n_train": 1536,
    "n_val": 384,
    "n_test": 512,
    "batch_size": 512,

    # Microgrid.
    "H": 24,
    "n_bus": 6,
    "scenario_level": "stress",

    # Limits.
    "theta_max": 0.60,
    "pg_max": [1.85, 1.45],
    "pg_min": [0.00, 0.00],
    "ramp": [0.70, 0.60],
    "pch_max": [0.42, 0.36],
    "pdis_max": [0.42, 0.36],
    "e_min": [0.15, 0.12],
    "e_max": [1.35, 1.15],
    "eta_ch": 0.95,
    "eta_dis": 0.95,

    # Hard network flow limits. Smaller values increase congestion.
    "flow_limit_scale_medium": 1.00,
    "flow_limit_scale_stress": 1.15,
    "flow_limit_scale_hard": 0.72,

    # Objective coefficients.
    "a_pg": [0.08, 0.10],
    "b_pg": [1.00, 1.10],
    "c_ch": 0.015,
    "c_dis": 0.015,
    "c_curt": 0.08,
    "c_shed": 14.00,
    "c_reserve_short": 10.00,
    "c_flow": 0.002,

    # Scenario generation.
    "load_base": [0.36, 0.43, 0.39, 0.46, 0.36, 0.32],
    "load_amp_medium": 0.22,
    "load_amp_stress": 0.20,
    "load_amp_hard": 0.36,
    "ren_amp_medium": [0.55, 0.45, 0.50],
    "ren_amp_stress": [0.72, 0.58, 0.62],
    "ren_amp_hard": [0.34, 0.28, 0.32],
    "forecast_noise_medium": 0.05,
    "forecast_noise_stress": 0.045,
    "forecast_noise_hard": 0.12,
    "profile_noise": 0.025,
    "reserve_factor_medium": 0.10,
    "reserve_factor_stress": 0.12,
    "reserve_factor_hard": 0.18,

    # ADMM base parameters.
    "alpha_base": 1.0,
    "beta_base": 1.0,
    "rho_base": 1.0,

    # Oracle fixed-parameter tuning.
    "oracle_grid": {
        "alphas": [0.8, 1.0, 1.3, 1.6],
        "betas": [0.2, 0.5, 1.0, 2.0, 4.0],
    },

    # Spectral-AADMM.
    "spectral_beta_min": 0.05,
    "spectral_beta_max": 8.0,
    "spectral_growth": 2.0,

    # DRE-Anderson.
    "anderson_omega": 0.25,
    "anderson_accept_tol": 1.05,

    # PDHG.
    "pdhg_safety": 0.95,
    "pdhg_theta": 0.8,
    "pdhg_dual_clip": 50.0,

    # RC controller.
    "epochs_rc": 60,
    "lr_rc": 1e-3,
    "weight_decay": 1e-5,
    "grad_clip": 5.0,
    "rc_hidden": 64,
    "rc_controller": "gru",

    # RC parameterization.
    "alpha_min": 0.2,
    "alpha_max": 1.8,
    "beta_min": 1e-3,
    "beta_max": 10.0,
    "rho_min": 1e-3,
    "rho_max": 1e3,
    "delta0": 2.0,
    "k0": 60.0,
    "p_decay": 1.20,
    "alpha_delta_scale": 0.25,
    "chi_rho": 10.0,
    "chi_beta": 10.0,

    # Training loss weights.  The loss is application-aware; equality is no
    # longer excessively dominant.
    "lambda_eq_train": 6.0,
    "lambda_shed_train": 12.0,
    "lambda_reserve_train": 8.0,
    "lambda_curt_train": 0.05,
    "lambda_box_train": 1.0,

    # Validation score weights for oracle tuning and checkpoint selection.
    "lambda_eq_val": 6.0,
    "lambda_shed_val": 12.0,
    "lambda_reserve_val": 8.0,
    "lambda_curt_val": 0.05,
    "lambda_box_val": 1.0,

    # Mean-threshold operational status used only for reporting.
    "pass_eq_tol": 0.035,
    "pass_box_tol": 1e-5,
    "pass_shed_rate_tol": 0.12,
    "pass_reserve_rate_tol": 0.08,
}

if CONFIG["quick"]:
    CONFIG.update({
        "H": 12,
        "K_values": [5],
        "K_ref": 80,
        "seeds": [0],
        "n_train": 128,
        "n_val": 64,
        "n_test": 64,
        "batch_size": 64,
        "epochs_rc": 2,
    })


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(cfg):
    return torch.float64 if cfg.get("dtype") == "float64" else torch.float32


def batch_matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x @ A.t()


def scalar_tensor(x: float, device, dtype):
    return torch.tensor(float(x), device=device, dtype=dtype)


# ============================================================
# Problem layout
# ============================================================

@dataclass
class MicrogridLayout:
    H: int
    n_bus: int
    n_line: int
    n_gen: int
    n_bat: int
    n_ren: int
    lines: List[Tuple[int, int]]
    susceptance: torch.Tensor
    base_flow_limit: torch.Tensor
    gen_bus: List[int]
    bat_bus: List[int]
    ren_bus: List[int]
    pg: slice
    pch: slice
    pdis: slice
    e: slice
    curt: slice
    shed: slice
    sres: slice
    theta: slice
    flow: slice
    n: int
    m: int
    row_balance: slice
    row_flow: slice
    row_ref: slice
    row_soc: slice
    row_init: slice
    row_term: slice
    row_reserve: slice


@dataclass
class MicrogridProblem:
    layout: MicrogridLayout
    A: torch.Tensor
    Pinv_AAt: torch.Tensor
    Hmat: torch.Tensor
    c_lin: torch.Tensor
    pg_max: torch.Tensor
    pdis_max: torch.Tensor


@dataclass
class DispatchData:
    b: torch.Tensor          # forecast RHS [M,m]
    b_real: torch.Tensor     # realized RHS [M,m]
    lb: torch.Tensor
    ub: torch.Tensor
    load_hat: torch.Tensor   # [M,B,H]
    load_real: torch.Tensor
    ren_hat: torch.Tensor    # [M,R,H]
    ren_real: torch.Tensor
    reserve_hat: torch.Tensor
    reserve_real: torch.Tensor
    e0: torch.Tensor         # [M,S]


def build_layout(cfg, device, dtype) -> MicrogridLayout:
    H = int(cfg["H"])
    n_bus = int(cfg["n_bus"])
    assert n_bus == 6, "The predefined network uses 6 buses."

    # A small meshed 6-bus microgrid.
    lines = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (2, 5)]
    susceptance = torch.tensor([5.0, 4.5, 4.2, 4.0, 4.3, 3.6, 3.8], device=device, dtype=dtype)
    base_flow_limit = torch.tensor([0.95, 0.85, 0.80, 0.75, 0.75, 0.70, 0.65], device=device, dtype=dtype)

    gen_bus = [0, 3]
    bat_bus = [2, 4]
    ren_bus = [1, 3, 5]
    n_gen, n_bat, n_ren, n_line = len(gen_bus), len(bat_bus), len(ren_bus), len(lines)

    idx = 0
    pg = slice(idx, idx + n_gen * H); idx += n_gen * H
    pch = slice(idx, idx + n_bat * H); idx += n_bat * H
    pdis = slice(idx, idx + n_bat * H); idx += n_bat * H
    e = slice(idx, idx + n_bat * (H + 1)); idx += n_bat * (H + 1)
    curt = slice(idx, idx + n_ren * H); idx += n_ren * H
    shed = slice(idx, idx + n_bus * H); idx += n_bus * H
    sres = slice(idx, idx + H); idx += H
    theta = slice(idx, idx + n_bus * H); idx += n_bus * H
    flow = slice(idx, idx + n_line * H); idx += n_line * H
    n = idx

    row = 0
    row_balance = slice(row, row + n_bus * H); row += n_bus * H
    row_flow = slice(row, row + n_line * H); row += n_line * H
    row_ref = slice(row, row + H); row += H
    row_soc = slice(row, row + n_bat * H); row += n_bat * H
    row_init = slice(row, row + n_bat); row += n_bat
    row_term = slice(row, row + n_bat); row += n_bat
    row_reserve = slice(row, row + H); row += H
    m = row

    return MicrogridLayout(
        H=H, n_bus=n_bus, n_line=n_line, n_gen=n_gen, n_bat=n_bat, n_ren=n_ren,
        lines=lines, susceptance=susceptance, base_flow_limit=base_flow_limit,
        gen_bus=gen_bus, bat_bus=bat_bus, ren_bus=ren_bus,
        pg=pg, pch=pch, pdis=pdis, e=e, curt=curt, shed=shed, sres=sres,
        theta=theta, flow=flow, n=n, m=m,
        row_balance=row_balance, row_flow=row_flow, row_ref=row_ref,
        row_soc=row_soc, row_init=row_init, row_term=row_term, row_reserve=row_reserve,
    )


def flat_idx(block: slice, unit: int, t: int, H: int) -> int:
    return block.start + unit * H + t


def e_idx(block: slice, bat: int, t: int, H: int) -> int:
    return block.start + bat * (H + 1) + t


def build_problem(cfg, device, dtype) -> MicrogridProblem:
    L = build_layout(cfg, device, dtype)
    H = L.H
    A = torch.zeros(L.m, L.n, device=device, dtype=dtype)

    # Bus power balance:
    # gen + pdis - pch - curt + shed - incidence*flow = load_hat - ren_hat.
    for t in range(H):
        for b in range(L.n_bus):
            row = L.row_balance.start + t * L.n_bus + b
            for g, gb in enumerate(L.gen_bus):
                if gb == b:
                    A[row, flat_idx(L.pg, g, t, H)] = 1.0
            for s, sb in enumerate(L.bat_bus):
                if sb == b:
                    A[row, flat_idx(L.pdis, s, t, H)] = 1.0
                    A[row, flat_idx(L.pch, s, t, H)] = -1.0
            for r, rb in enumerate(L.ren_bus):
                if rb == b:
                    A[row, flat_idx(L.curt, r, t, H)] = -1.0
            A[row, flat_idx(L.shed, b, t, H)] = 1.0
            for ell, (i, j) in enumerate(L.lines):
                fidx = flat_idx(L.flow, ell, t, H)
                if b == i:
                    A[row, fidx] -= 1.0
                elif b == j:
                    A[row, fidx] += 1.0

    # DC flow equations: f_l - B_l(theta_i - theta_j) = 0.
    for t in range(H):
        for ell, (i, j) in enumerate(L.lines):
            row = L.row_flow.start + t * L.n_line + ell
            A[row, flat_idx(L.flow, ell, t, H)] = 1.0
            A[row, flat_idx(L.theta, i, t, H)] = -float(L.susceptance[ell].item())
            A[row, flat_idx(L.theta, j, t, H)] = float(L.susceptance[ell].item())

    # Reference angle theta_0,t = 0.
    for t in range(H):
        row = L.row_ref.start + t
        A[row, flat_idx(L.theta, 0, t, H)] = 1.0

    # SOC dynamics.
    eta_ch = float(cfg["eta_ch"])
    eta_dis = float(cfg["eta_dis"])
    for t in range(H):
        for s in range(L.n_bat):
            row = L.row_soc.start + t * L.n_bat + s
            A[row, e_idx(L.e, s, t + 1, H)] = 1.0
            A[row, e_idx(L.e, s, t, H)] = -1.0
            A[row, flat_idx(L.pch, s, t, H)] = -eta_ch
            A[row, flat_idx(L.pdis, s, t, H)] = 1.0 / eta_dis

    # Initial and terminal SOC.
    for s in range(L.n_bat):
        A[L.row_init.start + s, e_idx(L.e, s, 0, H)] = 1.0
        A[L.row_term.start + s, e_idx(L.e, s, H, H)] = 1.0

    # Reserve adequacy:
    # sum(pgmax - pg) + sum(pdismax - pdis) + sres = reserve.
    # => -sum pg - sum pdis + sres = reserve - sum(pgmax) - sum(pdismax).
    for t in range(H):
        row = L.row_reserve.start + t
        for g in range(L.n_gen):
            A[row, flat_idx(L.pg, g, t, H)] = -1.0
        for s in range(L.n_bat):
            A[row, flat_idx(L.pdis, s, t, H)] = -1.0
        A[row, L.sres.start + t] = 1.0

    Pinv_AAt = torch.linalg.pinv(A @ A.t())

    Hmat = torch.zeros(L.n, L.n, device=device, dtype=dtype)
    c = torch.zeros(L.n, device=device, dtype=dtype)
    a_pg = cfg["a_pg"]
    b_pg = cfg["b_pg"]
    for t in range(H):
        for g in range(L.n_gen):
            idx = flat_idx(L.pg, g, t, H)
            Hmat[idx, idx] = 2.0 * float(a_pg[g])
            c[idx] = float(b_pg[g])
        for s in range(L.n_bat):
            c[flat_idx(L.pch, s, t, H)] = float(cfg["c_ch"])
            c[flat_idx(L.pdis, s, t, H)] = float(cfg["c_dis"])
        for r in range(L.n_ren):
            c[flat_idx(L.curt, r, t, H)] = float(cfg["c_curt"])
        for b in range(L.n_bus):
            c[flat_idx(L.shed, b, t, H)] = float(cfg["c_shed"])
        c[L.sres.start + t] = float(cfg["c_reserve_short"])
        for ell in range(L.n_line):
            idx = flat_idx(L.flow, ell, t, H)
            Hmat[idx, idx] = 2.0 * float(cfg["c_flow"])

    pg_max = torch.tensor(cfg["pg_max"], device=device, dtype=dtype)
    pdis_max = torch.tensor(cfg["pdis_max"], device=device, dtype=dtype)
    return MicrogridProblem(L, A, Pinv_AAt, Hmat, c, pg_max, pdis_max)


# ============================================================
# Dataset generation
# ============================================================

def make_rhs(problem: MicrogridProblem, load: torch.Tensor, ren: torch.Tensor, reserve: torch.Tensor, e0: torch.Tensor) -> torch.Tensor:
    L = problem.layout
    M, B, H = load.shape
    b = torch.zeros(M, L.m, device=load.device, dtype=load.dtype)

    # balance RHS load_bus - renewable_at_bus.
    for t in range(H):
        for bus in range(L.n_bus):
            rhs = load[:, bus, t].clone()
            for r, rb in enumerate(L.ren_bus):
                if rb == bus:
                    rhs = rhs - ren[:, r, t]
            b[:, L.row_balance.start + t * L.n_bus + bus] = rhs

    # Flow, reference, and SOC-dynamics rows are zero.
    for s in range(L.n_bat):
        b[:, L.row_init.start + s] = e0[:, s]
        b[:, L.row_term.start + s] = e0[:, s]

    reserve_const = float(problem.pg_max.sum().item() + problem.pdis_max.sum().item())
    for t in range(H):
        b[:, L.row_reserve.start + t] = reserve[:, t] - reserve_const
    return b


def generate_base_profiles(cfg, M: int, problem: MicrogridProblem, device, dtype):
    L = problem.layout
    H = L.H
    t = torch.arange(H, device=device, dtype=dtype).view(1, 1, H)
    level = cfg.get("scenario_level", "hard").lower()
    if level == "stress":
        load_amp = float(cfg["load_amp_stress"])
        ren_amp_base = cfg["ren_amp_stress"]
        forecast_noise = float(cfg["forecast_noise_stress"])
        reserve_factor = float(cfg["reserve_factor_stress"])
    elif level == "hard":
        load_amp = float(cfg["load_amp_hard"])
        ren_amp_base = cfg["ren_amp_hard"]
        forecast_noise = float(cfg["forecast_noise_hard"])
        reserve_factor = float(cfg["reserve_factor_hard"])
    else:
        load_amp = float(cfg["load_amp_medium"])
        ren_amp_base = cfg["ren_amp_medium"]
        forecast_noise = float(cfg["forecast_noise_medium"])
        reserve_factor = float(cfg["reserve_factor_medium"])

    base = torch.tensor(cfg["load_base"], device=device, dtype=dtype).view(1, L.n_bus, 1)
    scale = 0.85 + 0.35 * torch.rand(M, L.n_bus, 1, device=device, dtype=dtype)
    phase = 2 * math.pi * torch.rand(M, L.n_bus, 1, device=device, dtype=dtype)
    daily = 0.60 + 0.40 * torch.sin(2 * math.pi * (t - 7.0) / 24.0 + phase).clamp_min(-0.8)
    evening = 0.25 * torch.exp(-((t - 19.0) / 4.0) ** 2)
    load_true = base * scale + load_amp * (daily + evening)
    load_true += float(cfg["profile_noise"]) * torch.randn_like(load_true)
    load_true = load_true.clamp_min(0.10)

    ren_true = torch.zeros(M, L.n_ren, H, device=device, dtype=dtype)
    for r in range(L.n_ren):
        amp = float(ren_amp_base[r]) * (0.75 + 0.45 * torch.rand(M, 1, device=device, dtype=dtype))
        if r in (0, 2):
            shape = torch.sin(math.pi * (torch.arange(H, device=device, dtype=dtype).view(1, H) - 6.0) / 12.0).clamp_min(0.0)
        else:
            ph = 2 * math.pi * torch.rand(M, 1, device=device, dtype=dtype)
            shape = 0.55 + 0.30 * torch.sin(2 * math.pi * torch.arange(H, device=device, dtype=dtype).view(1, H) / 24.0 + ph)
        ren_true[:, r, :] = (amp * shape + float(cfg["profile_noise"]) * torch.randn(M, H, device=device, dtype=dtype)).clamp_min(0.0)

    # Forecasts are noisy but clipped to be nonnegative.
    load_hat = (load_true * (1.0 + forecast_noise * torch.randn_like(load_true))).clamp_min(0.05)
    ren_hat = (ren_true * (1.0 + 1.25 * forecast_noise * torch.randn_like(ren_true))).clamp_min(0.0)

    reserve_hat = reserve_factor * (load_hat.sum(dim=1) + 0.50 * ren_hat.sum(dim=1))
    reserve_real = reserve_factor * (load_true.sum(dim=1) + 0.50 * ren_true.sum(dim=1))
    reserve_hat = reserve_hat.clamp_min(0.02)
    reserve_real = reserve_real.clamp_min(0.02)

    e_min = torch.tensor(cfg["e_min"], device=device, dtype=dtype)
    e_max = torch.tensor(cfg["e_max"], device=device, dtype=dtype)
    e0 = e_min.view(1, -1) + (e_max - e_min).view(1, -1) * (0.25 + 0.50 * torch.rand(M, L.n_bat, device=device, dtype=dtype))

    return load_hat, load_true, ren_hat, ren_true, reserve_hat, reserve_real, e0


def make_dataset(cfg, problem: MicrogridProblem, M: int, device, dtype) -> DispatchData:
    L = problem.layout
    H = L.H
    load_hat, load_real, ren_hat, ren_real, reserve_hat, reserve_real, e0 = generate_base_profiles(cfg, M, problem, device, dtype)
    b = make_rhs(problem, load_hat, ren_hat, reserve_hat, e0)
    b_real = make_rhs(problem, load_real, ren_real, reserve_real, e0)

    lb = torch.full((M, L.n), -1e6, device=device, dtype=dtype)
    ub = torch.full((M, L.n), 1e6, device=device, dtype=dtype)

    # Nonnegative and upper-bounded operating variables.
    pg_max = torch.tensor(cfg["pg_max"], device=device, dtype=dtype)
    pg_min = torch.tensor(cfg["pg_min"], device=device, dtype=dtype)
    pch_max = torch.tensor(cfg["pch_max"], device=device, dtype=dtype)
    pdis_max = torch.tensor(cfg["pdis_max"], device=device, dtype=dtype)
    e_min = torch.tensor(cfg["e_min"], device=device, dtype=dtype)
    e_max = torch.tensor(cfg["e_max"], device=device, dtype=dtype)

    for t in range(H):
        for g in range(L.n_gen):
            idx = flat_idx(L.pg, g, t, H)
            lb[:, idx] = pg_min[g]
            ub[:, idx] = pg_max[g]
        for s in range(L.n_bat):
            lb[:, flat_idx(L.pch, s, t, H)] = 0.0
            ub[:, flat_idx(L.pch, s, t, H)] = pch_max[s]
            lb[:, flat_idx(L.pdis, s, t, H)] = 0.0
            ub[:, flat_idx(L.pdis, s, t, H)] = pdis_max[s]
        for r in range(L.n_ren):
            idx = flat_idx(L.curt, r, t, H)
            lb[:, idx] = 0.0
            ub[:, idx] = ren_hat[:, r, t]
        for bidx in range(L.n_bus):
            idx = flat_idx(L.shed, bidx, t, H)
            lb[:, idx] = 0.0
            ub[:, idx] = load_hat[:, bidx, t]
        lb[:, L.sres.start + t] = 0.0
        ub[:, L.sres.start + t] = reserve_hat[:, t] + 1.0
        for bus in range(L.n_bus):
            idx = flat_idx(L.theta, bus, t, H)
            lb[:, idx] = -float(cfg["theta_max"])
            ub[:, idx] = float(cfg["theta_max"])
        level = cfg.get("scenario_level", "stress").lower()
        if level == "stress":
            scale = float(cfg["flow_limit_scale_stress"])
        elif level == "hard":
            scale = float(cfg["flow_limit_scale_hard"])
        else:
            scale = float(cfg["flow_limit_scale_medium"])
        for ell in range(L.n_line):
            idx = flat_idx(L.flow, ell, t, H)
            lim = scale * L.base_flow_limit[ell]
            lb[:, idx] = -lim
            ub[:, idx] = lim

    for s in range(L.n_bat):
        for t in range(H + 1):
            idx = e_idx(L.e, s, t, H)
            lb[:, idx] = e_min[s]
            ub[:, idx] = e_max[s]

    return DispatchData(b=b, b_real=b_real, lb=lb, ub=ub,
                        load_hat=load_hat, load_real=load_real,
                        ren_hat=ren_hat, ren_real=ren_real,
                        reserve_hat=reserve_hat, reserve_real=reserve_real, e0=e0)


# ============================================================
# Projections, costs, and metrics
# ============================================================

def project_affine(problem: MicrogridProblem, v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    Av_minus_b = batch_matvec(problem.A, v) - b
    correction = (Av_minus_b @ problem.Pinv_AAt.t()) @ problem.A
    return v - correction


def project_box(v: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(v, lb), ub)


def dispatch_cost(problem: MicrogridProblem, z: torch.Tensor) -> torch.Tensor:
    q = 0.5 * (z * (z @ problem.Hmat.t())).sum(dim=1)
    l = (z * problem.c_lin.view(1, -1)).sum(dim=1)
    return q + l


def equality_violation(problem: MicrogridProblem, z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(batch_matvec(problem.A, z) - b, dim=1) / (1.0 + torch.linalg.norm(b, dim=1))


def box_violation(z: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    vio = torch.relu(lb - z) + torch.relu(z - ub)
    return torch.linalg.norm(vio, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))


def block_sum(z: torch.Tensor, block: slice) -> torch.Tensor:
    return z[:, block].sum(dim=1)


def realized_operational_metrics(problem: MicrogridProblem, z: torch.Tensor, data: DispatchData) -> Dict[str, torch.Tensor]:
    L = problem.layout
    H = L.H
    residual_real = batch_matvec(problem.A, z) - data.b_real
    bal = residual_real[:, L.row_balance].reshape(-1, H, L.n_bus).transpose(1, 2)  # [M,B,H]
    emergency_shed = torch.relu(-bal).sum(dim=(1, 2))
    emergency_spill = torch.relu(bal).sum(dim=(1, 2))

    planned_shed = z[:, L.shed].sum(dim=1)
    planned_curt = z[:, L.curt].sum(dim=1)
    total_load = data.load_real.sum(dim=(1, 2))
    total_ren = data.ren_real.sum(dim=(1, 2))

    shed_rate = (planned_shed + emergency_shed) / (1.0 + total_load)
    curt_rate = (planned_curt + emergency_spill) / (1.0 + total_ren)

    pg = z[:, L.pg].reshape(-1, L.n_gen, H)
    pdis = z[:, L.pdis].reshape(-1, L.n_bat, H)
    headroom = (problem.pg_max.view(1, -1, 1) - pg).clamp_min(0.0).sum(dim=1)
    headroom += (problem.pdis_max.view(1, -1, 1) - pdis).clamp_min(0.0).sum(dim=1)
    reserve_short = torch.relu(data.reserve_real - headroom).sum(dim=1) / (1.0 + data.reserve_real.sum(dim=1))

    return {
        "shed_rate": shed_rate,
        "curt_rate": curt_rate,
        "reserve_short_rate": reserve_short,
        "emergency_shed_rate": emergency_shed / (1.0 + total_load),
        "emergency_spill_rate": emergency_spill / (1.0 + total_ren),
    }


def validation_score(problem: MicrogridProblem, z: torch.Tensor, data: DispatchData, cfg) -> torch.Tensor:
    cost = dispatch_cost(problem, z)
    cost_term = cost / (1.0 + data.load_hat.sum(dim=(1, 2)))
    eq = equality_violation(problem, z, data.b)
    box = box_violation(z, data.lb, data.ub)
    op = realized_operational_metrics(problem, z, data)
    return (
        cost_term
        + float(cfg["lambda_eq_val"]) * eq
        + float(cfg["lambda_box_val"]) * box
        + float(cfg["lambda_shed_val"]) * op["shed_rate"]
        + float(cfg["lambda_reserve_val"]) * op["reserve_short_rate"]
        + float(cfg["lambda_curt_val"]) * op["curt_rate"]
    )


# ============================================================
# Solvers
# ============================================================

class DispatchSolver(nn.Module):
    name = "base"
    def forward(self, data: DispatchData, K: int) -> torch.Tensor:
        raise NotImplementedError


class FixedADMMSolver(DispatchSolver):
    def __init__(self, problem: MicrogridProblem, alpha: float, beta: float, name="Fixed-ADMM"):
        super().__init__()
        self.problem = problem
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.name = name

    def step(self, z, dual, data: DispatchData, alpha, beta):
        p = self.problem
        grad = z @ p.Hmat.t() + p.c_lin.view(1, -1)
        grad = grad / (torch.linalg.norm(grad, dim=1, keepdim=True) + 1e-8)
        w = z - dual - beta * grad
        x = project_affine(p, w, data.b)
        xbar = alpha * x + (1.0 - alpha) * z
        z_new = project_box(xbar + dual, data.lb, data.ub)
        dual_new = dual + xbar - z_new
        return x, z_new, dual_new

    def forward(self, data: DispatchData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
        dual = torch.zeros_like(z)
        for _ in range(K):
            _, z, dual = self.step(z, dual, data, self.alpha, self.beta)
        return z


class SpectralAADMM(FixedADMMSolver):
    def __init__(self, problem, alpha=1.0, beta=1.0, cfg=None):
        super().__init__(problem, alpha, beta, name="Spectral-AADMM")
        self.cfg = cfg or CONFIG

    def forward(self, data: DispatchData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
        dual = torch.zeros_like(z)
        beta = torch.full((B,), self.beta, device=data.b.device, dtype=data.b.dtype)
        beta_min, beta_max = float(self.cfg["spectral_beta_min"]), float(self.cfg["spectral_beta_max"])
        growth = float(self.cfg["spectral_growth"])
        for _ in range(K):
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
            beta = torch.clamp(torch.clamp(beta * ratio, beta / growth, beta * growth), beta_min, beta_max)
            z, dual = z_new, dual_new
        return z


class DREAndersonDRS(FixedADMMSolver):
    def __init__(self, problem, alpha, beta, omega=0.25, accept_tol=1.05):
        super().__init__(problem, alpha, beta, name="DRE-Anderson-DRS")
        self.omega = float(omega)
        self.accept_tol = float(accept_tol)

    def monitor(self, z, dual, data):
        return equality_violation(self.problem, z, data.b) + 0.05 * torch.linalg.norm(dual, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))

    def forward(self, data: DispatchData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
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
    def __init__(self, problem: MicrogridProblem, safety=0.95, theta=0.8, dual_clip=50.0):
        super().__init__()
        self.problem = problem
        self.theta = float(theta)
        self.dual_clip = float(dual_clip)
        with torch.no_grad():
            norm_A = torch.linalg.matrix_norm(problem.A, ord=2).item()
            norm_H = torch.linalg.matrix_norm(problem.Hmat, ord=2).item()
        norm_A_sq = max(norm_A ** 2, 1e-8)
        self.tau = 0.5 / (norm_H + norm_A_sq + 1e-8)
        self.sigma = min(0.5 / (norm_A_sq + 1e-8), float(safety) / (self.tau * norm_A_sq + 1e-8))

    def forward(self, data: DispatchData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        p = self.problem
        z = torch.zeros(B, p.layout.n, device=data.b.device, dtype=data.b.dtype)
        zbar = z.clone()
        y = torch.zeros(B, p.layout.m, device=data.b.device, dtype=data.b.dtype)
        for _ in range(K):
            y = torch.clamp(y + self.sigma * (batch_matvec(p.A, zbar) - data.b), -self.dual_clip, self.dual_clip)
            z_old = z
            grad = z @ p.Hmat.t() + p.c_lin.view(1, -1) + y @ p.A
            z = project_box(z - self.tau * grad, data.lb, data.ub)
            zbar = z + self.theta * (z - z_old)
        return z


class RCController(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg["rc_hidden"])
        kind = cfg.get("rc_controller", "gru").lower()
        if kind == "lstm":
            self.rnn = nn.LSTM(input_size=10, hidden_size=hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=10, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, seq):
        out, _ = self.rnn(seq)
        return self.head(out[:, -1, :])


class RCADMM(DispatchSolver):
    def __init__(self, cfg, problem: MicrogridProblem, name="RC-ADMM"):
        super().__init__()
        self.cfg = cfg
        self.problem = problem
        self.name = name
        self.ctrl = RCController(cfg)

    def map_params(self, raw, k, K, rho_prev, alpha_prev, beta_prev):
        cfg = self.cfg
        B = raw.shape[0]
        device, dtype = raw.device, raw.dtype
        rho_base = torch.full((B,), float(cfg["rho_base"]), device=device, dtype=dtype)
        alpha_base = torch.full((B,), float(cfg["alpha_base"]), device=device, dtype=dtype)
        beta_base = torch.full((B,), float(cfg["beta_base"]), device=device, dtype=dtype)
        if cfg.get("envelope", False):
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
        if cfg.get("growth", False):
            rho = torch.minimum(torch.maximum(rho, rho_prev / cfg["chi_rho"]), rho_prev * cfg["chi_rho"])
            beta = torch.minimum(torch.maximum(beta, beta_prev / cfg["chi_beta"]), beta_prev * cfg["chi_beta"])
        return rho, alpha, beta

    def make_feature(self, z, z_prev, dual, data, rho_prev, alpha_prev, beta_prev, prev_obj, k, K):
        cfg = self.cfg
        eq = equality_violation(self.problem, z, data.b)
        box = box_violation(z, data.lb, data.ub)
        dz = torch.linalg.norm(z - z_prev, dim=1) / (1.0 + torch.linalg.norm(z, dim=1))
        obj = dispatch_cost(self.problem, z)
        obj_scale = torch.abs(obj) / (1.0 + torch.linalg.norm(z, dim=1))
        delta_obj = (prev_obj - obj) / (1.0 + torch.abs(prev_obj))
        delta_obj = torch.clamp(delta_obj, -10.0, 10.0)
        arho = torch.log((rho_prev / float(cfg["rho_base"])).clamp_min(1e-12))
        aalpha = (alpha_prev - float(cfg["alpha_base"])) / max(float(cfg["alpha_max"]) - float(cfg["alpha_min"]), 1e-8)
        abeta = torch.log((beta_prev / float(cfg["beta_base"])).clamp_min(1e-12))
        time1 = torch.full_like(eq, float(k) / max(float(K), 1.0))
        time2 = torch.full_like(eq, float(K - k) / max(float(K), 1.0))
        return torch.stack([
            torch.log1p(box), torch.log1p(eq), torch.log1p(dz), torch.log1p(obj_scale),
            delta_obj, arho, aalpha, abeta, time1, time2
        ], dim=1)

    def forward(self, data: DispatchData, K: int) -> torch.Tensor:
        B = data.b.shape[0]
        z = torch.zeros(B, self.problem.layout.n, device=data.b.device, dtype=data.b.dtype)
        dual = torch.zeros_like(z)
        z_prev = z.clone()
        rho_prev = torch.full((B,), float(self.cfg["rho_base"]), device=data.b.device, dtype=data.b.dtype)
        alpha_prev = torch.full((B,), float(self.cfg["alpha_base"]), device=data.b.device, dtype=data.b.dtype)
        beta_prev = torch.full((B,), float(self.cfg["beta_base"]), device=data.b.device, dtype=data.b.dtype)
        prev_obj = dispatch_cost(self.problem, z)
        feats = []
        for k in range(K):
            feat = self.make_feature(z, z_prev, dual, data, rho_prev, alpha_prev, beta_prev, prev_obj, k, K).unsqueeze(1)
            feats.append(feat)
            raw = self.ctrl(torch.cat(feats, dim=1))
            rho, alpha, beta = self.map_params(raw, k, K, rho_prev, alpha_prev, beta_prev)
            grad = z @ self.problem.Hmat.t() + self.problem.c_lin.view(1, -1)
            grad = grad / (torch.linalg.norm(grad, dim=1, keepdim=True) + 1e-8)
            w = z - dual - beta.view(-1, 1) * grad
            x = project_affine(self.problem, w, data.b)
            xbar = alpha.view(-1, 1) * x + (1.0 - alpha.view(-1, 1)) * z
            z_new = project_box(xbar + dual, data.lb, data.ub)
            dual_new = dual + xbar - z_new
            z_prev = z
            prev_obj = dispatch_cost(self.problem, z)
            z, dual = z_new, dual_new
            rho_prev, alpha_prev, beta_prev = rho, alpha, beta
        return z


# ============================================================
# Training and evaluation
# ============================================================

def make_batches(data: DispatchData, batch_size: int, shuffle: bool):
    M = data.b.shape[0]
    ids = torch.randperm(M, device=data.b.device) if shuffle else torch.arange(M, device=data.b.device)
    for i in range(0, M, batch_size):
        idx = ids[i:i + batch_size]
        yield DispatchData(
            b=data.b[idx], b_real=data.b_real[idx], lb=data.lb[idx], ub=data.ub[idx],
            load_hat=data.load_hat[idx], load_real=data.load_real[idx],
            ren_hat=data.ren_hat[idx], ren_real=data.ren_real[idx],
            reserve_hat=data.reserve_hat[idx], reserve_real=data.reserve_real[idx], e0=data.e0[idx]
        )


def train_loss(problem, z, data, cfg):
    score = validation_score(problem, z, data, cfg)
    return score.mean()


@torch.no_grad()
def eval_for_selection(problem, solver, data, K, cfg):
    solver.eval()
    z = solver(data, K)
    score = validation_score(problem, z, data, cfg)
    return float(score.mean())


def tune_oracle(problem, cfg, val_data, K):
    print(f"[Tune] OracleGrid K={K}")
    best = (float(cfg["alpha_base"]), float(cfg["beta_base"]))
    best_score = float("inf")
    for alpha in cfg["oracle_grid"]["alphas"]:
        for beta in cfg["oracle_grid"]["betas"]:
            solver = FixedADMMSolver(problem, alpha, beta, name="OracleGrid-ADMM")
            s = eval_for_selection(problem, solver, val_data, K, cfg)
            print(f"  alpha={alpha:.2f}, beta={beta:.2f}: val_score={s:.4e}")
            if s < best_score:
                best_score = s
                best = (float(alpha), float(beta))
    print(f"  selected alpha={best[0]:.3g}, beta={best[1]:.3g}")
    return best


def make_rc_cfg(base_cfg, alpha_base, beta_base, envelope, growth, name_suffix):
    cfg = dict(base_cfg)
    cfg["alpha_base"] = float(alpha_base)
    cfg["beta_base"] = float(beta_base)
    cfg["rho_base"] = float(base_cfg.get("rho_base", 1.0))
    cfg["envelope"] = bool(envelope)
    cfg["growth"] = bool(growth)
    cfg["name_suffix"] = name_suffix
    return cfg


def train_rc(problem, cfg, train_data, val_data, K, name):
    print(f"[Train] {name} K={K}")
    solver = RCADMM(cfg, problem, name=name).to(train_data.b.device)
    opt = torch.optim.AdamW(solver.parameters(), lr=cfg["lr_rc"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["epochs_rc"]), eta_min=float(cfg["lr_rc"]) * 0.05)
    best_state = None
    best_score = float("inf")
    for ep in range(int(cfg["epochs_rc"])):
        solver.train()
        for batch in make_batches(train_data, int(cfg["batch_size"]), shuffle=True):
            opt.zero_grad(set_to_none=True)
            z = solver(batch, K)
            loss = train_loss(problem, z, batch, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(solver.parameters(), float(cfg["grad_clip"]))
            opt.step()
        scheduler.step()
        s = eval_for_selection(problem, solver, val_data, K, cfg)
        if s < best_score:
            best_score = s
            best_state = {key: val.detach().cpu().clone() for key, val in solver.state_dict().items()}
        print(f"  epoch {ep + 1:03d}: val_score={s:.4e}, best={best_score:.4e}")
    if best_state is not None:
        solver.load_state_dict(best_state)
    return solver


@torch.no_grad()
def compute_reference(problem, cfg, val_data, test_data):
    alpha_ref, beta_ref = tune_oracle(problem, cfg, val_data, int(cfg["K_ref"]))
    ref_solver = FixedADMMSolver(problem, alpha_ref, beta_ref, name="Long-Oracle-ADMM")
    zref = ref_solver(test_data, int(cfg["K_ref"]))
    cref = dispatch_cost(problem, zref)
    return cref, alpha_ref, beta_ref


@torch.no_grad()
def evaluate_method(problem, solver, data, K, ref_cost, method, cfg):
    solver.eval()
    device = data.b.device
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
    box = box_violation(z, data.lb, data.ub)
    op = realized_operational_metrics(problem, z, data)
    pass_mask = (
        (eq <= float(cfg["pass_eq_tol"]))
        & (box <= float(cfg["pass_box_tol"]))
        & (op["shed_rate"] <= float(cfg["pass_shed_rate_tol"]))
        & (op["reserve_short_rate"] <= float(cfg["pass_reserve_rate_tol"]))
    )

    return {
        "method": method,
        "K": int(K),
        "train_K": int(K),
        "cost_gap_mean": float(gap.mean()),
        "cost_gap_median": float(gap.median()),
        "cost_mean": float(cost.mean()),
        "eq_vio_mean": float(eq.mean()),
        "eq_vio_median": float(eq.median()),
        "box_vio_mean": float(box.mean()),
        "shed_rate_mean": float(op["shed_rate"].mean()),
        "curt_rate_mean": float(op["curt_rate"].mean()),
        "reserve_short_rate_mean": float(op["reserve_short_rate"].mean()),
        "emergency_shed_rate_mean": float(op["emergency_shed_rate"].mean()),
        "emergency_spill_rate_mean": float(op["emergency_spill_rate"].mean()),
        "pass_rate": float(pass_mask.float().mean()),
        "status_mean": "Pass" if bool(pass_mask.float().mean() >= 0.5) else "Fail",
        "runtime_ms": float((t1 - t0) * 1000.0 / data.b.shape[0]),
    }


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "cost_gap_mean", "cost_gap_median", "cost_mean", "eq_vio_mean", "eq_vio_median",
        "box_vio_mean", "shed_rate_mean", "curt_rate_mean", "reserve_short_rate_mean",
        "emergency_shed_rate_mean", "emergency_spill_rate_mean", "pass_rate", "runtime_ms",
        "oracle_alpha", "oracle_beta", "ref_alpha", "ref_beta",
    ]
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for c in metric_cols:
            if c in g.columns:
                row[c + "_mean"] = float(g[c].mean())
                row[c + "_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        row["status_majority"] = "Pass" if row.get("pass_rate_mean", 0.0) >= 0.5 else "Fail"
        rows.append(row)
    return pd.DataFrame(rows)


def run_one_seed(seed: int, cfg) -> pd.DataFrame:
    set_seed(seed)
    device = torch.device(cfg["device"])
    dtype = get_dtype(cfg)
    problem = build_problem(cfg, device, dtype)

    print("=" * 90)
    print(f"Seed={seed}, device={device}, dtype={dtype}, H={cfg['H']}, n={problem.layout.n}, m={problem.layout.m}")
    print("=" * 90)

    train_data = make_dataset(cfg, problem, int(cfg["n_train"]), device, dtype)
    val_data = make_dataset(cfg, problem, int(cfg["n_val"]), device, dtype)
    test_data = make_dataset(cfg, problem, int(cfg["n_test"]), device, dtype)

    ref_cost, ref_alpha, ref_beta = compute_reference(problem, cfg, val_data, test_data)

    rows = []
    for K in cfg["K_values"]:
        print("\n" + "#" * 90)
        print(f"# Forecast-aware dispatch seed={seed} K={K}")
        print("#" * 90)
        alpha_o, beta_o = tune_oracle(problem, cfg, val_data, int(K))

        solvers: Dict[str, DispatchSolver] = {
            "Fixed-ADMM": FixedADMMSolver(problem, cfg["alpha_base"], cfg["beta_base"], name="Fixed-ADMM").to(device),
            "OracleGrid-ADMM": FixedADMMSolver(problem, alpha_o, beta_o, name="OracleGrid-ADMM").to(device),
            "Spectral-AADMM": SpectralAADMM(problem, cfg["alpha_base"], cfg["beta_base"], cfg=cfg).to(device),
            "DRE-Anderson-DRS": DREAndersonDRS(problem, alpha_o, beta_o, cfg["anderson_omega"], cfg["anderson_accept_tol"]).to(device),
            "Stable-PDHG": StablePDHG(problem, cfg["pdhg_safety"], cfg["pdhg_theta"], cfg["pdhg_dual_clip"]).to(device),
        }

        cfg_env = make_rc_cfg(cfg, alpha_o, beta_o, envelope=True, growth=True, name_suffix="Env")
        cfg_noenv = make_rc_cfg(cfg, alpha_o, beta_o, envelope=False, growth=False, name_suffix="NoEnv")
        solvers["RC-ADMM-Env"] = train_rc(problem, cfg_env, train_data, val_data, int(K), "RC-ADMM-Env").to(device)
        solvers["RC-ADMM-NoEnv"] = train_rc(problem, cfg_noenv, train_data, val_data, int(K), "RC-ADMM-NoEnv").to(device)

        for method, solver in solvers.items():
            print(f"[Evaluate] seed={seed}, K={K}, method={method}")
            r = evaluate_method(problem, solver, test_data, int(K), ref_cost, method, cfg)
            r.update({
                "seed": int(seed),
                "scenario_level": cfg.get("scenario_level", "hard"),
                "oracle_alpha": float(alpha_o),
                "oracle_beta": float(beta_o),
                "ref_alpha": float(ref_alpha),
                "ref_beta": float(ref_beta),
            })
            rows.append(r)
            print(r)

    return pd.DataFrame(rows)


def run_experiment(cfg=CONFIG):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in cfg["seeds"]:
        df_seed = run_one_seed(int(seed), cfg)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        df_seed.to_csv(seed_dir / "forecast_dispatch_results_by_seed.csv", index=False)
        all_rows.append(df_seed)

    df = pd.concat(all_rows, ignore_index=True)
    front = ["scenario_level", "seed", "K", "train_K", "method"]
    df = df[front + [c for c in df.columns if c not in front]]
    all_path = out_dir / "forecast_dispatch_all_results.csv"
    df.to_csv(all_path, index=False)

    summary = summarize(df, ["scenario_level", "K", "method"])
    summary_path = out_dir / "forecast_dispatch_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:", all_path)
    print("Saved:", summary_path)
    print(summary)
    return df, summary


if __name__ == "__main__":
    run_experiment(CONFIG)
