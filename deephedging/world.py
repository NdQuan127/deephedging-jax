"""
NumPy port of world.SimpleWorld_Spot_ATM without TensorFlow.
The simulation reproduces the original draws and arithmetic so that the same seed gives the same arrays,
including the original behaviour that the payoff receives spots[:, :nSteps+2].
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import math
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class WorldConfig:
    steps: int = 10
    samples: int = 1000
    seed: int = 2312414312
    invar_steps: int = 5
    dt: float = 1. / 50.
    cost_s: float = 0.0002
    ubnd_as: float = 5.
    lbnd_as: float = -5.
    black_scholes: bool = False
    no_stoch_vol: bool = False
    no_stoch_drift: bool = False
    strike: float = 1.
    ttm_steps: int = 4
    cost_v: float = 0.02
    cost_p: float = 0.0005
    ubnd_av: float = 5.
    lbnd_av: float = -5.
    payoff: object = "atmcall"
    drift: float = 0.1
    meanrev_drift: float = 1.
    drift_vol: float = 0.1
    rvol: float = 0.2
    ivol: float = None
    meanrev_rvol: float = 2.
    meanrev_ivol: float = 0.1
    volvol_rvol: float = 0.5
    volvol_ivol: float = 0.5
    corr_ms: float = 0.5
    corr_vs: float = -0.7
    corr_vi: float = 0.8
    rcorr_vs: float = -0.5


@dataclass
class World:
    config: WorldConfig
    market: dict
    features: dict
    spot_all: np.ndarray
    details: dict
    nSamples: int
    nSteps: int
    nInst: int
    dt: float

    @property
    def inst_names(self) -> list:
        return ["spot", "ATM Call"][:self.nInst]

    @property
    def timeline(self) -> np.ndarray:
        return np.arange(self.nSteps + 1) * self.dt

    def clone(self, **overrides) -> "World":
        return simulate(replace(self.config, **overrides))


def simulate(c: WorldConfig) -> World:
    dtype = np.float32
    nSteps, nSamples, nIvSteps, dt = c.steps, c.samples, c.invar_steps, c.dt
    strike, ttm_steps = c.strike, c.ttm_steps
    kappa_m, xi_m = c.meanrev_drift, c.drift_vol
    kappa_v, kappa_i, xi_v, xi_i = c.meanrev_rvol, c.meanrev_ivol, c.volvol_rvol, c.volvol_ivol
    rvol_init = c.rvol
    ivol_init = c.ivol if c.ivol is not None else rvol_init
    no_sdrift, no_svol = c.no_stoch_drift, c.no_stoch_vol
    rho_ms, rho_vs, rho_vi, rho_vs_r = c.corr_ms, c.corr_vs, c.corr_vi, c.rcorr_vs

    payoff_f = c.payoff
    if payoff_f is None:
        payoff_f = np.zeros((nSamples,))
    elif isinstance(payoff_f, (int, float)):
        payoff_f = np.full((nSamples,), float(payoff_f))
    elif payoff_f == "atmcall":
        payoff_f = lambda spots: -np.maximum(spots[:, -1] - 1., 0.)
    elif payoff_f == "atmput":
        payoff_f = lambda spots: -np.maximum(1. - spots[:, -1], 0.)

    if c.black_scholes:
        strike, ttm_steps, nIvSteps, no_sdrift, no_svol = 0., 1, 0, True, True
    if no_sdrift:
        kappa_m, xi_m = 0., 0.
    if no_svol:
        kappa_v, kappa_i, xi_v, xi_i = 0., 0., 0., 0.

    sqrtDt = math.sqrt(dt)
    ttm_steps = ttm_steps if strike > 0. else 1
    ttm = ttm_steps * dt
    sqrtTTM = math.sqrt(ttm)
    xi_m, xi_v, xi_i = abs(xi_m), abs(xi_v), abs(xi_i)
    time_left = np.linspace(float(nSteps), 1., nSteps, endpoint=True, dtype=dtype) * dt
    sqrt_time_left = np.sqrt(time_left)

    np.random.seed(c.seed)
    nAll = nSteps + nIvSteps + ttm_steps
    dW = np.random.normal(size=(nSamples, nAll - 1, 4)).astype(dtype) * sqrtDt
    dW_s = dW[:, :, 0]
    dW_m = dW[:, :, 0] * rho_ms + math.sqrt(1. - rho_ms ** 2) * dW[:, :, 1]
    dW_v = dW[:, :, 0] * rho_vs + math.sqrt(1. - rho_vs ** 2) * dW[:, :, 2]
    dW_i = dW[:, :, 2] * rho_vi + math.sqrt(1. - rho_vi ** 2) * (dW[:, :, 0] * rho_vs_r + math.sqrt(1. - rho_vs_r ** 2) * dW[:, :, 3])

    spot = np.zeros((nSamples, nAll), dtype=dtype)
    rdrift = np.full((nSamples, nAll), c.drift, dtype=dtype)
    rvol = np.full((nSamples, nAll), rvol_init, dtype=dtype)
    ivol = np.full((nSamples, nAll), ivol_init, dtype=dtype)
    spot[:, 0] = 1.
    log_ivol_init = np.log(ivol_init)
    log_rvol = np.log(rvol_init)
    log_ivol = log_ivol_init
    mrdrift = 0. * dW_m[:, 0]
    expdriftdt = np.exp(c.drift * dt)
    bStochDrift = kappa_m != 0. or xi_m != 0.
    bStochVol = kappa_v != 0. or xi_v != 0. or kappa_i != 0. or xi_i != 0.

    for j in range(1, nAll):
        spot[:, j] = spot[:, j - 1] * np.exp(rdrift[:, j - 1] * dt + rvol[:, j - 1] * dW_s[:, j - 1] - 0.5 * (rvol[:, j - 1] ** 2) * dt)
        spot[:, j] *= expdriftdt / np.mean(spot[:, j])
        if bStochDrift:
            mrdrift = mrdrift - kappa_m * mrdrift * dt + xi_m * dW_m[:, j - 1]
            mrdrift = np.exp(mrdrift * dt)
            mrdrift = np.log(mrdrift / np.mean(mrdrift)) / dt
            rdrift[:, j] = c.drift + mrdrift
        if bStochVol:
            log_rvol += kappa_v * (log_ivol - log_rvol) * dt + xi_v * dW_v[:, j - 1] - 0.5 * (xi_v ** 2) * dt
            log_ivol += kappa_i * (log_ivol_init - log_ivol) * dt + xi_i * dW_i[:, j - 1] - 0.5 * (xi_i ** 2) * dt
            rvol[:, j] = np.exp(log_rvol)
            ivol[:, j] = np.exp(log_ivol)

    spot = spot[:, nIvSteps:]
    rdrift = rdrift[:, nIvSteps:nIvSteps + nSteps]
    rvol = rvol[:, nIvSteps:nIvSteps + nSteps]
    ivol = ivol[:, nIvSteps:nIvSteps + nSteps]
    ixs = np.argsort(spot[:, nSteps])
    spot, rdrift, rvol, ivol = spot[ixs, :], rdrift[ixs, :], rvol[ixs, :], ivol[ixs, :]

    dS = spot[:, nSteps][:, np.newaxis] - spot[:, :nSteps]
    cost_dS = spot[:, :nSteps] * c.cost_s
    if strike <= 0.:
        hedges, cost, price = dS[:, :, np.newaxis], cost_dS[:, :, np.newaxis], spot[:, :nSteps]
        ubnd_a = np.full((nSamples, nSteps, 1), c.ubnd_as)
        lbnd_a = np.full((nSamples, nSteps, 1), c.lbnd_as)
        option = None
    else:
        mat_spot = spot[:, ttm_steps:ttm_steps + nSteps]
        opt_spot = spot[:, :nSteps]
        payoffs = np.maximum(0, mat_spot - strike * opt_spot)
        d1 = (-np.log(strike) + 0.5 * ivol * ivol * ttm) / (ivol * sqrtTTM)
        d2 = d1 - ivol * sqrtTTM
        call_price = norm.cdf(d1) * opt_spot - norm.cdf(d2) * strike * opt_spot
        call_delta = norm.cdf(d1)
        call_vega = opt_spot * norm.pdf(d1) * sqrtTTM
        cost_dC = c.cost_v * np.abs(call_vega) + c.cost_s * np.abs(call_delta) + c.cost_p * abs(call_price)
        hedges = np.stack([dS, payoffs - call_price], axis=2).astype(dtype)
        cost = np.stack([cost_dS, cost_dC], axis=2).astype(dtype)
        price = np.stack([spot[:, :nSteps], call_price], axis=2).astype(dtype)
        ubnd_a = np.stack([np.full((nSamples, nSteps), c.ubnd_as), np.full((nSamples, nSteps), c.ubnd_av)], axis=2).astype(dtype)
        lbnd_a = np.stack([np.full((nSamples, nSteps), c.lbnd_as), np.full((nSamples, nSteps), c.lbnd_av)], axis=2).astype(dtype)
        option = dict(call_price=call_price, call_delta=call_delta, call_vega=call_vega, cost_v=cost_dC)

    py_feat = None
    if isinstance(payoff_f, np.ndarray):
        payoff = payoff_f
    else:
        payoff = payoff_f(spot[:, :nSteps + 2])
        if isinstance(payoff, Mapping):
            py_feat = np.asarray(payoff["features"])
            payoff = np.asarray(payoff["payoff"])
        payoff = np.asarray(payoff)
        payoff = payoff[:, 0] if payoff.shape == (nSamples, 1) else payoff
    if payoff.shape != (nSamples,):
        raise ValueError(f"payoff must have shape {(nSamples,)}, found {payoff.shape}")

    per_step = dict(cost=cost, price=price, ubnd_a=ubnd_a, lbnd_a=lbnd_a,
                    time_left=np.full((nSamples, nSteps), time_left[np.newaxis, :], dtype=dtype),
                    sqrt_time_left=np.full((nSamples, nSteps), sqrt_time_left[np.newaxis, :], dtype=dtype),
                    spot=spot[:, :nSteps], ivol=ivol)
    if option is not None:
        per_step.update(option)
    if py_feat is not None:
        per_step["payoff_features"] = py_feat
    per_step = {k: _as_3d(np.asarray(v, dtype=dtype)) for k, v in per_step.items()}

    market = dict(hedges=hedges.astype(dtype), cost=cost.astype(dtype), ubnd_a=ubnd_a.astype(dtype),
                  lbnd_a=lbnd_a.astype(dtype), payoff=payoff.astype(dtype))
    return World(config=c, market=market, features=per_step, spot_all=spot[:, :nSteps + 1],
                 details=dict(drift=rdrift, rvol=rvol),
                 nSamples=nSamples, nSteps=nSteps, nInst=hedges.shape[2], dt=dt)


def _as_3d(x: np.ndarray) -> np.ndarray:
    return x[:, :, np.newaxis] if x.ndim == 2 else x
