"""
JAX port of the gym (gym.py), the recurrent agent (agents.py), the soft clip (softclip.py)
and the OCE monetary utilities (objectives.py). Min and max follow TensorFlow's gradient
convention at ties so that gradients match the original exactly.
"""
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize_scalar

STATE = "recurrent_state"
UTILITIES = ("mean", "exp", "exp2", "vicky", "cvar", "quad")
ACTIVATIONS = dict(relu=jax.nn.relu, softplus=jax.nn.softplus, tanh=jnp.tanh, sigmoid=jax.nn.sigmoid, linear=lambda x: x)


@dataclass(frozen=True)
class AgentConfig:
    features: tuple = ("price", "delta", "time_left")
    width: int = 20
    depth: int = 3
    activation: str = "relu"
    classic: int = 0
    aggregate: int = 0
    past_repr: int = 0
    event: int = 0
    bound_aggr_states: bool = False
    init_delta: bool = True

    @property
    def n_states(self) -> int:
        return self.classic + self.aggregate + self.past_repr + self.event


@dataclass(frozen=True)
class GymConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    utility: str = "exp2"
    lmbda: float = 1.


def input_features(cfg: AgentConfig) -> list:
    return sorted(set(cfg.features) | ({STATE} if cfg.n_states else set()))


def _glorot(key, shape):
    fan_in, fan_out = (shape[0], shape[0]) if len(shape) == 1 else shape
    lim = jnp.sqrt(6. / (fan_in + fan_out))
    return jax.random.uniform(key, shape, jnp.float32, -lim, lim)


def init_params(key, cfg: GymConfig, feature_dims: dict, n_inst: int) -> dict:
    a = cfg.agent
    dims = dict(feature_dims, action=n_inst, delta=n_inst, **({STATE: a.n_states} if a.n_states else {}))
    sizes = [sum(dims[f] for f in input_features(a))] + [a.width] * a.depth + [n_inst + a.n_states + a.aggregate + a.past_repr + a.event]
    keys = jax.random.split(key, len(sizes) + 1)
    params = dict(dense=[dict(W=_glorot(k, (i, o)), b=jnp.zeros((o,), jnp.float32)) for k, i, o in zip(keys, sizes[:-1], sizes[1:])],
                  y=jnp.zeros((1,), jnp.float32), y0=jnp.zeros((1,), jnp.float32))
    if a.n_states:
        params["state0"] = _glorot(keys[-2], (a.n_states,))
    if a.init_delta:
        params["idelta"] = _glorot(keys[-1], (n_inst,))
    return params


def _unit(x):
    return jnp.where(x > 0.5, 1., 0.)


def _split(x, sizes):
    out, start = [], 0
    for s in sizes:
        out.append(x[:, start:start + s]); start += s
    return out


def agent_step(p: dict, cfg: AgentConfig, act, feats: dict, state):
    feats = dict(feats)
    if cfg.n_states:
        classic, aggr, past, event = _split(state, (cfg.classic, cfg.aggregate, cfg.past_repr, cfg.event))
        classic, event = jnp.tanh(classic), _unit(event)
        aggr = jnp.tanh(aggr) if cfg.bound_aggr_states else aggr
        feats[STATE] = jnp.concatenate([classic, aggr, past, event], axis=1)
    z = jnp.concatenate([feats[f] for f in input_features(cfg)], axis=1)
    for layer in p["dense"][:-1]:
        z = act(z @ layer["W"] + layer["b"])
    out = z @ p["dense"][-1]["W"] + p["dense"][-1]["b"]
    n_inst = out.shape[1] - cfg.n_states - cfg.aggregate - cfg.past_repr - cfg.event
    if not cfg.n_states:
        return out, state
    rec = out[:, n_inst:]
    c_new, a_cand, p_cand, e_cand, a_upd, p_upd, e_upd = _split(
        rec, (cfg.classic, cfg.aggregate, cfg.past_repr, cfg.event, cfg.aggregate, cfg.past_repr, cfg.event))
    a_cand = jnp.tanh(a_cand) if cfg.bound_aggr_states else a_cand
    a_upd, p_upd, e_upd = jax.nn.sigmoid(a_upd), _unit(p_upd), _unit(e_upd)
    new_state = jnp.concatenate([
        jnp.tanh(c_new),
        (1. - a_upd) * aggr + a_upd * a_cand,
        (1. - p_upd) * past + p_upd * p_cand,
        (1. - e_upd) * event + e_upd * _unit(e_cand),
    ], axis=1)
    return out[:, :n_inst], new_state


def softclip(a, lb, ub):
    sp = jax.nn.softplus
    a = jnp.minimum(a, ub * 10.)
    a = jnp.maximum(a, lb * 10.)
    d = ub - lb
    r = (a - lb) / d
    r = 1. - sp(1. - sp(r)) / sp(1.)
    return jnp.where(d > 0., r * d + lb, 0.)


def _neg(g):
    return jnp.where(g < 0., g, 0.)


def utility(name: str, lmbda: float, X, y):
    if lmbda < 1E-12:
        name, lmbda = "mean", 1.
    gains = X + y
    if name == "mean":
        return gains
    if name == "cvar":
        return (1. + lmbda) * _neg(gains) - y
    if name == "quad":
        x0 = 1. / lmbda
        xx = _neg(gains - x0)
        return -0.5 * lmbda * xx ** 2 + 0.5 * lmbda * x0 ** 2 - y
    if name == "exp":
        inf = jax.lax.stop_gradient(jnp.min(X))
        return (1. - jnp.exp(-lmbda * (gains - inf))) / lmbda - y + inf
    if name == "exp2":
        g1 = jnp.where(gains >= 0., gains, 0.)
        g2 = jnp.where(gains <= 0., gains, 0.)
        u1 = (1. - jnp.exp(-lmbda * g1)) / lmbda - y
        u2 = g2 - 0.5 * lmbda * g2 * g2 - y
        return jnp.where(gains > 0., u1, u2)
    if name == "vicky":
        return (1. + lmbda * gains - jnp.sqrt(1. + (lmbda * gains) ** 2)) / lmbda - y
    raise ValueError(f"Unknown utility '{name}'. Use one of {UTILITIES}")


def oce_utility(name: str, lmbda: float, X, sample_weights=None) -> float:
    """ sup_y E[u(X+y)-y] for a sample X, in closed form where objectives.oce_utility has one. """
    X = np.asarray(X, dtype=np.float64)
    P = np.full(X.shape, 1. / len(X)) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape(X.shape)
    P = P / np.sum(P)
    if name == "mean" or lmbda == 0.:
        return float(np.sum(P * X))
    if name == "exp":
        return float(-np.log(np.sum(P * np.exp(-lmbda * X))) / lmbda)
    if name == "cvar":
        p = 1. / (1. + lmbda)
        if sample_weights is None:
            return float(np.mean(X[X <= np.percentile(X, p * 100.)]))
        ixs = np.argsort(X)
        X, P = X[ixs], P[ixs]
        cumP = np.cumsum(P)
        cumP[-1] = 1.
        if p <= cumP[0]:
            return float(X[0])
        ix = min(len(X) - 1, int(np.searchsorted(cumP, p)))
        return float(np.sum((P * X)[:ix + 1]) / np.sum(P[:ix + 1]))
    Xj, Pj = jnp.asarray(X, jnp.float32), jnp.asarray(P, jnp.float32)
    objective = jax.jit(lambda y: -jnp.sum(Pj * utility(name, lmbda, Xj, y)))
    r = minimize_scalar(lambda y: float(objective(jnp.float32(y))), tol=1E-6)
    if not r.success:
        raise RuntimeError(f"Failed to find the OCE intercept for {name}@{lmbda}: {r.message}")
    return -float(r.fun)


def rollout(p: dict, cfg: GymConfig, batch: dict) -> dict:
    """ Same quantities per path as VanillaDeepHedgingGym.call(). """
    a_cfg, act = cfg.agent, ACTIVATIONS[cfg.agent.activation]
    m = batch["market"]
    payoff, hedges = m["payoff"], m["hedges"]
    n, _, n_inst = hedges.shape
    zeros_i = jnp.zeros((n, n_inst), jnp.float32)
    state0 = jnp.broadcast_to(p["state0"], (n, a_cfg.n_states)) if a_cfg.n_states else jnp.zeros((n, 0), jnp.float32)
    idelta = p["idelta"][None, :] if a_cfg.init_delta else zeros_i
    tm = lambda x: jnp.swapaxes(x, 0, 1)
    xs = (tm(hedges), tm(m["cost"]), tm(m["lbnd_a"]), tm(m["ubnd_a"]), {k: tm(v) for k, v in batch["features"].items()})

    def body(carry, x):
        delta, action, cost, pnl, state, first = carry
        h, c, lb, ub, f = x
        feats = dict(action=action, delta=delta, **f)
        a, state = agent_step(p, a_cfg, act, feats, state)
        a = softclip(a + idelta * first, lb, ub)
        delta = delta + a
        cost = cost + jnp.sum(jnp.abs(a) * c, axis=1)
        pnl = pnl + jnp.sum(a * h, axis=1)
        return (delta, a, cost, pnl, state, 0.), jax.lax.stop_gradient(a)

    z = jnp.zeros_like(payoff)
    (_, _, cost, pnl, _, _), actions = jax.lax.scan(body, (zeros_i, zeros_i, z, z, state0, 1.), xs)
    y = p["y"][0] if cfg.utility != "mean" else 0.
    y0 = p["y0"][0] if cfg.utility != "mean" else 0.
    u = utility(cfg.utility, cfg.lmbda, payoff + pnl - cost, y)
    u0 = utility(cfg.utility, cfg.lmbda, payoff, y0)
    sg = jax.lax.stop_gradient
    return dict(loss=-u - u0, utility=sg(u), utility0=sg(u0), gains=sg(payoff + pnl - cost), payoff=payoff,
                pnl=sg(pnl), cost=sg(cost), actions=jnp.swapaxes(actions, 0, 1))


def loss(p: dict, cfg: GymConfig, batch: dict):
    return jnp.mean(rollout(p, cfg, batch)["loss"])


def as_batch(world) -> dict:
    return dict(market={k: jnp.asarray(v) for k, v in world.market.items()},
                features={k: jnp.asarray(v) for k, v in world.features.items()})
