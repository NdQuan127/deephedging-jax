"""
Training loop replacing trainer.train(): shuffled minibatches, the TF 2.11 Keras Adam update
(including its clipnorm > global_clipnorm > clipvalue precedence), best weights selected on the
full training-set loss after every epoch as trainer.Monitor does, and resumable checkpoints.
"""
from dataclasses import dataclass, replace
import hashlib
import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from .model import GymConfig, rollout, loss, init_params, as_batch


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    clipnorm: float = None
    global_clipnorm: float = None
    clipvalue: float = None
    seed: int = 0


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return dict(m=zeros, v=zeros, t=jnp.zeros((), jnp.float32))


def adam_update(c: TrainConfig, grads, opt, params):
    if c.clipnorm:
        grads = jax.tree_util.tree_map(lambda g: g * c.clipnorm / jnp.maximum(jnp.linalg.norm(g), c.clipnorm), grads)
    elif c.global_clipnorm:
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
        grads = jax.tree_util.tree_map(lambda g: g * c.global_clipnorm / jnp.maximum(norm, c.global_clipnorm), grads)
    elif c.clipvalue:
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -c.clipvalue, c.clipvalue), grads)
    t = opt["t"] + 1.
    alpha = c.learning_rate * jnp.sqrt(1. - c.beta_2 ** t) / (1. - c.beta_1 ** t)
    m = jax.tree_util.tree_map(lambda m, g: m + (g - m) * (1. - c.beta_1), opt["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: v + (g * g - v) * (1. - c.beta_2), opt["v"], grads)
    params = jax.tree_util.tree_map(lambda p, m, v: p - m * alpha / (jnp.sqrt(v) + c.epsilon), params, m, v)
    return params, dict(m=m, v=v, t=t)


def _take(batch, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], batch)


def make_step(gym: GymConfig, c: TrainConfig):
    def step(carry, idx, data):
        params, opt = carry
        g = jax.grad(loss)(params, gym, _take(data, idx))
        return adam_update(c, g, opt, params), None
    return step


@dataclass
class TrainResult:
    params: dict
    last_params: dict
    best_epoch: int
    history: dict


HISTORY_KEYS = ("loss", "val_loss", "utility", "utility0", "val_utility", "val_utility0", "time")


def checkpoint_path(directory: str, gym: GymConfig, c: TrainConfig, params: dict, data: dict, val_data: dict) -> str:
    """ One file per run identity: configs except the epoch count, initial weights and both data sets. """
    h = hashlib.sha256(repr((gym, replace(c, epochs=0))).encode())
    for leaf in jax.tree_util.tree_leaves((params, data, val_data)):
        h.update(np.ascontiguousarray(leaf).tobytes())
    return os.path.join(directory, h.hexdigest()[:24] + ".pkl")


def _save(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, state), f)
    os.replace(tmp, path)


def train(gym: GymConfig, c: TrainConfig, params: dict, data: dict, val_data: dict, *, log_every: int = 0,
          checkpoint_dir: str = None, checkpoint_every: int = 10, on_epoch=None) -> TrainResult:
    """
    Trains for c.epochs epochs. With checkpoint_dir, a rerun with the same configs, initial weights and data
    resumes from the last checkpoint and continues to c.epochs, and returns at once if it already got there.
    Each epoch shuffles with a generator seeded by (c.seed, epoch), so a resumed run equals an uninterrupted one.
    on_epoch(epoch, history, best_params) is called after every epoch, e.g. for live plots.
    """
    n = data["market"]["payoff"].shape[0]
    n_full = n // c.batch_size
    step = make_step(gym, c)

    @jax.jit
    def full_batches(carry, perm, data):
        carry, _ = jax.lax.scan(lambda cr, idx: step(cr, idx, data), carry, perm.reshape(n_full, c.batch_size))
        return carry

    tail_step = jax.jit(lambda carry, idx, data: step(carry, idx, data)[0])

    @jax.jit
    def evaluate(params, data):
        r = rollout(params, gym, data)
        return jnp.mean(r["loss"]), jnp.mean(r["utility"]), jnp.mean(r["utility0"])

    path = checkpoint_path(checkpoint_dir, gym, c, params, data, val_data) if checkpoint_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            st = jax.tree_util.tree_map(jnp.asarray, pickle.load(f), is_leaf=lambda x: isinstance(x, np.ndarray))
        st["history"] = {k: list(map(float, v)) for k, v in st["history"].items()}
        st["best_loss"], st["best_epoch"] = float(st["best_loss"]), int(st["best_epoch"])
    else:
        st = dict(carry=(params, adam_init(params)), best_params=params, best_loss=float(evaluate(params, data)[0]),
                  best_epoch=-1, history={k: [] for k in HISTORY_KEYS})
    hist = st["history"]
    for epoch in range(len(hist["loss"]), c.epochs):
        t0 = time.time()
        perm = np.random.default_rng([c.seed, epoch]).permutation(n)
        carry = st["carry"]
        if n_full:
            carry = full_batches(carry, jnp.asarray(perm[:n_full * c.batch_size]), data)
        if n % c.batch_size:
            carry = tail_step(carry, jnp.asarray(perm[n_full * c.batch_size:]), data)
        st["carry"] = carry
        tl, tu, tu0 = evaluate(carry[0], data)
        vl, vu, vu0 = evaluate(carry[0], val_data)
        for k, v in zip(HISTORY_KEYS[:-1], (tl, vl, tu, tu0, vu, vu0)):
            hist[k].append(float(v))
        hist["time"].append(time.time() - t0)
        if hist["loss"][-1] < st["best_loss"]:
            st["best_loss"], st["best_params"], st["best_epoch"] = hist["loss"][-1], carry[0], epoch
        if path and ((epoch + 1) % checkpoint_every == 0 or epoch + 1 == c.epochs):
            _save(path, st)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1}/{c.epochs}: loss {tl:.5f} val {vl:.5f} best@{st['best_epoch'] + 1} ({hist['time'][-1]:.2f}s)", flush=True)
        if on_epoch is not None:
            on_epoch(epoch, hist, st["best_params"])
    return TrainResult(params=st["best_params"], last_params=st["carry"][0], best_epoch=st["best_epoch"], history=hist)


@dataclass
class Fit:
    gym: GymConfig
    result: TrainResult
    train: dict
    val: dict


def fit(gym: GymConfig, c: TrainConfig, world, val_world, *, key: int = 0, **train_kwargs) -> Fit:
    """ Initializes, trains and rolls out the best weights on both worlds, as a notebook's train() cell does. """
    data, val_data = as_batch(world), as_batch(val_world)
    params = init_params(jax.random.PRNGKey(key), gym, {k: v.shape[2] for k, v in world.features.items()}, world.nInst)
    res = train(gym, c, params, data, val_data, **train_kwargs)
    as_np = lambda r: {k: np.asarray(v) for k, v in r.items()}
    return Fit(gym=gym, result=res, train=as_np(rollout(res.params, gym, data)), val=as_np(rollout(res.params, gym, val_data)))
