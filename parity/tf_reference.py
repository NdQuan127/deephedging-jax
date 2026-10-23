"""
TensorFlow side of the parity check. Runs the original code and writes one reference file per case.
Needs the original environment (TF 2.11, tensorflow_probability 0.19, cdxbasics).
The original package is also named 'deephedging'. DH_ORIGINAL names the directory that contains it
(default ../archive) and goes first on sys.path, so the imports below load the original, not the port.

    python -m parity.tf_reference <out_dir> world|gym|train|helpers <case>
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.environ.get("DH_ORIGINAL", os.path.join(os.path.dirname(__file__), "..", "..", "archive"))))
import numpy as np
import tensorflow as tf
from cdxbasics.config import Config
from deephedging.world import SimpleWorld_Spot_ATM
from deephedging.gym import VanillaDeepHedgingGym
from deephedging.base import create_optimizer
from deephedging.trainer import default_loss
from deephedging.objectives import oce_utility
from deephedging.fd import bs_fd
from deephedging.base import mean_bins, mean_cum_bins, perct_exp
from .cases import WORLDS, GYM_CASES, TRAIN_CASES, OCE_CASES, train_batches, helper_inputs, fd_payoff


def tf_world(name):
    c = Config()
    for k, v in WORLDS[name].items():
        c[k] = v
    return SimpleWorld_Spot_ATM(c)


def tf_gym(case):
    c = Config()
    spec = GYM_CASES[case]
    c.objective.utility, c.objective.lmbda = spec["gym"]["utility"], spec["gym"]["lmbda"]
    for k, v in spec["agent"].items():
        if k in ("depth", "width", "activation"):
            c.agent.network[k] = v
        elif k in ("classic", "aggregate", "past_repr", "event"):
            c.agent.recurrence.states[k] = v
        elif k == "bound_aggr_states":
            c.agent.recurrence.bound_aggr_states = v
        elif k == "features":
            c.agent.features = list(v)
    return VanillaDeepHedgingGym(c)


def roles(gym):
    agent, out = gym.agent, {}
    dense = [l for l in agent._layer.model.layers if isinstance(l, tf.keras.layers.Dense)]
    for i, l in enumerate(dense):
        out[f"dense{i}_W"], out[f"dense{i}_b"] = l.kernel, l.bias
    if agent.is_recurrent:
        out["state0"] = agent._init_state.trainable_weights[0]
    if agent.has_initial_delta:
        out["idelta"] = agent._init_delta.trainable_weights[0]
    for name, u in (("y", gym.utility), ("y0", gym.utility0)):
        if u.trainable_weights:
            out[name] = u.trainable_weights[0]
    return out


def world_arrays(w):
    out = {f"market_{k}": np.asarray(v) for k, v in w.data.market.items()}
    out.update({f"features_{k}": np.asarray(v) for k, v in w.data.features.per_step.items()})
    return out


def gather(d, idx):
    return {k: gather(v, idx) if isinstance(v, dict) else tf.gather(v, idx) for k, v in d.items()}


def main():
    out_dir, kind, case = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    if kind == "world":
        res.update(world_arrays(tf_world(case)))
    elif kind == "gym":
        w, gym = tf_world(GYM_CASES[case]["world"]), tf_gym(case)
        with tf.GradientTape() as tape:
            r = gym(w.tf_data)
            loss = tf.reduce_mean(r["loss"])
        variables = roles(gym)
        grads = tape.gradient(loss, list(variables.values()))
        res["loss"], res["loss_per_path"] = float(loss), r["loss"].numpy()
        res.update(world_arrays(w))
        res.update({f"param_{k}": v.numpy() for k, v in variables.items()})
        res.update({f"grad_{k}": g.numpy() for k, g in zip(variables, grads)})
    elif kind == "train":
        spec = TRAIN_CASES[case]
        w, gym = tf_world(GYM_CASES[spec["gym_case"]]["world"]), tf_gym(spec["gym_case"])
        gym(w.tf_data)
        res.update(world_arrays(w))
        variables = roles(gym)
        res.update({f"init_{k}": v.numpy() for k, v in variables.items()})
        tc = Config()
        tc.optimizer.name = "adam"
        for k, v in spec["optimizer"].items():
            tc.optimizer[k] = v
        gym.compile(optimizer=create_optimizer(tc), loss=dict(loss=default_loss), weighted_metrics=dict(loss=default_loss))
        for idx in train_batches(w.nSamples):
            gym.train_on_batch(x=gather(w.tf_data, idx), y=tf.zeros((len(idx),)), sample_weight=tf.ones((len(idx),)))
        res.update({f"final_{k}": v.numpy() for k, v in variables.items()})
    elif kind == "helpers":
        X, w, fd_spots, fd_times = helper_inputs()
        for u, l in OCE_CASES:
            res[f"oce_{u}_{l}"] = oce_utility(u, l, X)
            res[f"oce_w_{u}_{l}"] = oce_utility(u, l, X, sample_weights=w.copy())
        strips = bs_fd(spots=[fd_spots] * len(fd_times), times=fd_times, payoff=fd_payoff, vol=0.2, cn_factor="implicit")
        res["fd_F0"], res["fd_delta0"] = strips[0].F, strips[0].fd_delta[1]
        strips = bs_fd(spots=[fd_spots] * len(fd_times), times=fd_times, payoff=fd_payoff, vol=0.2, cn_factor=0.5)
        res["fd_cn_F0"] = strips[0].F
        xs = np.sort(X)
        res["mean_bins"], res["mean_bins_std"] = mean_bins(xs, 37, weights=w / w.sum(), return_std=True)
        res["mean_cum_bins"] = mean_cum_bins(xs, 37, weights=w / w.sum())
        res["perct_exp"] = perct_exp(X, 0.25, 0.75, weights=w)
    np.savez(os.path.join(out_dir, f"{kind}_{case}.npz"), **res)
    print(f"{kind}/{case}: wrote {len(res)} arrays")


if __name__ == "__main__":
    main()
