"""
JAX side of the parity check. Feeds the exact TF data from tf_reference.py into the port and
compares loss per path, gradients and parameters after optimizer steps. Exits non-zero on any mismatch.

    python -m parity.check_parity <ref_dir>
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from deephedging.model import AgentConfig, GymConfig, loss, rollout, oce_utility
from deephedging.fd import bs_fd
from deephedging.stats import mean_bins, mean_cum_bins, perct_exp
from deephedging.train import TrainConfig, adam_init, make_step
from .cases import GYM_CASES, TRAIN_CASES, OCE_CASES, train_batches, helper_inputs, fd_payoff

jax.config.update("jax_platform_name", "cpu")
LOSS_RTOL, GRAD_RTOL, TRAIN_RTOL = 1e-5, 1e-3, 1e-3
FLOAT64_RTOL = 1e-8  # float64 sums differ in the last bits across NumPy releases


def gym_config(case):
    spec = GYM_CASES[case]
    return GymConfig(agent=AgentConfig(**spec["agent"]), **spec["gym"])


def ref_batch(ref):
    as3d = lambda x: x[:, :, None] if x.ndim == 2 else x
    market = {k[7:]: jnp.asarray(ref[k], jnp.float32) for k in ref.files if k.startswith("market_")}
    features = {k[9:]: jnp.asarray(as3d(ref[k]), jnp.float32) for k in ref.files if k.startswith("features_")}
    return dict(market=market, features=features)


def to_params(ref, prefix):
    n = len([k for k in ref.files if k.startswith(f"{prefix}dense") and k.endswith("_W")])
    p = dict(dense=[dict(W=jnp.asarray(ref[f"{prefix}dense{i}_W"]), b=jnp.asarray(ref[f"{prefix}dense{i}_b"])) for i in range(n)])
    for k in ("state0", "idelta", "y", "y0"):
        p[k] = jnp.asarray(ref[prefix + k]) if prefix + k in ref.files else jnp.zeros((1,), jnp.float32)
    return p


def flat(p):
    out = {f"dense{i}_{k}": np.asarray(l[k]) for i, l in enumerate(p["dense"]) for k in ("W", "b")}
    out.update({k: np.asarray(p[k]) for k in ("state0", "idelta", "y", "y0") if k in p})
    return out


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-8))


def main():
    ref_dir, failures = sys.argv[1], []

    def check(label, value, tol):
        ok = value <= tol
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {value:.2e} (tol {tol:.0e})")
        if not ok:
            failures.append(label)

    for case in GYM_CASES:
        ref = np.load(os.path.join(ref_dir, f"gym_{case}.npz"))
        gym, data = gym_config(case), ref_batch(ref)
        p = to_params(ref, "param_")
        value, grads = jax.value_and_grad(loss)(p, gym, data)
        check(f"gym {case} loss", abs(float(value) - float(ref["loss"])) / abs(float(ref["loss"])), LOSS_RTOL)
        per_path = np.asarray(rollout(p, gym, data)["loss"])
        check(f"gym {case} per-path loss", rel(per_path, ref["loss_per_path"]), LOSS_RTOL * 10)
        g = flat(grads)
        worst = max(rel(g[k], ref[f"grad_{k}"]) for k in g if f"grad_{k}" in ref.files)
        check(f"gym {case} grads (worst of {len(g)})", worst, GRAD_RTOL)

    for case, spec in TRAIN_CASES.items():
        ref = np.load(os.path.join(ref_dir, f"train_{case}.npz"))
        gym, data = gym_config(spec["gym_case"]), ref_batch(ref)
        tc = TrainConfig(**spec["optimizer"])
        p = to_params(ref, "init_")
        step = jax.jit(lambda carry, idx, data: make_step(gym, tc)(carry, idx, data)[0])
        carry, n = (p, adam_init(p)), data["market"]["payoff"].shape[0]
        for idx in train_batches(n):
            carry = step(carry, jnp.asarray(idx), data)
        fin = flat(carry[0])
        worst = max(rel(fin[k], ref[f"final_{k}"]) for k in fin if f"final_{k}" in ref.files)
        moved = max(float(np.max(np.abs(fin[k] - ref[f"init_{k}"]))) for k in fin if f"init_{k}" in ref.files)
        check(f"train {case} params after {len(train_batches(n))} steps (max param change {moved:.1e})", worst, TRAIN_RTOL)

    ref = np.load(os.path.join(ref_dir, "helpers_all.npz"))
    X, w, fd_spots, fd_times = helper_inputs()
    for u, l in OCE_CASES:
        for tag, weights in (("", None), ("w_", w)):
            ref_v = float(ref[f"oce_{tag}{u}_{l}"])
            check(f"oce_utility {tag}{u}@{l}", abs(oce_utility(u, l, X, weights) - ref_v) / max(abs(ref_v), 1e-8), 1e-4)
    strips = bs_fd(spots=[fd_spots] * len(fd_times), times=fd_times, payoff=fd_payoff, vol=0.2, cn_factor="implicit")
    check("bs_fd implicit value", rel(strips[0].F, ref["fd_F0"]), FLOAT64_RTOL)
    check("bs_fd implicit delta", rel(strips[0].fd_delta[1], ref["fd_delta0"]), FLOAT64_RTOL)
    strips = bs_fd(spots=[fd_spots] * len(fd_times), times=fd_times, payoff=fd_payoff, vol=0.2, cn_factor=0.5)
    check("bs_fd crank-nicolson value", rel(strips[0].F, ref["fd_cn_F0"]), FLOAT64_RTOL)
    xs = np.sort(X)
    m, s = mean_bins(xs, 37, weights=w / w.sum(), return_std=True)
    check("mean_bins", max(rel(m, ref["mean_bins"]), rel(s, ref["mean_bins_std"])), FLOAT64_RTOL)
    check("mean_cum_bins", rel(mean_cum_bins(xs, 37, weights=w / w.sum()), ref["mean_cum_bins"]), FLOAT64_RTOL)
    check("perct_exp", rel(perct_exp(X, 0.25, 0.75, weights=w), ref["perct_exp"]), FLOAT64_RTOL)

    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
