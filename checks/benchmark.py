"""
Epoch time of the TensorFlow original (archive/deephedging) against this port on notebook configurations.
Both sides run their own trainer, including the full training and validation evaluation after each epoch.
Runs alternate between the two sides, one process at a time under a memory cap.

    python checks/benchmark.py --tf-python <TF env python> --jax-python <JAX env python> --reps 3
"""
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(os.path.dirname(ROOT), "archive")
BARRIER = 0.95


def ko_put(spots):
    notko = np.cumprod(np.where(spots > BARRIER, 1., 0.), axis=1)
    return dict(payoff=-np.maximum(1. - spots[:, -1], 0.) * notko[:, -1], features=notko[:, :-1])


CASES = dict(
    trainer=dict(world=dict(samples=10000, steps=20, black_scholes=True), val=1000,
                 agent=dict(depth=3, activation="softplus"), gym=dict(utility="cvar", lmbda=1.),
                 train=dict(batch_size=32), epochs=4),
    fwdstart_all_modes=dict(world=dict(samples=20000, steps=20, drift=0., cost_s=0., black_scholes=True, payoff="fwd_start"), val=10000,
                            agent=dict(depth=5, activation="softplus", classic=2, aggregate=2, past_repr=2, event=2),
                            gym=dict(utility="cvar", lmbda=1.01), train=dict(batch_size=32, clipvalue=0.9, global_clipnorm=1.), epochs=4),
    barrier_100=dict(world=dict(samples=20000, steps=100, dt=1. / 250., drift=0., cost_s=0., black_scholes=True, payoff="ko_put"), val=10000,
                     agent=dict(depth=5, width=50, activation="softplus", classic=1, aggregate=1, past_repr=1, event=1,
                                features=("price", "delta", "time_left", "payoff_features")),
                     gym=dict(utility="exp", lmbda=1.01), train=dict(batch_size=32, clipvalue=0.9, global_clipnorm=1.), epochs=3),
    trainer_bs_batch2000=dict(world=dict(samples=10000, steps=20, dt=5. / 250., drift=0., cost_s=0., black_scholes=True), val=5000,
                              agent=dict(depth=3, activation="softplus"), gym=dict(utility="cvar", lmbda=1.),
                              train=dict(batch_size=2000), epochs=8),
)
PAYOFFS = dict(fwd_start=lambda s: -np.maximum(s[:, 10] - s[:, -1], 0.), ko_put=ko_put)


def world_kwargs(case):
    w = dict(CASES[case]["world"])
    if "payoff" in w:
        w["payoff"] = PAYOFFS[w["payoff"]]
    return w


def run_tf(case):
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3")
    sys.path.insert(0, ARCHIVE)
    from cdxbasics.config import Config
    from deephedging.world import SimpleWorld_Spot_ATM
    from deephedging.gym import VanillaDeepHedgingGym
    import deephedging.trainer as T
    spec, times, losses = CASES[case], [], []
    orig = T.TrainingProgressData.on_epoch_end

    def hook(self, **kw):
        orig(self, **kw)
        times.append(kw["time_epoch"]); losses.append(float(self.losses.training[-1]))
    T.TrainingProgressData.on_epoch_end = hook
    c = Config()
    for k, v in world_kwargs(case).items():
        c.world[k] = v
    for k, v in spec["agent"].items():
        if k in ("depth", "width", "activation"):
            c.gym.agent.network[k] = v
        elif k == "features":
            c.gym.agent.features = list(v)
        else:
            c.gym.agent.recurrence.states[k] = v
    c.gym.objective.utility, c.gym.objective.lmbda = spec["gym"]["utility"], spec["gym"]["lmbda"]
    c.trainer.train.optimizer.name = "adam"
    for k in ("clipvalue", "global_clipnorm"):
        if k in spec["train"]:
            c.trainer.train.optimizer[k] = spec["train"][k]
    c.trainer.train.batch_size, c.trainer.train.epochs = spec["train"]["batch_size"], spec["epochs"]
    c.trainer.caching.mode, c.trainer.output_level = "off", "quiet"
    world = SimpleWorld_Spot_ATM(c.world)
    val = world.clone(samples=spec["val"], seed=1)
    gym = VanillaDeepHedgingGym(c.gym)
    t0 = time.time()
    T.train(gym, world, val, config=c.trainer)
    return dict(wall=time.time() - t0, times=times, losses=losses)


def run_jax(case):
    sys.path.insert(0, ROOT)
    import jax
    jax.config.update("jax_platform_name", "cpu")
    from deephedging.world import WorldConfig, simulate
    from deephedging.model import AgentConfig, GymConfig
    from deephedging.train import TrainConfig, fit
    spec = CASES[case]
    world = simulate(WorldConfig(**world_kwargs(case)))
    val = world.clone(samples=spec["val"], seed=1)
    gym = GymConfig(agent=AgentConfig(**spec["agent"]), **spec["gym"])
    t0 = time.time()
    f = fit(gym, TrainConfig(epochs=spec["epochs"], **spec["train"]), world, val)
    return dict(wall=time.time() - t0, times=f.result.history["time"], losses=f.result.history["loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-python")
    ap.add_argument("--jax-python")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--out", default="/tmp/dh_benchmark.jsonl")
    ap.add_argument("--memory", default="3G")
    ap.add_argument("--inner", nargs=2, metavar=("BACKEND", "CASE"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.inner:
        backend, case = args.inner
        print("RESULT " + json.dumps(dict(backend=backend, case=case, **(run_tf if backend == "tf" else run_jax)(case))), flush=True)
        return
    with open(args.out, "w") as out:
        for rep in range(args.reps):
            for case in args.cases:
                for backend, python in (("tf", args.tf_python), ("jax", args.jax_python)) if rep % 2 == 0 else (("jax", args.jax_python), ("tf", args.tf_python)):
                    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={args.memory}", "-p", "MemorySwapMax=0",
                           python, os.path.abspath(__file__), "--inner", backend, case]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
                    rec = json.loads(line[7:]) if line else dict(backend=backend, case=case, error=r.returncode, stderr=r.stderr[-500:])
                    rec["rep"] = rep
                    out.write(json.dumps(rec) + "\n"); out.flush()
                    print(f"rep {rep} {case:22s} {backend:3s} " + (f"median epoch {np.median(rec['times'][1:]):.3f}s, first {rec['times'][0]:.2f}s, losses {[round(x, 4) for x in rec['losses']]}" if line else f"FAILED exit {r.returncode}"), flush=True)


if __name__ == "__main__":
    main()
