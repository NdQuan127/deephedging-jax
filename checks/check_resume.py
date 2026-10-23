import os, sys, shutil, tempfile, numpy as np, jax
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deephedging.world import WorldConfig, simulate
from deephedging.model import AgentConfig, GymConfig, init_params, as_batch
from deephedging.train import TrainConfig, train
from parity.cases import fwd_start

w = simulate(WorldConfig(steps=20, samples=2000, drift=0., cost_s=0., black_scholes=True, payoff=fwd_start))
v = w.clone(samples=500, seed=1)
gym = GymConfig(agent=AgentConfig(depth=5, activation="softplus", classic=1, aggregate=1, past_repr=1, event=1), utility="cvar", lmbda=1.01)
p0 = init_params(jax.random.PRNGKey(0), gym, {k: x.shape[2] for k, x in w.features.items()}, w.nInst)
tc = TrainConfig(epochs=6, batch_size=32, clipvalue=0.9, global_clipnorm=1.)
d, vd = as_batch(w), as_batch(v)
ref = train(gym, tc, p0, d, vd)
CKPT = tempfile.mkdtemp()
calls = []
part = train(gym, TrainConfig(**{**tc.__dict__, "epochs": 3}), p0, d, vd, checkpoint_dir=CKPT, checkpoint_every=1)
res = train(gym, tc, p0, d, vd, checkpoint_dir=CKPT, checkpoint_every=1, on_epoch=lambda e, h, b: calls.append(e))
again = train(gym, tc, p0, d, vd, checkpoint_dir=CKPT, on_epoch=lambda e, h, b: calls.append(("again", e)))
leaves = lambda t: jax.tree_util.tree_leaves(t)
same = lambda a, b: all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(leaves(a), leaves(b)))
assert calls == [3, 4, 5], f"resume should run epochs 3..5 only, ran {calls}"
assert same(res.last_params, ref.last_params), "resumed last params differ from uninterrupted run"
assert same(res.params, ref.params) and res.best_epoch == ref.best_epoch, "resumed best params differ"
assert res.history["loss"] == ref.history["loss"] and res.history["val_loss"] == ref.history["val_loss"], "history differs"
assert same(again.params, ref.params), "a finished run must return the same best params without training"
print("resume ok: best epoch", ref.best_epoch + 1, "losses", [round(x, 5) for x in ref.history["loss"]])
