# Deep Hedging in JAX

A JAX port of Hans Buehler's [Deep Hedging sample code](https://github.com/hansbuehler/deephedging), the reference implementation of [Deep Hedging](https://arxiv.org/abs/1802.03042).
It trains the same networks on the same simulated worlds with the same objective and optimizer, and matches the TensorFlow code to float32 rounding.
It needs no TensorFlow and runs on current Python and on Google Colab.

`checks/benchmark.py` times one epoch of each version on notebook configurations, including the evaluation on the training and validation sets after each epoch.
On a 12-core CPU, with the two versions alternating and three runs each (median, range in brackets):

| Configuration | TensorFlow | JAX | Speedup |
| --- | --- | --- | --- |
| `trainer`: 20 steps, width 20, batch 32 | 3.99 s [3.76, 4.15] | 0.211 s [0.208, 0.212] | 18.9x |
| `trainer-recurrent-fwdstart`, all modes: 20 steps, 2 states of each kind | 6.58 s [6.51, 7.12] | 0.757 s [0.747, 0.815] | 8.7x |
| `trainer-recurrent-barrier`: 100 steps, width 50, recurrent | 32.6 s [31.9, 35.2] | 7.05 s [6.97, 8.23] | 4.6x |
| `trainer-bs`: 20 steps, batch 2000 | 0.157 s [0.152, 0.157] | 0.158 s [0.157, 0.159] | 1.0x |

The port gains most at batch 32, where the TensorFlow loop spends its time launching many small operations at every time step.
At batch 2000 both versions spend the time on the same arithmetic, so the port is no faster.

## Install

```bash
pip install -e ".[notebooks]"
```

The package needs `jax`, `numpy`, `scipy`, and `matplotlib`.
For a GPU, install the CUDA build of JAX as described in the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
On Colab, clone this folder and run the notebooks from `notebooks/`; JAX is preinstalled there.

## Run the notebooks

Open a notebook in `notebooks/` and run all cells.

| Notebook | Port of |
| --- | --- |
| `trainer.ipynb` | `notebooks/trainer.ipynb` |
| `trainer-bs.ipynb` | `notebooks/trainer-bs.ipynb` |
| `simpleWorld_Spot_ATM.ipynb` | `notebooks/simpleWorld_Spot_ATM.ipynb` |
| `blackscholes.ipynb` | the five `notebooks/blackscholes/trainer_bs_*.ipynb` |
| `trainer-recurrent-fwdstart.ipynb` | `notebooks/trainer-recurrent-fwdstart.ipynb` |
| `trainer-recurrent-barrier_smalldt.ipynb` | `notebooks/trainer-recurrent-barrier_smalldt.ipynb` |
| `trainer-recurrent-barrier.ipynb` | `notebooks/trainer-recurrent-barrier.ipynb` |
| `trainer-american_digital.ipynb` | `notebooks/trainer-ray-american_digital.ipynb`, without Ray |

Two environment variables control a run:

- `DH_EPOCHS=2` overrides every epoch count, for a quick check that a notebook runs.
- `DH_CHECKPOINTS=""` turns checkpoints off. By default each training writes a checkpoint to `notebooks/checkpoints/` every 10 epochs.

A training that stops, for example after a crash or a kernel restart, resumes from its last checkpoint when you run the cell again.
A training that already reached its epoch count returns at once.
The resumed result is identical to an uninterrupted run, because each epoch shuffles with a generator seeded by the seed and the epoch number.
`checks/check_resume.py` proves this.

To run every notebook once with one epoch and a memory cap:

```bash
python checks/run_notebooks.py notebooks/*.ipynb
```

## Use the package

```python
from deephedging.world import WorldConfig, simulate
from deephedging.model import AgentConfig, GymConfig
from deephedging.train import TrainConfig, fit
from deephedging.plotting import plot_training

world = simulate(WorldConfig(samples=10000, steps=20, black_scholes=True))
val_world = world.clone(samples=1000, seed=1)
gym = GymConfig(agent=AgentConfig(depth=3, activation="softplus"), utility="cvar", lmbda=1.)
result = fit(gym, TrainConfig(epochs=800), world, val_world, checkpoint_dir="checkpoints")
plot_training(result.result, result.train, result.val, world, val_world)
```

The configs are frozen dataclasses whose fields keep the original `Config` names.

| Original | Port |
| --- | --- |
| `config.world.*`, `SimpleWorld_Spot_ATM` | `WorldConfig`, `world.simulate` |
| `config.gym.agent.network.*`, `recurrence.*`, `features` | `AgentConfig` |
| `config.gym.objective.utility`, `lmbda` | `GymConfig` |
| `config.trainer.train.*` and the optimizer settings | `TrainConfig` (`batch_size=None` in the original means 32) |
| `VanillaDeepHedgingGym(...)(world.tf_data)` | `model.rollout(params, gym, model.as_batch(world))` |
| `trainer.train` | `train.fit`, or `train.train` with explicit initial weights |
| `world.details.spot_all` | `world.spot_all` |
| `world.data.market.X`, `world.data.features.per_step.X` | `world.market["X"]`, `world.features["X"]` with shape `(samples, steps, n)` |
| `objectives.oce_utility` | `model.oce_utility` |
| `fd.bs_fd`, `base.mean_bins`, `base.perct_exp` | `fd.bs_fd`, `stats.mean_bins`, `stats.perct_exp` |
| `plot_training.Plotter`, `plot_bs_hedge.plot_blackscholes`, `world.plot` | `plotting.plot_training`, `plotting.plot_blackscholes`, `plotting.plot_world` |

## Parity with the TensorFlow code

`parity/` compares the port with the original code on the same inputs.
`parity/tf_reference.py` runs in the original environment (Python 3.10, TensorFlow 2.11, `tensorflow_probability` 0.19, `cdxbasics`) and writes one reference file per case in `parity/cases.py`.
It loads the original package from `../archive/deephedging`; set `DH_ORIGINAL` to the directory that contains the original `deephedging` if it lives elsewhere.

```bash
python -m parity.tf_reference refs world fwdstart   # once per case, original environment
python -m parity.check_world refs                   # original environment
python -m parity.check_parity refs                  # JAX environment
```

The last verified run passed every check:

- The simulated worlds are bit-identical for the forward-start, barrier, and stochastic-volatility worlds.
  Run `check_world` with the same NumPy version as the TensorFlow side, because float32 `exp` and `mean` differ in the last bit across NumPy releases.
- Per-path losses match within 1e-5 and the gradients of every trainable variable within 7e-5, relative.
  The cases cover the six forward-start variants, the barrier payoff with `payoff_features` and the `exp` utility, the two-instrument world with `exp2`, and `quad` and `vicky`.
- Parameters after 15 Adam steps match within 1.1e-6, both with the notebooks' clipping and with `clipvalue` alone.
- `oce_utility` matches for all six utilities and three risk aversions, with and without sample weights.
  `bs_fd` and the statistics helpers match within 1e-8.

## Differences from the original

- The validation world uses a fixed seed. The original drew a random seed.
- Keras 2.11 ignores `clipvalue` when `global_clipnorm` is set. Several notebooks set both, so only the global-norm clip applies, in the original and in the port.
- The original `trainer-recurrent-fwdstart.ipynb` cell for `aggregate_5bnd` assigns `event` twice and never resets `past_repr`. That variant keeps one `past_repr` state in both versions.
- `utility="mean"` crashes in the original because `VariableLayer(0.)` returns shape `[N]`. It works in the port.
- `plot_blackscholes` shows the utility of the Black & Scholes hedged position in its bar chart. The original showed the utility of the unhedged payoff under the "BS" label.
- `plot_world` uses a linear time axis. The original plotted against the cumulative sum of the step indices.
- The analysis cells of both barrier notebooks accumulate each strategy's P&L as the trade at each step times `S(T) - S(t)`.
  The original multiplied the position (the cumulative delta) by `S(T) - S(t)`, which counts every position again at each later step.
  The trade formula reproduces the gym's `pnl` to 7e-7 on the same actions, while the original formula differed by up to 34 per path.
  The original `trainer-ray-american_digital.ipynb` already used trades.
- The Black & Scholes reference in both barrier notebooks hedges with minus the FD delta.
  `fd_payoff` prices the payoff Z itself, so its delta replicates Z, and the hedge of Z is that delta with the opposite sign.
  The original plotted the replicating delta next to the learned hedges, so the reference was mirrored.
  `trainer-recurrent-fwdstart.ipynb` already used the hedge sign (`N(d1) - 1`, the delta of a long put against a short put payoff).
- The training monitor is a figure drawn after training instead of a live plot. Pass `on_epoch` to `train` or `fit` for your own progress display.

These behaviours of the original are kept on purpose, so that results stay comparable:

- The FD grid for the Black & Scholes reference scales with `vol * T` in the barrier notebooks and with `vol * sqrt(T)` in the American digital notebook.
- `trainer-recurrent-barrier_smalldt.ipynb` has 20 steps at the default `dt`, and `trainer-recurrent-barrier.ipynb` has 100 smaller steps. The names say the opposite.

- With an option as hedging instrument (`black_scholes=False`), the payoff function receives `spots[:, :steps+2]`, so the default payoff uses the spot one step after maturity.
- The simulator rescales spots so that their cross-sectional mean is `exp(drift * dt)` at every step, which removes the drift after the first step.

## Not ported

`sigmoid_1` recurrent states, network-valued OCE intercepts (`objective.y.features`), `hard_clip`, TensorBoard, and the `notebooks/misc` and `notebooks/under_construction` notebooks.
The `misc` notebooks use a separate model (`model_base.py`).

## License

GPL-3.0, as the original. See `LICENSE`.
