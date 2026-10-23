"""
Parity cases shared by tf_reference.py (TensorFlow side) and check_parity.py (JAX side).
Keys mirror the original Config names so both sides build the same objects.
"""
import numpy as np

I_RESET, BARRIER = 10, 0.95


def fwd_start(spots):
    return -np.maximum(1. * spots[:, I_RESET] - spots[:, -1], 0.)


def ko_put(spots):
    notko = np.cumprod(np.where(spots[:, :] > BARRIER, 1., 0.), axis=1)
    return dict(payoff=-np.maximum(1. - spots[:, -1], 0.) * notko[:, -1], features=notko[:, :-1])


FWD_WORLD = dict(steps=20, samples=4000, drift=0., cost_s=0., black_scholes=True, payoff=fwd_start)
WORLDS = dict(
    fwdstart=FWD_WORLD,
    barrier=dict(steps=20, samples=4000, drift=0., cost_s=0., black_scholes=True, payoff=ko_put),
    sv=dict(steps=20, samples=4000),
)

FWD_AGENT = dict(depth=5, activation="softplus")
FWD_GYM = dict(utility="cvar", lmbda=1.01)
FWD_VARIANTS = dict(
    none=dict(classic=0, aggregate=0, past_repr=0, event=0),
    all_modes=dict(classic=2, aggregate=2, past_repr=2, event=2),
    aggregate_2=dict(classic=0, aggregate=2, past_repr=0, event=0),
    aggregate_5bnd=dict(classic=0, aggregate=5, past_repr=1, event=0, bound_aggr_states=True),
    aggregate_5=dict(classic=0, aggregate=5, past_repr=0, event=0),
    past_repr=dict(classic=0, aggregate=0, past_repr=2, event=0),
)

GYM_CASES = {f"fwd_{k}": dict(world="fwdstart", agent=dict(FWD_AGENT, **v), gym=FWD_GYM) for k, v in FWD_VARIANTS.items()}
GYM_CASES.update(
    barrier_exp=dict(world="barrier", gym=dict(utility="exp", lmbda=1.01),
                     agent=dict(depth=5, width=50, activation="softplus", classic=1, aggregate=1, past_repr=1, event=1,
                                features=("price", "delta", "time_left", "payoff_features"))),
    sv_exp2=dict(world="sv", gym=dict(utility="exp2", lmbda=1.), agent=dict()),
    fwd_quad=dict(world="fwdstart", gym=dict(utility="quad", lmbda=1.), agent=dict(FWD_AGENT)),
    fwd_vicky=dict(world="fwdstart", gym=dict(utility="vicky", lmbda=1.), agent=dict(FWD_AGENT)),
)

TRAIN_STEPS, TRAIN_BATCH = 15, 32
TRAIN_CASES = dict(
    notebook_clip=dict(gym_case="fwd_all_modes", optimizer=dict(learning_rate=0.001, clipvalue=0.9, global_clipnorm=1.)),
    clipvalue_only=dict(gym_case="barrier_exp", optimizer=dict(learning_rate=0.001, clipvalue=0.05)),
)


def train_batches(n: int):
    rng = np.random.default_rng(123)
    return np.stack([rng.choice(n, TRAIN_BATCH, replace=False) for _ in range(TRAIN_STEPS)])


def helper_inputs():
    rng = np.random.default_rng(7)
    X = (rng.standard_normal(3000) * 0.3 - np.maximum(rng.standard_normal(3000), 0.)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, 3000)
    fd_spots = np.exp(np.linspace(-1., 1., 201)).astype(np.float64)
    fd_times = np.linspace(0., 20. / 50., 21)
    return X, w, fd_spots, fd_times


OCE_CASES = [(u, l) for u in ("mean", "exp", "exp2", "vicky", "cvar", "quad") for l in (0.5, 1.01, 10.)]


def fd_payoff(X, F, t):
    terminal = np.maximum(1. - X, 0.) if F is None else F
    return np.where(X > BARRIER, terminal, 0.)
