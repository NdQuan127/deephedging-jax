"""
Static version of plot_training.Plotter: the same panels, drawn once from a TrainResult.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from .stats import mean_bins, mean_cum_bins, perct_exp

GAINS, PAYOFF, HEDGE = "tab:blue", "tab:orange", "tab:green"


def _uniform(n):
    return np.full((n,), 1. / n)


def _ylim(ax, *arrays):
    lo, hi = min(np.min(a) for a in arrays), max(np.max(a) for a in arrays)
    d = max(hi - lo, 1e-4)
    ax.set_ylim(lo - d / 20., hi + d / 20.)


def plot_losses(ax, history, best_epoch, lookback=None):
    n = len(history["loss"])
    start = 0 if lookback is None else max(0, n - lookback)
    x = np.arange(start + 1, n + 1)
    ax.plot(x, history["loss"][start:], label="training")
    ax.plot(x, history["val_loss"][start:], label="validation")
    if best_epoch >= start:
        ax.plot([best_epoch + 1], [history["loss"][best_epoch]], "*", color="red", label="best")
    ax.set_title("Losses" + (" (recent)" if lookback else " (all)"))
    ax.set_xlabel("Epoch")
    ax.legend()


def plot_utility_by_epoch(ax, history, key, label):
    x = np.arange(1, len(history[key]) + 1)
    ax.plot(x, history[key], label="training")
    ax.plot(x, history["val_" + key], label="validation")
    ax.set_title(f"Utility of {label}")
    ax.set_xlabel("Epoch")
    ax.legend()


def plot_returns_by_spot(ax, r, spot_all, title, bins, with_std):
    P = _uniform(len(r["gains"]))
    gains = r["gains"] - np.mean(r["utility"])
    payoff = r["payoff"] - np.mean(r["utility0"])
    hedge = gains - payoff
    ret = spot_all[:, -1] / spot_all[:, 0] - 1.
    ixs = np.argsort(ret)
    x = mean_bins(ret[ixs], bins=bins, weights=P)
    for y, name, color, style in ((gains, "gains", GAINS, "-"), (payoff, "payoff", PAYOFF, ":"), (-hedge, "-hedge", HEDGE, "-")):
        m, s = mean_bins(y[ixs], bins=bins, weights=P, return_std=True)
        ax.plot(x, m, style, color=color, label=name)
        if with_std:
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
    ax.plot(x, x * 0., ":", color="black")
    _ylim(ax, mean_bins(gains[ixs], bins=bins, weights=P), mean_bins(payoff[ixs], bins=bins, weights=P))
    ax.set_title(title)
    ax.set_xlabel("Spot return")
    ax.legend()


def plot_utility_by_percentile(ax, r, title, bins):
    P = _uniform(len(r["utility"]))
    u = mean_cum_bins(np.sort(r["utility"]), bins=bins, weights=P)
    u0 = mean_cum_bins(np.sort(r["utility0"]), bins=bins, weights=P)
    x = np.linspace(0., 1., len(u), endpoint=True)
    ax.plot(x, u, label="gains", color=GAINS)
    ax.plot([x[-1]], [u[-1]], "*", color=GAINS)
    ax.plot(x, u0, label="payoff", color=PAYOFF)
    ax.plot(x, u0 * 0., ":", color="black")
    _ylim(ax, u, u0)
    ax.set_title(title)
    ax.set_xlabel("Percentile")
    ax.legend()


def plot_activity_by_step(ax, activity, title, inst_names, pcnt_lo, pcnt_hi):
    P = _uniform(activity.shape[0])
    x = np.arange(1, activity.shape[1] + 1)
    for i, name in enumerate(inst_names):
        a = activity[:, :, i]
        pct = perct_exp(a, lo=pcnt_lo, hi=pcnt_hi, weights=P)
        line, = ax.plot(x, np.sum(a * P[:, None], axis=0), label=name)
        ax.fill_between(x, pct[:, 0], pct[:, 1], color=line.get_color(), alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.legend()


def plot_activity_by_spot(ax, activity, spot_all, title, bins, slices, with_std):
    P = _uniform(activity.shape[0])
    n_steps = activity.shape[1]
    for t in np.linspace(0, n_steps - 1, min(slices, n_steps), endpoint=True, dtype=np.int32):
        x = spot_all[:, t] / spot_all[:, 0] - 1.
        ixs = np.argsort(x)
        m, s = mean_bins(activity[:, t, 0][ixs], bins=bins, weights=P, return_std=True)
        xb = mean_bins(x[ixs], bins=bins, weights=P)
        r = 2. * (1. - float(t + 1) / float(n_steps))
        color = (1., max(min(r - 1., 1.), 0.), max(min(r, 1.), 0.))
        ax.plot(xb, m, color=color, label=str(t + 1))
        if with_std:
            ax.fill_between(xb, m - s, m + s, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Spot return")
    ax.legend(fontsize="small")


def plot_training(res, train_rollout, val_rollout, world, val_world, *, inst_names=None, bins=200, slices=10,
                  pcnt_lo=0.25, pcnt_hi=0.75, lookback=100, col_size=5, row_size=4):
    """ Training summary figure. The rollouts are model.rollout() of res.params on both worlds. """
    tr = {k: np.asarray(v) for k, v in train_rollout.items()}
    va = {k: np.asarray(v) for k, v in val_rollout.items()}
    inst_names = inst_names or (["spot", "ATM Call"][:world.nInst])
    deltas = np.cumsum(tr["actions"], axis=1)
    fig, axes = plt.subplots(4, 4, figsize=(4 * col_size, 4 * row_size))
    axes = axes.ravel()
    plot_losses(axes[0], res.history, res.best_epoch, lookback=lookback)
    plot_losses(axes[1], res.history, res.best_epoch)
    plot_utility_by_epoch(axes[2], res.history, "utility", "model gains")
    plot_utility_by_epoch(axes[3], res.history, "utility0", "original payoff")
    plot_returns_by_spot(axes[4], tr, world.spot_all, "Returns less utility\n(training set)", bins, False)
    plot_returns_by_spot(axes[5], tr, world.spot_all, "Returns less utility, with std\n(training set)", bins, True)
    plot_utility_by_percentile(axes[6], tr, "Utility by cumulative percentile\n(training set)", bins)
    axes[7].axis("off")
    plot_returns_by_spot(axes[8], va, val_world.spot_all, "Returns less utility\n(validation set)", bins, False)
    plot_returns_by_spot(axes[9], va, val_world.spot_all, "Returns less utility, with std\n(validation set)", bins, True)
    plot_utility_by_percentile(axes[10], va, "Utility by cumulative percentile\n(validation set)", bins)
    plot_activity_by_step(axes[11], tr["actions"], "Action by time step\n(training set)", inst_names, pcnt_lo, pcnt_hi)
    plot_activity_by_step(axes[12], deltas, "Delta by time step\n(training set)", inst_names, pcnt_lo, pcnt_hi)
    plot_activity_by_spot(axes[13], tr["actions"], world.spot_all, "Spot action by time step\n(training set)", bins, slices, False)
    plot_activity_by_spot(axes[14], deltas, world.spot_all, "Spot delta by time step\n(training set)", bins, slices, False)
    plot_activity_by_spot(axes[15], deltas, world.spot_all, "Spot delta by time step, with std\n(training set)", bins, slices, True)
    fig.tight_layout()
    return fig


def plot_world(world, plot_samples=5, col_size=5, row_size=4):
    """ Port of SimpleWorld_Spot_ATM.plot(), with a linear time axis. """
    samples = np.linspace(0, world.nSamples, plot_samples, endpoint=False, dtype=int)
    t_all, t = world.timeline, world.timeline[:-1]
    f = {k: v[:, :, 0] for k, v in world.features.items() if v.shape[2] == 1}
    panels = [("Spot", t_all, world.spot_all, None), ("Drift", t, world.details["drift"], None),
              ("Volatilities", t, f["ivol"], world.details["rvol"])]
    if world.nInst > 1:
        panels += [("Call prices", t, f["call_price"], None), ("Call deltas", t, f["call_delta"], None), ("Call vegas", t, f["call_vega"], None)]
    fig, axes = plt.subplots((len(panels) + 2) // 3, 3, figsize=(3 * col_size, (len(panels) + 2) // 3 * row_size), squeeze=False)
    for ax, (title, x, y, y2) in zip(axes.ravel(), panels):
        for i in samples:
            line, = ax.plot(x, y[i], "-")
            if y2 is not None:
                ax.plot(x, y2[i], ":", color=line.get_color())
        if y2 is None:
            ax.plot(x, np.mean(y, axis=0), "_", color="black", label="mean")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_blackscholes(world, rollout_result, gym, *, vol=0.2, strike=1., iscall=True, delta_bins=100, steps=10):
    """
    Port of plot_bs_hedge.plot_blackscholes for a hedged vanilla option in a Black & Scholes world.
    'steps' limits the per-step price and delta panels to that many evenly spaced steps.
    Unlike the original, the BS bar shows the utility of the BS-hedged position (it showed the unhedged payoff's).
    """
    from .model import oce_utility
    r = {k: np.asarray(v) for k, v in rollout_result.items()}
    spot, spotT = world.spot_all[:, :-1], world.spot_all[:, -1]
    hedges, costs = world.market["hedges"][:, :, 0], world.market["cost"][:, :, 0]
    deltas = np.cumsum(r["actions"][:, :, 0], axis=1)
    time_left = world.features["time_left"][0, :, 0]
    payoff, dhpnl = r["payoff"], r["pnl"]
    utility, utility0 = np.mean(r["utility"]), np.mean(r["utility0"])
    dprice = utility - utility0
    n, n_steps = spot.shape
    bin_ixs = np.linspace(0, n, delta_bins + 1, endpoint=True, dtype=np.int32)
    binned = lambda x: np.array([np.mean(x[bin_ixs[i]:bin_ixs[i + 1]]) for i in range(delta_bins)])
    lohi = perct_exp(np.sort(spotT), 0.01, .99)
    bin_bnd = np.linspace(lohi[0], lohi[1], 21, endpoint=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax_payoff, ax_terminal, ax_utility, ax_spots, ax_hedges, ax_off = axes.ravel()
    ax_off.axis("off")
    shown = set(np.linspace(0, n_steps - 1, min(steps, n_steps), dtype=int))
    fig_path, axes_path = plt.subplots((2 * len(shown) + 5) // 6, 6, figsize=(24, 3.5 * ((2 * len(shown) + 5) // 6)), squeeze=False)
    path_axes = iter(axes_path.ravel())

    last_delta = last_bsdelta = 0.
    pnl = bspnl = bscost = 0.
    for j in range(n_steps):
        spot_t, delta_t, hedges_t, cost_t, res_t = spot[:, j], deltas[:, j], hedges[:, j], costs[:, j], time_left[j]
        d1 = (np.log(spot_t / strike) + 0.5 * vol * vol * res_t) / np.sqrt(res_t * vol * vol)
        d2 = d1 - vol * np.sqrt(res_t)
        bsprice_t = spot_t * norm.cdf(d1) - strike * norm.cdf(d2)
        bsprice_t = bsprice_t if iscall else bsprice_t + strike - spot_t
        bsdelta_t = norm.cdf(d1) if iscall else norm.cdf(d1) - 1.
        pnl = (delta_t - last_delta) * hedges_t + pnl
        bspnl = (bsdelta_t - last_bsdelta) * hedges_t + bspnl
        bscost = np.abs(bsdelta_t - last_bsdelta) * cost_t + bscost
        last_delta, last_bsdelta = delta_t, bsdelta_t
        days = j * world.dt * 255
        hist, _ = np.histogram(spot_t, bins=bin_bnd)
        ax_spots.plot(0.5 * (bin_bnd[1:] + bin_bnd[:-1]), hist / max(hist.sum(), 1), color=(float(j + 1) / n_steps, 0.5, 0.5))
        ixs = np.argsort(spot_t)
        ax_hedges.plot(spot_t[ixs], hedges_t[ixs], color=(0.7, float(j + 1) / n_steps, 0.7) if j < n_steps - 1 else (0.8, 1, 0.8))
        if j not in shown:
            continue
        marker = "-" if j > 0 else "o"
        sim_price_t = pnl + delta_t * (spotT - spot_t)
        ax = next(path_axes)
        ax.plot(binned(spot_t[ixs]), binned(sim_price_t[ixs]), marker, label="model approximation", color=(0., 0, 1))
        ax.plot(binned(spot_t[ixs]), binned(bsprice_t[ixs]) - dprice, marker, label="Black Scholes", color=(0.8, 0., 0.))
        ax.set_title("-Payoff %g days" % days)
        ax.legend(fontsize="small")
        ax = next(path_axes)
        ax.plot(spot_t[ixs], delta_t[ixs], marker, label="model", color=(0., 0., 1.), alpha=0.3)
        ax.plot(binned(spot_t[ixs]), binned(delta_t[ixs]), marker, label="model (smoothed)", color=(0., 0., 0.5))
        ax.plot(spot_t[ixs], bsdelta_t[ixs], marker, label="black scholes", color=(0.8, 0., 0.))
        ax.set_title("Delta %g days" % days)
        ax.legend(fontsize="small")
    for ax in path_axes:
        ax.axis("off")

    utility_bs = oce_utility(gym.utility, gym.lmbda, payoff + bspnl - bscost)
    ixs = np.argsort(spotT)
    x = binned(spotT[ixs])
    gains = [binned((pnl + payoff)[ixs]), binned((bspnl + payoff)[ixs]), binned((dhpnl + payoff)[ixs])]
    for y, style, color, label in zip(gains, ("*-", "-", ":"), ("orange", "green", "black"), ("hedged pnl", "bs hedged pnl", "hedged pnl from DH")):
        ax_terminal.plot(x, y, style, color=color, label=label)
    _ylim(ax_terminal, *gains)
    ax_terminal.legend()
    ax_terminal.set_title("Terminal hedged results")
    eff = [binned(payoff[ixs]), -binned(pnl[ixs]) - dprice, -binned(bspnl[ixs]) - dprice]
    for y, style, color, label in zip(eff, ("-", "*-", "-"), ("blue", "orange", "green"), ("payoff", "-model hedged pnl", "-bs hedged pnl")):
        ax_payoff.plot(x, y, style, color=color, label=label)
    _ylim(ax_payoff, *eff)
    ax_payoff.legend()
    ax_payoff.set_title("Effective terminal payoffs")
    ax_utility.bar(["Unhedged", "BS", "DH"], [utility0, utility_bs, utility])
    ax_utility.set_title(f"Utility {gym.utility}@{gym.lmbda:g}\n(higher is better)")
    ax_spots.set_title("Spot distribution in t")
    ax_hedges.set_title("Hedge returns")
    fig.tight_layout()
    fig_path.tight_layout()
    return fig, fig_path
