"""Turns results/*.json into figures (figs/*.svg) and summary numbers (summary.json)."""
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = lambda n: json.load(open(f"results/{n}.json"))
os.makedirs("figs", exist_ok=True)

# reference light palette (dataviz skill), fixed order
C_ES, C_PW, C_FULL, C_ALT = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "lines.markersize": 5, "legend.frameon": False, "svg.fonttype": "none",
    "axes.titlesize": 9, "axes.titleweight": "bold",
})

S = {}


def ci(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return xs.mean(), 0.0
    t = {8: 2.365, 15: 2.145, 20: 2.093, 30: 2.045}.get(len(xs), 1.96)
    return xs.mean(), t * xs.std(ddof=1) / math.sqrt(len(xs))


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key] if isinstance(key, str) else key(r)].append(r)
    return dict(sorted(g.items()))


def line(ax, xs, rows_by_x, f, color, label, marker="o", ls="-"):
    m = [ci([f(r) for r in rows_by_x[x]]) for x in xs]
    y = np.array([a for a, _ in m]); e = np.array([b for _, b in m])
    ax.plot(xs, y, color=color, marker=marker, ls=ls, label=label)
    ax.fill_between(xs, y - e, y + e, color=color, alpha=0.15, lw=0)
    return y


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figs/{name}.svg")
    plt.close(fig)


# ---------------------------------------------------------- overhead (H2)
ov = R("overhead")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for nrx, col, lab in ((1, C_ES, "1 receiver (unicast)"), (10, C_PW, "10 receivers (broadcast, max)")):
    g = group([r for r in ov if r["n_rx"] == nrx], "d")
    xs = list(g)
    y = line(ax, xs, g, lambda r: r["ratio"], col, lab)
    S[f"overhead_{nrx}"] = {str(x): round(v, 3) for x, v in zip(xs, y)}
    S[f"overhead_{nrx}_p95"] = {str(x): round(float(np.percentile([r["ratio"] for r in g[x]], 95)), 3) for x in xs}
ax.axhline(1.35, color=INK2, ls="--", lw=1)
ax.text(300, 1.2, "asymptotic 1.35", color=INK2, fontsize=7.5, ha="center")
ax.set_xscale("log"); ax.set_ylim(1, 3.6)
ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"]); ax.minorticks_off()
ax.set_xlabel("set difference d (items)"); ax.set_ylabel("coded symbols / d")
ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 1.02))
save(fig, "fig_overhead")

# ---------------------------------------------------------- scaling (H1, RQ1)
sc = R("scaling")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, metric, ylab in ((axs[0], "airtime", "total network airtime (s)"),
                         (axs[1], "wall", "time to full convergence (h)")):
    rows = [r for r in sc if r["loss"] == 0.0]
    g = group(rows, "n"); xs = list(g)
    k = 1 if metric == "airtime" else 1 / 3600
    line(ax, xs, g, lambda r: r["es2"][metric] * k, C_ES, "EchoSync (2-phase)")
    line(ax, xs, g, lambda r: r["pw"][metric] * k, C_PW, "Pairwise RIBLT (star)", marker="s")
    line(ax, xs, g, lambda r: r["full"][metric] * k, C_FULL, "Full-state broadcast", marker="^")
    if metric == "airtime":
        line(ax, xs, g, lambda r: r["oracle"]["airtime"], INK2, "Oracle lower bound", marker="", ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(xs); ax.minorticks_off()
    ax.set_xlabel("number of nodes N"); ax.set_ylabel(ylab)
axs[0].set_title("(a) Airtime, lossless"); axs[1].set_title("(b) Wall-clock time, 1% duty cycle")
axs[0].legend(fontsize=7)
save(fig, "fig_scaling")

for loss in (0.0, 0.1):
    for n, rs in group([r for r in sc if r["loss"] == loss], "n").items():
        d = S.setdefault(f"scaling_{loss}", {})[str(n)] = {}
        for p in ("es2", "es1", "pw", "full"):
            for m in ("airtime", "wall", "energy", "converged"):
                vals = [r[p][m] for r in rs if r[p].get(m) is not None]
                if vals:
                    d[f"{p}_{m}"] = round(float(np.mean(vals)), 3)
        d["oracle_airtime"] = round(float(np.mean([r["oracle"]["airtime"] for r in rs])), 2)
        d["dmax"] = round(float(np.mean([r["dmax"] for r in rs])), 1)
        d["airtime_reduction_vs_pw"] = round(1 - d["es2_airtime"] / d["pw_airtime"], 3)
        d["wall_reduction_vs_pw"] = round(1 - d["es2_wall"] / d["pw_wall"], 3)
        if "single_max" in rs[0]:
            d["h1_ratio"] = round(float(np.mean([r["es2"]["phase1_wall"] / r["single_max"] for r in rs])), 3)
            d["h1_ratio_max"] = round(float(np.max([r["es2"]["phase1_wall"] / r["single_max"] for r in rs])), 3)

# ---------------------------------------------------------- heterogeneity (RQ2)
he = R("hetero")
g = group(he, "lag"); xs = list(g)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax = axs[0]
labels = [f"{int(x*100)}%" for x in xs]
pos = np.arange(len(xs))
for off, f, col, lab in ((-0.2, lambda r: r["es2_sleep"]["phase1_wall"] / 3600, C_ES, "EchoSync broadcast (phase 1)"),
                         (0.2, lambda r: r["single_max"] / 3600, C_PW, "Largest single pairwise stream")):
    m = [ci([f(r) for r in g[x]]) for x in xs]
    ax.bar(pos + off, [a for a, _ in m], 0.38, yerr=[b for _, b in m], color=col, label=lab,
           error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
ax.set_xticks(pos); ax.set_xticklabels(labels)
ax.set_xlabel("records missing at the lagging node"); ax.set_ylabel("time (h)")
ax.set_title("(a) Broadcast time is set by the largest gap"); ax.legend(fontsize=7)
ax = axs[1]


def fast_rx_energy(res_key):
    return lambda r: r[res_key]["energy"]


line(ax, list(range(len(xs))), {i: g[x] for i, x in enumerate(xs)}, fast_rx_energy("es2_sleep"), C_ES, "with early sleep")
line(ax, list(range(len(xs))), {i: g[x] for i, x in enumerate(xs)}, fast_rx_energy("es2_nosleep"), C_ALT, "always listening", marker="D")
ax.set_xticks(pos); ax.set_xticklabels(labels)
ax.set_xlabel("records missing at the lagging node"); ax.set_ylabel("network radio energy (J)")
ax.set_title("(b) Early sleep protects fast nodes"); ax.legend(fontsize=7)
save(fig, "fig_hetero")
S["hetero"] = {str(x): dict(
    dmax=round(float(np.mean([r["dmax"] for r in g[x]])), 1),
    phase1_h=round(float(np.mean([r["es2_sleep"]["phase1_wall"] for r in g[x]])) / 3600, 3),
    single_h=round(float(np.mean([r["single_max"] for r in g[x]])) / 3600, 3),
    h1_ratio=round(float(np.mean([r["es2_sleep"]["phase1_wall"] / r["single_max"] for r in g[x]])), 3),
    h1_ratio_max=round(float(np.max([r["es2_sleep"]["phase1_wall"] / r["single_max"] for r in g[x]])), 3),
    e_sleep=round(float(np.mean([r["es2_sleep"]["energy"] for r in g[x]])), 2),
    e_nosleep=round(float(np.mean([r["es2_nosleep"]["energy"] for r in g[x]])), 2),
    es_wall_h=round(float(np.mean([r["es2_sleep"]["wall"] for r in g[x]])) / 3600, 3),
    pw_wall_h=round(float(np.mean([r["pw"]["wall"] for r in g[x]])) / 3600, 3),
    es_air=round(float(np.mean([r["es2_sleep"]["airtime"] for r in g[x]])), 1),
    pw_air=round(float(np.mean([r["pw"]["airtime"] for r in g[x]])), 1),
) for x in xs}

# ---------------------------------------------------------- loss (RQ3, H3)
lo = R("loss")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, burst, title in ((axs[0], 1, "(a) Independent (Bernoulli) loss"), (axs[1], 4, "(b) Bursty loss, mean burst 4 frames")):
    g = group([r for r in lo if r["burst"] == burst], "loss"); xs = list(g)
    xp = [100 * x for x in xs]
    line(ax, xp, dict(zip(xp, g.values())), lambda r: r["es2"]["airtime"], C_ES, "EchoSync + prefix repair")
    line(ax, xp, dict(zip(xp, g.values())), lambda r: r["es2_norepair"]["airtime"], C_ALT, "EchoSync, no repair", marker="D", ls="--")
    line(ax, xp, dict(zip(xp, g.values())), lambda r: r["pw"]["airtime"], C_PW, "Pairwise RIBLT", marker="s")
    ax.set_yscale("log"); ax.set_xlabel("frame loss rate (%)"); ax.set_title(title)
    ax.set_yticks([50, 100, 200, 400, 800]); ax.set_yticklabels(["50", "100", "200", "400", "800"]); ax.minorticks_off()
    for x in xs:
        rs = g[x]
        S.setdefault("loss", {})[f"{burst}_{x}"] = dict(
            es_air=round(float(np.mean([r["es2"]["airtime"] for r in rs])), 1),
            es_air_p95=round(float(np.percentile([r["es2"]["airtime"] for r in rs], 95)), 1),
            nr_air=round(float(np.mean([r["es2_norepair"]["airtime"] for r in rs])), 1),
            nr_air_p95=round(float(np.percentile([r["es2_norepair"]["airtime"] for r in rs], 95)), 1),
            pw_air=round(float(np.mean([r["pw"]["airtime"] for r in rs])), 1),
            es_wall_h=round(float(np.mean([r["es2"]["wall"] for r in rs])) / 3600, 2),
            nr_wall_h=round(float(np.mean([r["es2_norepair"]["wall"] for r in rs])) / 3600, 2),
            pw_wall_h=round(float(np.mean([r["pw"]["wall"] for r in rs])) / 3600, 2),
            es_conv=float(np.mean([r["es2"]["converged"] == 1.0 and not r["es2"]["timeout"] for r in rs])),
            nr_conv=float(np.mean([r["es2_norepair"]["converged"] == 1.0 and not r["es2_norepair"]["timeout"] for r in rs])),
            pw_conv=float(np.mean([r["pw"]["converged"] == 1.0 and not r["pw"]["timeout"] for r in rs])),
            es_rec=round(float(np.mean([r["es2"]["recoveries"] for r in rs])), 2),
            nr_rec=round(float(np.mean([r["es2_norepair"]["recoveries"] for r in rs])), 2),
            es_sym_ratio=round(float(np.mean([r["es2"]["symbols"] / r["dmax"] for r in rs])), 2),
            nr_sym_ratio=round(float(np.mean([r["es2_norepair"]["symbols"] / r["dmax"] for r in rs])), 2),
            n=len(rs))
axs[0].set_ylabel("total network airtime (s)"); axs[0].legend(fontsize=7)
save(fig, "fig_loss")

# ---------------------------------------------------------- spreading factor
sf = R("sf")
g = group(sf, lambda r: (r["sf"], r["loss"]))
S["sf"] = {f"{k[0]}_{k[1]}": dict(
    es_air=round(float(np.mean([r["es2"]["airtime"] for r in v])), 1),
    pw_air=round(float(np.mean([r["pw"]["airtime"] for r in v])), 1),
    es_wall_h=round(float(np.mean([r["es2"]["wall"] for r in v])) / 3600, 2),
    pw_wall_h=round(float(np.mean([r["pw"]["wall"] for r in v])) / 3600, 2),
    es_conv=float(np.mean([r["es2"]["converged"] for r in v])),
) for k, v in g.items()}
fig, ax = plt.subplots(figsize=(3.4, 2.4))
sfs = [7, 9, 12]; pos = np.arange(3)
for off, key, col, lab in ((-0.2, "es2", C_ES, "EchoSync"), (0.2, "pw", C_PW, "Pairwise RIBLT")):
    m = [ci([r[key]["wall"] / 3600 for r in g[(s, 0.0)]]) for s in sfs]
    ax.bar(pos + off, [a for a, _ in m], 0.38, yerr=[b for _, b in m], color=col, label=lab,
           error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
ax.set_yscale("log"); ax.set_xticks(pos); ax.set_xticklabels([f"SF{s}" for s in sfs])
ax.set_ylabel("time to convergence (h)"); ax.legend(fontsize=7)
save(fig, "fig_sf")

# ---------------------------------------------------------- misc tables
S["asym"] = {str(k): dict(
    es_air=round(float(np.mean([r["es2"]["airtime"] for r in v])), 1),
    pw_air=round(float(np.mean([r["pw"]["airtime"] for r in v])), 1),
    es_wall_h=round(float(np.mean([r["es2"]["wall"] for r in v])) / 3600, 2),
    pw_wall_h=round(float(np.mean([r["pw"]["wall"] for r in v])) / 3600, 2),
    es_conv=float(np.mean([r["es2"]["converged"] for r in v])),
    pw_conv=float(np.mean([r["pw"]["converged"] for r in v])),
) for k, v in group(R("asym"), "far").items()}
S["batch"] = {f"{k[0]}_{k[1]}": dict(
    air=round(float(np.mean([r["es2"]["airtime"] for r in v])), 1),
    wall_h=round(float(np.mean([r["es2"]["wall"] for r in v])) / 3600, 2),
    sym=round(float(np.mean([r["es2"]["symbols"] for r in v])), 1),
) for k, v in group(R("batch"), lambda r: (r["batch"], r["loss"])).items()}
S["crdt"] = {str(k): dict(
    runs=len(v), identical=sum(r["identical"] for r in v), matches_ref=sum(r["matches_ref"] for r in v),
    mean_ops=round(float(np.mean([r["n_ops"] for r in v])), 1),
    mean_conflicting_keys=round(float(np.mean([r["conflicting_keys"] for r in v])), 1),
) for k, v in group(R("crdt"), "loss").items()}

json.dump(S, open("summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
