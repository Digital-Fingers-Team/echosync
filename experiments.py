"""Runs every experiment in the paper and writes results/*.json.
usage: python experiments.py [name ...]"""
import json
import os
import random
import sys
import time
from multiprocessing import Pool

from riblt import Encoder, Decoder, fingerprint
from sim import (Cfg, Phy, MAX_PAYLOAD, Net, echosync, pairwise, full_broadcast, oracle,
                 workload, elect_leader, stream_session, beacons)

OUT = "results"
T_MAX = 120 * 86400.0


def cfg_for(sf=9, **kw):
    k = {7: 8, 9: 4, 12: 1}[sf]
    return Cfg(phy=Phy(sf=sf, max_payload=MAX_PAYLOAD[sf]), sym_per_frame=k, t_max=T_MAX, **kw)


def largest_single_stream(sets, cfg, seed):
    """Wall time of the slowest one-receiver session L -> i (same mechanism)."""
    L = elect_leader(sets)
    worst = 0.0
    for i in range(len(sets)):
        if i == L:
            continue
        net = Net(len(sets), cfg, random.Random(seed * 1000 + i))
        ss = [set(s) for s in sets]
        stream_session(net, L, [i], ss, {})
        worst = max(worst, net.t)
    return worst


def d_max(sets):
    L = elect_leader(sets)
    return max(len(sets[L] ^ s) for s in sets)


# ---------------------------------------------------------------- tasks --

def t_overhead(args):
    d, n_rx, seed = args
    """One sender stream; each of n_rx receivers lacks its own random d items of S.
    The broadcast must run until the slowest receiver decodes: m = max_i m_i."""
    rng = random.Random(seed)
    S = [rng.getrandbits(128) | 1 for _ in range(max(4 * d, 200))]
    fpS = fingerprint(S)
    worst = 0
    for _ in range(n_rx):
        miss = set(rng.sample(S, d))
        R = set(S) - miss
        e = Encoder(miss)                 # = symbols(S) - symbols(R), by linearity
        dec = Decoder(fpS ^ fingerprint(R), d)
        j = 0
        while not dec.done():
            dec.add(j, e.symbol(j)); j += 1
        worst = max(worst, j)
    return dict(d=d, n_rx=n_rx, seed=seed, m=worst, ratio=worst / d)


def t_scaling(args):
    n, loss, seed = args
    rng = random.Random(seed)
    sets = workload(rng, n)
    cfg = cfg_for(loss=loss)
    r = dict(n=n, loss=loss, seed=seed, dmax=d_max(sets))
    r["es2"] = echosync(sets, cfg, random.Random(seed + 1))
    r["es1"] = echosync(sets, cfg, random.Random(seed + 1), two_phase=False)
    r["pw"] = pairwise(sets, cfg, random.Random(seed + 1))
    r["full"] = full_broadcast(sets, cfg, random.Random(seed + 1))
    r["oracle"] = oracle(sets, cfg)
    if n <= 20:
        r["single_max"] = largest_single_stream(sets, cfg, seed)
    return r


def t_hetero(args):
    lag, seed = args
    rng = random.Random(seed)
    sets = workload(rng, 10, laggards={9: lag})
    r = dict(lag=lag, seed=seed, dmax=d_max(sets))
    for sleep in (True, False):
        cfg = cfg_for(early_sleep=sleep)
        res = echosync(sets, cfg, random.Random(seed + 1))
        r["es2_sleep" if sleep else "es2_nosleep"] = res
    r["pw"] = pairwise(sets, cfg_for(), random.Random(seed + 1))
    r["single_max"] = largest_single_stream(sets, cfg_for(), seed)
    return r


def t_loss(args):
    loss, burst, seed = args
    rng = random.Random(seed)
    sets = workload(rng, 10)
    r = dict(loss=loss, burst=burst, seed=seed, dmax=d_max(sets))
    r["es2"] = echosync(sets, cfg_for(loss=loss, burst=burst), random.Random(seed + 1))
    r["es2_norepair"] = echosync(sets, cfg_for(loss=loss, burst=burst, repair=False),
                                 random.Random(seed + 1))
    r["pw"] = pairwise(sets, cfg_for(loss=loss, burst=burst), random.Random(seed + 1))
    return r


def t_asym(args):
    far, seed = args
    rng = random.Random(seed)
    sets = workload(rng, 10)
    nl = {i: 0.05 for i in range(10)}
    for i in range(far):
        nl[9 - i] = 0.30
    cfg = cfg_for(node_loss=nl)
    r = dict(far=far, seed=seed)
    r["es2"] = echosync(sets, cfg, random.Random(seed + 1))
    r["pw"] = pairwise(sets, cfg, random.Random(seed + 1))
    return r


def t_sf(args):
    sf, loss, seed = args
    rng = random.Random(seed)
    sets = workload(rng, 10)
    cfg = cfg_for(sf=sf, loss=loss)
    return dict(sf=sf, loss=loss, seed=seed,
                es2=echosync(sets, cfg, random.Random(seed + 1)),
                pw=pairwise(sets, cfg, random.Random(seed + 1)))


def t_batch(args):
    b, loss, seed = args
    rng = random.Random(seed)
    sets = workload(rng, 10)
    return dict(batch=b, loss=loss, seed=seed,
                es2=echosync(sets, cfg_for(batch=b, loss=loss), random.Random(seed + 1)))


def t_crdt(args):
    loss, seed = args
    """LWW-register map: each item is an operation (key, value, lamport, node, nonce)."""
    rng = random.Random(seed)
    n, keys = 10, 20
    ops = []
    for node in range(n):
        for _ in range(rng.randint(5, 30)):
            k, v, lam = rng.randrange(keys), rng.getrandbits(32), rng.randrange(1, 60)
            ops.append((node, (k << 112) | (v << 80) | (lam << 48) | (node << 32) | rng.getrandbits(32) | 1))
    sets = [set() for _ in range(n)]
    for node, op in ops:                 # each op is known to its author ...
        sets[node].add(op)
        for o in range(n):               # ... and gossiped to some others already
            if o != node and rng.random() < 0.5:
                sets[o].add(op)
    from sim import Net as _N  # noqa
    import sim
    res_sets = {}
    orig = sim.run_result

    def capture(net, s, stats, extra=None):
        res_sets["final"] = s
        return orig(net, s, stats, extra)
    sim.run_result = capture
    try:
        r = echosync(sets, cfg_for(loss=loss), random.Random(seed + 1))
    finally:
        sim.run_result = orig

    def state(s):
        st = {}
        for op in s:
            k = op >> 112
            ts = ((op >> 48) & 0xFFFFFFFF, (op >> 32) & 0xFFFF, op)   # (lamport, node, tiebreak)
            if k not in st or ts > st[k][0]:
                st[k] = (ts, (op >> 80) & 0xFFFFFFFF)
        return {k: v for k, (_, v) in st.items()}
    ref = state({op for _, op in ops})
    states = [state(s) for s in res_sets["final"]]
    conflicts = sum(1 for k in range(keys) if len({op for _, op in ops if op >> 112 == k}) > 1)
    return dict(loss=loss, seed=seed, n_ops=len(ops), conflicting_keys=conflicts,
                identical=all(s == states[0] for s in states), matches_ref=states[0] == ref,
                converged=r["converged"])


EXPS = {
    "overhead": (t_overhead, [(d, nrx, s) for d in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
                              for nrx in (1, 10) for s in range(100 if d <= 100 else 30)]),
    "scaling": (t_scaling, [(n, loss, s) for n in (3, 5, 10, 20, 50, 100) for loss in (0.0, 0.1)
                            for s in range(20 if n < 50 else 8)]),
    "hetero": (t_hetero, [(lag, s) for lag in (0.03, 0.1, 0.25, 0.5, 1.0) for s in range(20)]),
    "loss": (t_loss, [(p, b, s) for p in (0.0, 0.05, 0.1, 0.2, 0.3) for b in (1, 4)
                      for s in range(30)]),
    "asym": (t_asym, [(f, s) for f in (0, 1, 3) for s in range(20)]),
    "sf": (t_sf, [(sf, loss, s) for sf in (7, 9, 12) for loss in (0.0, 0.1) for s in range(15)]),
    "batch": (t_batch, [(b, loss, s) for b in (1, 2, 4, 8) for loss in (0.0, 0.2) for s in range(20)]),
    "crdt": (t_crdt, [(loss, s) for loss in (0.0, 0.2, 0.3) for s in range(100)]),
}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(EXPS)
    with Pool(2) as pool:
        for name in names:
            fn, tasks = EXPS[name]
            t = time.time()
            res = pool.map(fn, tasks, chunksize=1)
            with open(f"{OUT}/{name}.json", "w") as f:
                json.dump(res, f)
            print(f"{name}: {len(tasks)} runs in {time.time() - t:.0f}s", flush=True)
