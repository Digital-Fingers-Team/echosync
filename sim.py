"""EchoSync discrete-event simulator: LoRa PHY airtime, duty cycle, lossy links,
and four synchronization protocols (EchoSync-2P, EchoSync-1P, pairwise RIBLT
star, full-state broadcast) plus an oracle lower bound."""
import math
import random
from dataclasses import dataclass, field

from riblt import Encoder, Decoder, fingerprint, ITEM_BYTES

# ---------------------------------------------------------------- PHY ----

@dataclass
class Phy:
    sf: int = 9
    bw: float = 125e3
    cr: int = 1             # coding rate 4/(4+cr)
    preamble: int = 8
    crc: int = 1
    implicit: int = 0
    max_payload: int = 115  # LoRaWAN RP002 EU868 limit for this SF (DR3 = SF9)
    dc: float = 0.01        # regulatory duty cycle (ETSI EN 300 220, g1 sub-band)

    def toa(self, pl):
        """Time on air in seconds (Semtech AN1200.13)."""
        tsym = (2 ** self.sf) / self.bw
        de = 1 if tsym > 0.016 else 0
        tpre = (self.preamble + 4.25) * tsym
        num = 8 * pl - 4 * self.sf + 28 + 16 * self.crc - 20 * self.implicit
        npay = 8 + max(math.ceil(num / (4 * (self.sf - 2 * de))) * (self.cr + 4), 0)
        return tpre + npay * tsym


MAX_PAYLOAD = {7: 222, 8: 222, 9: 115, 10: 51, 11: 51, 12: 51}

HDR = 8            # type 1, sender 2, session 1, first symbol index 3, n 1
SYM = ITEM_BYTES + 8 + 2   # sum 16 B, checksum 8 B, count 2 B (mod 2^16)
BEACON = HDR + 4 + 8       # + set size 4 B + state fingerprint 8 B
REQ = HDR                  # "more symbols" request (busy tone, content-free)
ACK = HDR + 1              # + 8-bit frame bitmap
NACK_FRAMES = 64           # request carries a bitmap of lost frames among the first 64
NACK_BYTES = NACK_FRAMES // 8

# radio current draw, SX1276/77/78 datasheet (Semtech rev. 7), Vdd = 3.3 V
I_TX = 0.087   # +17 dBm, PA_BOOST
I_RX = 0.0108  # LnaBoost off, 125 kHz
VDD = 3.3


@dataclass
class Cfg:
    phy: Phy = field(default_factory=Phy)
    sym_per_frame: int = 4
    batch: int = 2               # frames per batch before a request window
    loss: float = 0.0            # mean frame loss rate per link
    burst: float = 1.0           # mean burst length (1 = Bernoulli)
    node_loss: dict = None       # optional per-node loss override (asymmetric links)
    early_sleep: bool = True
    overhear: bool = True
    size_hint: bool = True
    repair: bool = True          # NACK-driven retransmission of lost prefix frames
    slots: int = 4               # request-window slots
    t_rec: float = 30.0         # recovery timeout after a missed request window (s)
    t_to: float = 5.0            # unicast feedback timeout (s)
    t_max: float = 14 * 86400.0  # give up after two simulated weeks

    @property
    def items_per_frame(self):
        return (self.phy.max_payload - HDR) // ITEM_BYTES


# ------------------------------------------------------------ channel ----

class Net:
    def __init__(self, n, cfg, rng):
        self.n, self.cfg, self.rng = n, cfg, rng
        self.t = 0.0
        self.next_ok = [0.0] * n
        self.tx = [0.0] * n
        self.rx = [0.0] * n
        self.frames = 0
        self.ge = {}            # Gilbert-Elliott state per unordered link
        self.recoveries = 0

    def _p(self, a, b):
        nl = self.cfg.node_loss or {}
        return max(nl.get(a, self.cfg.loss), nl.get(b, self.cfg.loss))

    def lost(self, a, b):
        p = self._p(a, b)
        if p <= 0:
            return False
        L = self.cfg.burst
        if L <= 1:
            return self.rng.random() < p
        key = (a, b) if a < b else (b, a)
        bad = self.ge.get(key)
        if bad is None:
            bad = self.rng.random() < p
        # Gilbert model: stationary P(bad) = p, mean bad-run length L
        if bad:
            bad = self.rng.random() >= 1.0 / L
        else:
            bad = self.rng.random() < p / (L * (1 - p))
        self.ge[key] = bad
        return bad

    def send(self, src, pl, listeners):
        phy = self.cfg.phy
        a = phy.toa(pl)
        start = max(self.t, self.next_ok[src])
        self.t = start + a
        self.next_ok[src] = self.t + a * (1 / phy.dc - 1)
        self.tx[src] += a
        self.frames += 1
        got = []
        for l in listeners:
            self.rx[l] += a
            if not self.lost(src, l):
                got.append(l)
        return got

    def request_window(self, requesters, dst, nacks=None):
        """Undecoded receivers each send a request in one of `slots` random slots.
        Any activity (Channel Activity Detection) tells the sender to continue, so
        collisions never stop the stream; a request alone in its slot is also
        decoded, and its NACK bitmap (lost prefix frames) is returned.
        Returns (heard, list of decoded NACK sets)."""
        phy = self.cfg.phy
        slots = self.cfg.slots if nacks is not None else 1
        pl = REQ + (NACK_BYTES if nacks is not None else 0)
        a = phy.toa(pl)
        heard, decoded, use = False, [], {}
        for r in requesters:
            if self.next_ok[r] > self.t:      # duty cycle forbids it now
                continue
            self.tx[r] += a
            self.next_ok[r] = self.t + a / phy.dc
            use.setdefault(self.rng.randrange(slots), []).append(r)
        for s, rs in use.items():
            ok = [r for r in rs if not self.lost(r, dst)]
            if ok:
                heard = True
            if len(rs) == 1 and ok and nacks is not None:
                decoded.append(nacks[rs[0]])
        w = slots * a * 1.5
        self.rx[dst] += w
        self.t += w
        return heard, decoded

    def wait(self, dt):
        self.t += dt

    def energy(self):
        return VDD * (I_TX * sum(self.tx) + I_RX * sum(self.rx))


# ---------------------------------------------------------- building blocks

def beacons(net, nodes):
    for i in nodes:
        net.send(i, BEACON, [j for j in nodes if j != i])


def stream_session(net, src, receivers, sets, stats):
    """src broadcasts RIBLT coded symbols of its set; each receiver decodes its own
    difference. Returns number of coded symbols transmitted."""
    cfg = net.cfg
    K = cfg.sym_per_frame
    S = sets[src]
    fpS = fingerprint(S)
    dec, enc, pending = {}, {}, set()
    for i in receivers:
        plus, minus = S - sets[i], sets[i] - S
        if not plus and not minus:
            continue                     # beacon fingerprints already match
        dec[i] = Decoder(fpS ^ fingerprint(sets[i]), len(S) - len(sets[i]))
        enc[i] = Encoder(plus, minus)
        pending.add(i)
    if not pending:
        return 0
    nf = 0                       # frames of new symbols sent so far
    batch = cfg.batch
    if cfg.size_hint:   # first batch sized from the beacons' set-size gap
        gap = max(abs(len(S) - len(sets[i])) for i in pending)
        batch = max(batch, math.ceil(1.35 * gap / K))
    listening = set(pending)
    got_frames = {i: set() for i in pending}
    repair = []
    while pending:
        for f in repair + list(range(nf, nf + batch)):
            got = net.send(src, HDR + K * SYM, sorted(listening))
            for i in got:
                if i in pending:
                    got_frames[i].add(f)
                    d = dec[i]
                    for q in range(f * K, f * K + K):
                        d.add(q, enc[i].symbol(q))
            done = {i for i in pending if dec[i].done()}
            pending -= done
            if cfg.early_sleep:
                listening -= done
        stats["repairs"] = stats.get("repairs", 0) + len(repair)
        nf += batch
        batch = cfg.batch
        if net.t > cfg.t_max:
            stats["timeout"] = True
            break
        if not pending:
            net.request_window([], src)   # sender hears silence and stops
            break
        nacks = None
        if cfg.repair:
            nacks = {i: {f for f in range(min(nf, NACK_FRAMES)) if f not in got_frames[i]}
                     for i in pending}
        heard, dec_nacks = net.request_window(sorted(pending), src, nacks)
        while not heard:
            net.recoveries += 1           # sender stopped; laggards retry later
            net.wait(cfg.t_rec)
            if net.t > cfg.t_max:
                stats["timeout"] = True
                return nf * K
            heard, dec_nacks = net.request_window(sorted(pending), src, nacks)
        repair = sorted(set().union(*dec_nacks)) if dec_nacks else []
    j = nf * K
    for i, d in dec.items():
        if d.done():
            sets[i] |= set(d.recovered(+1))
    stats["symbols"] = stats.get("symbols", 0) + j
    return j


def reliable_items(net, src, dst, items, overhearers=()):
    """Packed item frames with per-burst ACK + selective retransmission."""
    cfg = net.cfg
    ipf = cfg.items_per_frame
    items = sorted(items)
    frames = [items[k:k + ipf] for k in range(0, len(items), ipf)] or [[]]
    out = list(range(len(frames)))
    heard_by = {o: set() for o in overhearers}
    while out:
        if net.t > cfg.t_max:
            return heard_by
        nxt = []
        for f in out:
            got = net.send(src, HDR + ITEM_BYTES * len(frames[f]), [dst, *overhearers])
            for o in overhearers:
                if o in got:
                    heard_by[o].update(frames[f])
            if dst not in got:
                nxt.append(f)
        if src in net.send(dst, ACK, [src]):
            out = nxt
        else:
            net.wait(cfg.t_to)            # ACK lost: resend the whole burst
    return heard_by


def elect_leader(sets):
    # deterministic from beacons: largest set, ties broken by lowest id
    return min(range(len(sets)), key=lambda i: (-len(sets[i]), i))


def run_result(net, sets, stats, extra=None):
    U = set().union(*sets) if stats.get("union") is None else stats["union"]
    conv = sum(1 for s in sets if s == U) / len(sets)
    r = dict(airtime=sum(net.tx), wall=net.t, energy=net.energy(), frames=net.frames,
             converged=conv, recoveries=net.recoveries, symbols=stats.get("symbols", 0),
             timeout=stats.get("timeout", False), leader_air=max(net.tx))
    if extra:
        r.update(extra)
    return r


# ------------------------------------------------------------ protocols ----

def echosync(sets, cfg, rng, two_phase=True):
    sets = [set(s) for s in sets]
    stats = {"union": set().union(*sets)}
    n = len(sets)
    net = Net(n, cfg, rng)
    nodes = list(range(n))
    beacons(net, nodes)
    L = elect_leader(sets)
    rx = [i for i in nodes if i != L]
    S0 = set(sets[L])
    t0 = net.t
    stream_session(net, L, rx, sets, stats)
    t_phase1 = net.t - t0
    if two_phase:
        # Phase U: receivers upload items the leader lacks, in id order (TDMA from
        # the beacon roster); later uploaders skip what they overheard.
        known = {i: set(S0) for i in rx}
        for i in rx:
            todo = sets[i] - known[i]
            if not todo:
                net.wait(net.cfg.phy.toa(HDR))       # idle slot
                continue
            others = [o for o in rx if o != i] if cfg.overhear else []
            heard = reliable_items(net, i, L, todo, others)
            sets[L] |= todo
            for o, h in heard.items():
                sets[o] |= h
                known[o] |= h
        # Phase 2: POLL with the union fingerprint; anyone still behind requests.
        U = sets[L]
        behind = [i for i in rx if sets[i] != U]
        got = net.send(L, BEACON, rx)
        if behind:
            asked = [i for i in behind if i in got]
            if not net.request_window(asked, L)[0]:
                while not net.request_window(behind, L)[0]:
                    net.recoveries += 1
                    net.wait(cfg.t_rec)
                    if net.t > cfg.t_max:
                        stats["timeout"] = True
                        break
            stream_session(net, L, behind, sets, stats)
    beacons(net, nodes)   # verification: all fingerprints must match
    return run_result(net, sets, stats, dict(phase1_wall=t_phase1, leader=L))


def pairwise(sets, cfg, rng):
    """Star of sequential bidirectional RIBLT sessions with the same leader;
    two passes so early nodes also receive later nodes' items."""
    sets = [set(s) for s in sets]
    stats = {"union": set().union(*sets)}
    n = len(sets)
    net = Net(n, cfg, rng)
    nodes = list(range(n))
    beacons(net, nodes)
    L = elect_leader(sets)
    rx = [i for i in nodes if i != L]
    longest = 0.0
    for i in rx + rx[:-1]:
        t0 = net.t
        stream_session_unicast(net, i, L, sets, stats)
        longest = max(longest, net.t - t0)
        if net.t > cfg.t_max:
            stats["timeout"] = True
            break
    beacons(net, nodes)
    return run_result(net, sets, stats, dict(longest_session=longest))


def stream_session_unicast(net, i, L, sets, stats):
    """Node i streams symbols to L; L decodes, replies MORE until done, then sends
    the items i lacks (reliably). L learns i's extra items from the decode."""
    cfg = net.cfg
    K = cfg.sym_per_frame
    plus, minus = sets[i] - sets[L], sets[L] - sets[i]
    if not plus and not minus:
        return
    d = Decoder(fingerprint(sets[i]) ^ fingerprint(sets[L]), len(sets[i]) - len(sets[L]))
    enc = Encoder(plus, minus)
    nf = 0
    batch = cfg.batch
    if cfg.size_hint:
        batch = max(batch, math.ceil(1.35 * abs(len(sets[i]) - len(sets[L])) / K))
    got_frames, repair = set(), []
    while True:
        for f in repair + list(range(nf, nf + batch)):
            if L in net.send(i, HDR + K * SYM, [L]):
                got_frames.add(f)
                for q in range(f * K, f * K + K):
                    d.add(q, enc.symbol(q))
        stats["repairs"] = stats.get("repairs", 0) + len(repair)
        nf += batch
        batch = cfg.batch
        if d.done() or net.t > cfg.t_max:
            break
        # unicast MORE carries the same NACK bitmap (always readable: no contention)
        if i in net.send(L, REQ + (NACK_BYTES if cfg.repair else 0), [i]):
            repair = sorted(f for f in range(min(nf, NACK_FRAMES)) if f not in got_frames) \
                if cfg.repair else []
        else:
            repair = []
            net.wait(cfg.t_to)
    j = nf * K
    stats["symbols"] = stats.get("symbols", 0) + j
    sets[L] |= set(d.recovered(+1))
    reliable_items(net, L, i, set(d.recovered(-1)))   # also acts as DONE
    sets[i] |= set(d.recovered(-1))


def full_broadcast(sets, cfg, rng):
    """Every node broadcasts its full set (lossless lower bound for flooding)."""
    sets = [set(s) for s in sets]
    n = len(sets)
    net = Net(n, cfg, rng)
    ipf = cfg.items_per_frame
    queues = [[sorted(s)[k:k + ipf] for k in range(0, len(s), ipf)] for s in sets]
    while any(queues):
        for i in range(n):              # round-robin to share duty budgets
            if queues[i]:
                fr = queues[i].pop(0)
                net.send(i, HDR + ITEM_BYTES * len(fr), [])
    U = set().union(*sets)
    return dict(airtime=sum(net.tx), wall=net.t, energy=None, converged=1.0)


def oracle(sets, cfg):
    """Information lower bound: every item missing somewhere is sent once."""
    U = set().union(*sets)
    common = set.intersection(*map(set, sets))
    k = len(U - common)
    ipf = cfg.items_per_frame
    full, rem = divmod(k, ipf)
    a = full * cfg.phy.toa(HDR + ITEM_BYTES * ipf) + (cfg.phy.toa(HDR + ITEM_BYTES * rem) if rem else 0)
    return dict(airtime=a)


# ------------------------------------------------------------- workloads --

def workload(rng, n, base=1000, miss=0.03, fresh=(5, 20), laggards=None):
    """Shared history of `base` records; each node misses each with prob `miss`
    and holds 5-20 fresh records of its own. laggards: {node: missing fraction}."""
    B = [rng.getrandbits(128) | 1 for _ in range(base)]
    sets = []
    for i in range(n):
        m = (laggards or {}).get(i, miss)
        s = {x for x in B if rng.random() >= m}
        s |= {rng.getrandbits(128) | 1 for _ in range(rng.randint(*fresh))}
        sets.append(s)
    return sets
