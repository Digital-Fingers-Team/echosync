"""Rateless Invertible Bloom Lookup Table (RIBLT) codec.

Follows Yang, Gilad & Alizadeh, "Practical Rateless Set Reconciliation"
(SIGCOMM 2024) and their reference implementation (github.com/yangl1996/riblt):
each source item maps to coded-symbol index 0 and then to a pseudo-random,
increasingly sparse sequence of indices seeded by the item's 64-bit hash.

A coded symbol is (sum, checksum, count):
  sum      XOR of the 16-byte items mapped to it
  checksum XOR of their 64-bit hashes
  count    signed number of items (+1 remote, -1 local after subtraction)
"""
import hashlib
import heapq
import math
from bisect import bisect_left, bisect_right

MASK64 = (1 << 64) - 1
ITEM_BYTES = 16
MUL = 0xDA942042E4DD58B5


def h64(x):
    return int.from_bytes(
        hashlib.blake2b(x.to_bytes(ITEM_BYTES, "little"), digest_size=8).digest(), "little")


class _Info:
    __slots__ = ("h", "idx", "prng", "last")


_info = {}


def info(x):
    it = _info.get(x)
    if it is None:
        it = _Info()
        it.h = h64(x)
        it.idx = [0]
        it.prng = it.h
        it.last = 0
        _info[x] = it
    return it


def _extend(it, m):
    """Grow the item's index list until it holds an index >= m."""
    last, prng, idx = it.last, it.prng, it.idx
    while last < m:
        prng = (prng * MUL) & MASK64
        last += max(1, math.ceil((last + 1.5) * ((1 << 32) / math.sqrt(prng + 1) - 1)))
        idx.append(last)
    it.last, it.prng = last, prng


def indices_in(x, lo, hi):
    """Coded-symbol indices in [lo, hi) that item x maps to."""
    it = info(x)
    if it.last < hi:
        _extend(it, hi)
    return it.idx[bisect_left(it.idx, lo):bisect_left(it.idx, hi)]


def fingerprint(items):
    fp = 0
    for x in items:
        fp ^= info(x).h
    return fp


class Encoder:
    """Coded symbols of a signed multiset, produced on demand (prefix of a rateless stream)."""

    def __init__(self, plus=(), minus=()):
        self.items = [(x, 1) for x in plus] + [(x, -1) for x in minus]
        self.S, self.C, self.N = [], [], []

    def _ensure(self, m):
        lo = len(self.S)
        if m <= lo:
            return
        m = max(m, 2 * lo, 16)
        S, C, N = self.S, self.C, self.N
        S.extend([0] * (m - lo)); C.extend([0] * (m - lo)); N.extend([0] * (m - lo))
        for x, sg in self.items:
            hx = info(x).h
            for i in indices_in(x, lo, m):
                S[i] ^= x; C[i] ^= hx; N[i] += sg

    def symbol(self, j):
        self._ensure(j + 1)
        return self.S[j], self.C[j], self.N[j]


def subtract(a, b):
    return a[0] ^ b[0], a[1] ^ b[1], a[2] - b[2]


class Decoder:
    """Streaming peeling decoder for the difference D = remote - local.

    Symbols may arrive with gaps (lost frames); every received symbol carries its
    index. Completion is detected by fingerprint verification: the XOR of hashes
    of all recovered items must equal fp(remote) ^ fp(local), and the signed
    count must equal |remote| - |local| (both announced in the sender's beacon).
    """

    def __init__(self, target_fp, target_cnt):
        self.cells = {}
        self.heap = []          # (next index, seq, item, sign) for recovered items
        self.seq = 0
        self.decoded = {}
        self.fp = 0
        self.cnt = 0
        self.target_fp = target_fp
        self.target_cnt = target_cnt
        self.maxj = -1
        self.received = 0

    def done(self):
        return self.fp == self.target_fp and self.cnt == self.target_cnt

    def _push(self, x, sg, after):
        it = info(x)
        if it.last <= after:
            _extend(it, after + 1)
        k = bisect_right(it.idx, after)
        heapq.heappush(self.heap, (it.idx[k], self.seq, x, sg))
        self.seq += 1

    def add(self, j, sym):
        if j in self.cells:
            return                      # duplicate (retransmitted) symbol
        s, c, n = sym
        self.received += 1
        if j < self.maxj:
            # late symbol from a repair retransmission: strip every recovered
            # item that maps to j, then peel
            for x, sg in self.decoded.items():
                it = info(x)
                k = bisect_left(it.idx, j)
                if k < len(it.idx) and it.idx[k] == j:
                    s ^= x; c ^= it.h; n -= sg
            self.cells[j] = [s, c, n]
            self._peel([j])
            return
        self.maxj = j
        hp = self.heap
        while hp and hp[0][0] <= j:
            nxt, _, x, sg = heapq.heappop(hp)
            if nxt == j or j in indices_in(x, nxt, j + 1):
                s ^= x; c ^= info(x).h; n -= sg
            self._push(x, sg, j)
        self.cells[j] = [s, c, n]
        self._peel([j])

    def _pure(self, cell):
        s, c, n = cell
        return (n == 1 or n == -1) and s != 0 and h64(s) == c

    def _peel(self, queue):
        cells = self.cells
        while queue:
            j = queue.pop()
            cell = cells.get(j)
            if cell is None or not self._pure(cell):
                continue
            x, sg = cell[0], cell[2]
            if x in self.decoded:
                continue
            self.decoded[x] = sg
            hx = info(x).h
            self.fp ^= hx
            self.cnt += sg
            for i in indices_in(x, 0, self.maxj + 1):
                ci = cells.get(i)
                if ci is not None:
                    ci[0] ^= x; ci[1] ^= hx; ci[2] -= sg
                    queue.append(i)
            self._push(x, sg, self.maxj)

    def recovered(self, sign):
        return [x for x, sg in self.decoded.items() if sg == sign]
