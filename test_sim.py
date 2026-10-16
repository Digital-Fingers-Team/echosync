"""Validation tests: run with  python -m pytest -q  or  python test_sim.py"""
import random

from riblt import Encoder, Decoder, fingerprint, subtract
from sim import Phy, Cfg, echosync, pairwise, workload


def test_airtime_matches_semtech_calculator():
    # reference values: Semtech LoRa calculator / TTN airtime tables
    assert abs(Phy(sf=7).toa(20) * 1e3 - 56.576) < 0.01
    assert abs(Phy(sf=12).toa(51) * 1e3 - 2465.792) < 0.01
    assert abs(Phy(sf=9).toa(20) * 1e3 - 185.344) < 0.01


def _decode(remote, local, loss=0.0, rng=None):
    enc_r, enc_l = Encoder(remote), Encoder(local)
    d = Decoder(fingerprint(remote) ^ fingerprint(local), len(remote) - len(local))
    j = 0
    while not d.done():
        sym = subtract(enc_r.symbol(j), enc_l.symbol(j))
        if not rng or rng.random() >= loss:
            d.add(j, sym)
        j += 1
        assert j < 100000
    return d, j


def test_full_sets_decode_exact_difference():
    rng = random.Random(1)
    for _ in range(20):
        common = {rng.getrandbits(128) | 1 for _ in range(300)}
        a = {rng.getrandbits(128) | 1 for _ in range(rng.randint(0, 40))}
        b = {rng.getrandbits(128) | 1 for _ in range(rng.randint(0, 40))}
        d, _ = _decode(common | a, common | b)
        assert set(d.recovered(+1)) == a and set(d.recovered(-1)) == b


def test_difference_stream_equals_subtracted_full_streams():
    rng = random.Random(2)
    common = {rng.getrandbits(128) | 1 for _ in range(200)}
    a = {rng.getrandbits(128) | 1 for _ in range(15)}
    b = {rng.getrandbits(128) | 1 for _ in range(9)}
    full_r, full_l, diff = Encoder(common | a), Encoder(common | b), Encoder(a, b)
    for j in range(300):
        assert subtract(full_r.symbol(j), full_l.symbol(j)) == diff.symbol(j)


def test_overhead_converges_to_about_1_35():
    rng = random.Random(3)
    tot_d = tot_m = 0
    for _ in range(10):
        a = {rng.getrandbits(128) | 1 for _ in range(1000)}
        _, m = _decode(a, set())
        tot_d += 1000; tot_m += m
    assert 1.30 < tot_m / tot_d < 1.45, tot_m / tot_d


def test_decoding_survives_symbol_loss():
    rng = random.Random(4)
    a = {rng.getrandbits(128) | 1 for _ in range(200)}
    d, _ = _decode(a, set(), loss=0.3, rng=rng)
    assert set(d.recovered(+1)) == a


def test_protocols_converge_under_loss():
    rng = random.Random(5)
    for loss, burst in [(0.0, 1), (0.2, 1), (0.3, 4)]:
        sets = workload(rng, 8)
        cfg = Cfg(loss=loss, burst=burst)
        assert echosync(sets, cfg, random.Random(6))["converged"] == 1.0
        assert pairwise(sets, cfg, random.Random(6))["converged"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
