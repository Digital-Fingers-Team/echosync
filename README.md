# EchoSync

Broadcast-based data synchronization for resource-constrained LoRa networks, using
Rateless Invertible Bloom Lookup Tables (RIBLT). This repository holds the
discrete-event simulator, the experiment data and the research paper
([`EchoSync_Research_Paper_v2.pdf`](EchoSync_Research_Paper_v2.pdf)).

## What is in here

| File | Purpose |
|---|---|
| `riblt.py` | RIBLT encoder and streaming peeling decoder (accepts lost and out-of-order symbols) |
| `sim.py` | LoRa airtime (Semtech AN1200.13), duty cycle, Gilbert–Elliott loss, energy model, and the protocols: EchoSync 2-phase / 1-phase, pairwise RIBLT star, full-state broadcast, oracle |
| `experiments.py` | All 2,782 simulation runs (8 experiments) → `results/*.json` |
| `analyze.py` | Figures (`figs/*.svg`) and `summary.json` |
| `test_sim.py` | Validation tests (airtime vs Semtech calculator, codec correctness, overhead, loss, convergence) |
| `paper.html`, `fig_protocol.svg` | Paper source; printed to PDF with headless Chromium |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python test_sim.py        # validation tests
.venv/bin/python experiments.py     # ~9 min on 2 cores; or: experiments.py loss scaling
.venv/bin/python analyze.py         # figures + summary.json
chromium-browser --headless --no-pdf-header-footer --virtual-time-budget=15000 \
  --print-to-pdf=EchoSync_Research_Paper_v2.pdf file://$PWD/paper.html
```

Every run is seeded, so the results regenerate exactly.

## Key results (simulation, SF9, 1% duty cycle)

- The original single-phase design never brings all nodes to the full union; the two-phase protocol does (100%).
- RIBLT has a heavy tail under frame loss; 64-frame NACK prefix repair cuts airtime 3.2× and timeouts 115× at 30% loss.
- 10 nodes: 88% less airtime and 93% less time than sequential pairwise RIBLT; 100 nodes: 3.2 h vs 45 days.
- 100% convergence at every loss rate from 0% to 30% (independent and bursty).

Hardware validation on ESP32 + SX1276/SX1278 nodes is future work; all results here are simulated.
