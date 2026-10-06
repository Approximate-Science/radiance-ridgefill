#!/usr/bin/env python3
"""rho_ref.py -- a NumPy transcription of the reference +st rho (research repo ridgefill/b0/st_hook.py:25-34
and 229-276, StController._rho; log_gate from b0/worker_ext.py:71-74), written as the fixture
tests/kernel_test.cpp holds ridgefill_rho_update to (R34).

The reference computes each chunk in closed form (a cumulative sum of the log decays); ridgefill_rho_update
runs the same recurrence one row at a time. The two are equal in exact arithmetic and differ only by
rounding, which is what the test's tolerance is for.

    python3 tests/rho_ref.py > tests/fixtures/rho_numpy.txt     (needs numpy only)
"""
import sys

import numpy as np

HEADS = 6
CHUNKS = (37, 64, 50)        # rows per chunk, carried across like one prompt's approximate chunks
SEED = 20261004


def log_gate(a, a_log, dt_bias):
    """worker_ext.py log_gate: -exp(A_log) * softplus(a + dt_bias), fp32, softplus threshold 20 (torch's)."""
    x = a.astype(np.float32) + dt_bias.astype(np.float32)
    sp = np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0)))).astype(np.float32)
    return (-np.exp(a_log.astype(np.float32)) * sp).astype(np.float32)


def rho_chunk(g, approx, n_mass, d_mass):
    """st_hook.py _rho for one chunk: g [rows, heads] <= 0, approx [rows] 1 = approximated row."""
    cs = np.cumsum(g, axis=0, dtype=np.float32)
    w = np.exp(cs[-1] - cs)                       # each row's decay to the chunk end
    carry = np.exp(cs[-1])                        # the carried mass's decay over the chunk
    d_mass = (carry * d_mass + w.sum(0)).astype(np.float32)
    n_mass = (carry * n_mass + (w * approx.astype(np.float32)[:, None]).sum(0)).astype(np.float32)
    return n_mass, d_mass


def floats(name, v):
    return f"{name} {v.size} " + " ".join(f"{float(x):.9g}" for x in v.reshape(-1))


def main():
    rng = np.random.default_rng(SEED)
    a_log = rng.uniform(-1.0, 2.5, HEADS).astype(np.float32)
    a_log[0] = -4.0                                # a slow-forgetting head: D grows with the prompt
    a_log[-1] = 4.0                                # a fast-forgetting head: rho follows the last rows
    dt_bias = rng.uniform(-3.0, 1.0, HEADS).astype(np.float32)
    out = [f"ridgefill-rho-fixture 1 heads {HEADS} chunks {len(CHUNKS)} seed {SEED}",
           floats("A_log", a_log), floats("dt_bias", dt_bias)]
    n_mass = np.zeros(HEADS, np.float32)
    d_mass = np.zeros(HEADS, np.float32)
    seen = False
    for c, rows in enumerate(CHUNKS):
        a = (rng.standard_normal((rows, HEADS)) * 2.0).astype(np.float32)
        a[0, 0] = 25.0                             # past softplus's threshold
        approx = (rng.uniform(size=rows) > 0.1).astype(np.int32)
        if c == 0:
            approx[:] = 1                          # no exact row yet: the reference's rho is "None" = 1
        n_mass, d_mass = rho_chunk(log_gate(a, a_log, dt_bias), approx, n_mass, d_mass)
        seen = seen or not approx.all()
        rho = np.clip(n_mass / np.maximum(d_mass, 1e-30), 0.0, 1.0) if seen else np.ones(HEADS)
        out += [f"chunk {c} rows {rows}", floats("a", a), "mask " + str(rows) + " " +
                " ".join(str(int(x)) for x in approx),
                floats("N", n_mass), floats("D", d_mass), floats("rho", rho.astype(np.float32))]
    out.append("end")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
