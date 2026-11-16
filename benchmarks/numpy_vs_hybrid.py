"""Same file run two ways: `python bench.py` and `pyrust run bench.py`.

The plain-loop functions below are ordinary Python. Under plain Python they run
interpreted; under PyRust's hybrid tier they are swapped for native code. The
numpy and numba versions are identical in both runs.
"""
import sys
import time

import numpy as np

try:
    import numba
except Exception:
    numba = None


# ---- 1. vectorisable: sum of i*i % 7 -------------------------------------
def sumsq_loop(n: int):
    t = 0
    for i in range(n):
        t += i * i % 7
    return t


def sumsq_numpy(n):
    a = np.arange(n, dtype=np.int64)
    return int((a * a % 7).sum())


# ---- 2. branchy, data-dependent: longest Collatz chain -------------------
def collatz_loop(limit: int):
    best = 0
    for start in range(1, limit):
        n = start
        steps = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        if steps > best:
            best = steps
    return best


def collatz_numpy(limit):
    n = np.arange(1, limit, dtype=np.int64)
    steps = np.zeros(limit - 1, dtype=np.int64)
    active = n != 1
    while active.any():
        cur = n[active]
        even = cur % 2 == 0
        cur = np.where(even, cur // 2, 3 * cur + 1)
        n[active] = cur
        steps[active] += 1
        active = n != 1
    return int(steps.max())


# ---- 3. numeric kernel: Mandelbrot point count ---------------------------
def mandel_loop(size: int, maxiter: int):
    inside = 0
    for py in range(size):
        ci = -1.5 + 3.0 * py / size
        for px in range(size):
            cr = -2.0 + 3.0 * px / size
            zr = 0.0
            zi = 0.0
            k = 0
            while k < maxiter and zr * zr + zi * zi <= 4.0:
                t = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = t
                k += 1
            if k == maxiter:
                inside += 1
    return inside


def mandel_numpy(size, maxiter):
    cr = -2.0 + 3.0 * np.arange(size) / size
    ci = -1.5 + 3.0 * np.arange(size) / size
    c_r, c_i = np.meshgrid(cr, ci)
    zr = np.zeros_like(c_r)
    zi = np.zeros_like(c_r)
    alive = np.ones(c_r.shape, dtype=bool)
    for _ in range(maxiter):
        t = zr * zr - zi * zi + c_r
        zi = np.where(alive, 2.0 * zr * zi + c_i, zi)
        zr = np.where(alive, t, zr)
        alive &= zr * zr + zi * zi <= 4.0
    return int(alive.sum())


def best_of(fn, args, repeat):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, result


CASES = [
    ("sumsq", sumsq_loop, sumsq_numpy, (5_000_000,)),
    ("collatz", collatz_loop, collatz_numpy, (300_000,)),
    ("mandel", mandel_loop, mandel_numpy, (400, 200)),
]

native = getattr(sumsq_loop, "__pyrust_native__", False)
label = "pyrust-native" if native else "python-loop"
for name, loop_fn, numpy_fn, args in CASES:
    t_loop, r_loop = best_of(loop_fn, args, 3 if native else 1)
    t_np, r_np = best_of(numpy_fn, args, 3)
    assert r_loop == r_np, (name, r_loop, r_np)
    print(f"{name:8s} {label:14s} {t_loop * 1000:10.2f} ms")
    print(f"{name:8s} {'numpy':14s} {t_np * 1000:10.2f} ms")
    if numba is not None and not native:
        orig = getattr(loop_fn, "__wrapped__", loop_fn)
        jitted = numba.njit(cache=False)(orig)
        t0 = time.perf_counter()
        jitted(*args)
        t_first = time.perf_counter() - t0
        t_nb, r_nb = best_of(jitted, args, 3)
        assert r_nb == r_loop
        print(f"{name:8s} {'numba':14s} {t_nb * 1000:10.2f} ms   (first call incl. compile {t_first * 1000:.0f} ms)")
    print(f"{name:8s} result={r_loop}")
    sys.stdout.flush()
