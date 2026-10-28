"""
Formation patterns as functions of time.

Every pattern returns, for drone slot i out of n at time t:
    c_dot : (3,) velocity of the formation centre
    d     : (3,) offset of slot i from the centre
    d_dot : (3,) velocity of that offset

Each drone evaluates the pattern itself, so no waypoints are streamed.
Add a new pattern by writing a function with the same signature and
registering it in PATTERNS.
"""
import math
import numpy as np

SPACING = 0.6   # m, distance between neighbouring slots
ZERO = np.zeros(3)


def _line_offset(i, n, spacing=SPACING):
    return np.array([0.0, (i - (n - 1) / 2.0) * spacing, 0.0])


def _ring(i, n, angle, rate, spacing=SPACING):
    """Slot on a ring; for n=2 the ring diameter equals `spacing`."""
    r = spacing / (2.0 * math.sin(math.pi / n)) if n > 1 else 0.0
    a = angle + 2.0 * math.pi * i / n - math.pi / 2   # slot 0 starts at -y
    d = r * np.array([math.cos(a), math.sin(a), 0.0])
    d_dot = r * rate * np.array([-math.sin(a), math.cos(a), 0.0])
    return d, d_dot


def hover(t, i, n):
    """Static line, centre still. Use this first on hardware."""
    return ZERO, _line_offset(i, n), ZERO


def line(t, i, n):
    """Static line along y; centre moves +x for 4 s, back for 4 s."""
    v = 0.25 if (t % 8.0) < 4.0 else -0.25
    return np.array([v, 0.0, 0.0]), _line_offset(i, n), ZERO


def circle(t, i, n):
    """Formation rotates about a fixed centre, one turn every 8 s."""
    w = 2.0 * math.pi / 8.0
    d, d_dot = _ring(i, n, w * t, w)
    return ZERO, d, d_dot


def eight(t, i, n):
    """Centre follows a figure-eight, formation stays a fixed line."""
    A, T = 0.6, 12.0
    w = 2.0 * math.pi / T
    c_dot = np.array([A * w * math.cos(w * t), A * w * math.cos(2 * w * t), 0.0])
    return c_dot, _line_offset(i, n), ZERO


def swap(t, i, n):
    """Slots rotate half a turn and back (12 s cycle), never overflying."""
    tau = t % 12.0
    if 1.0 <= tau < 5.0:
        u = (tau - 1.0) / 4.0
        s, s_dot = 0.5 * (1 - math.cos(math.pi * u)), 0.5 * math.pi * math.sin(math.pi * u) / 4.0
    elif 5.0 <= tau < 6.0:
        s, s_dot = 1.0, 0.0
    elif 6.0 <= tau < 10.0:
        u = (tau - 6.0) / 4.0
        s, s_dot = 0.5 * (1 + math.cos(math.pi * u)), -0.5 * math.pi * math.sin(math.pi * u) / 4.0
    else:
        s, s_dot = 0.0, 0.0
    d, d_dot = _ring(i, n, math.pi * s, math.pi * s_dot)
    return ZERO, d, d_dot


PATTERNS = {f.__name__: f for f in (hover, line, circle, eight, swap)}


def get_pattern(name):
    try:
        return PATTERNS[name]
    except KeyError:
        raise ValueError(f"unknown pattern '{name}', choose from {list(PATTERNS)}")
