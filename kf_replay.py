"""
Replay a recorded flight (flight.npz from kf_crazyflie.py, with raw IMU)
through the KF offline:

    python kf_replay.py flight3.npz --fit            fit the in-flight noise model
    python kf_replay.py flight3.npz --coast          IMU-only coasting test
    python kf_replay.py flight3.npz --fit --coast --drag --out imu_noise_flight.json
    python kf_replay.py data/*.npz --rw-check --fit  several flights at once

Several flights can be given; --fit then maximises their joint likelihood.
Each flight is only used from 3 s after liftoff until 0.5 s before the first
jump in the onboard estimate (> 5 m/s or a > 8 cm height step: the Flow deck
losing it, or the range sensor seeing an object) or 1 s before the motors
stop — whichever comes first.

--fit
    Replays the KF with the onboard estimate fused as position measurements
    and picks the noise parameters (--fit-params, default acc_vrw, tilt_sigma,
    tilt_tau, bias_rw) that maximise the likelihood of the innovations while
    the motors run. A motors-off Allan fit can't see what matters in flight —
    attitude error, thrust/drag effects, propwash — this can.

--rw-check
    Is the in-flight bias a Gaussian random walk? Without the KF: the
    acceleration error eps = (IMU world accel) - (onboard accel) is integrated
    to a velocity error, and its Allan deviation is computed and pooled over
    the flights. White noise falls as tau^-1/2, a constant offset is flat, a
    random walk rises as tau^+1/2 with level K sqrt(tau/3), K = bias_rw.
    The fit of N^2/tau + C + K^2 tau/3 gives an estimate of bias_rw that is
    independent of the KF likelihood fit.

--coast
    The handover, offline, many times: from every 0.5 s of the aided hover,
    the KF continues on the IMU alone for 2 s and 4 s; its displacement is
    compared with the onboard one. Reports the drift and whether the KF's
    3-sigma actually covers it. This predicts what a KF segment can do before
    flying it.

--drag
    Rotor drag: on a quadrotor the body x/y accelerometer reads mostly drag,
    a_xy = -k v_body_xy (flight3: corr -0.79). Used as a measurement it gives
    an IMU-only handle on velocity. k and the residual are fitted from the
    aided data; the residual is time-correlated, so updates use 10-sample
    means and the residual sigma x DRAG_SIGMA_MULT (flight3: x1 at 100 Hz made
    things worse, x10 at 10 Hz cut the 4 s drift 77 -> 47 cm). Written into
    the output JSON; the coasting test runs with and without it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, replace

import numpy as np

import kf_crazyflie as k

REF_SIGMA = 0.03
DRAG_SIGMA_MULT = 10.0


def load(path):
    d = np.load(path)
    if 't_imu' not in d:
        raise SystemExit(f"{path} has no raw IMU — record it with the current kf_crazyflie.py")
    R = np.array([k.rotation(math.radians(r), k.PITCH_SIGN * math.radians(p), math.radians(y))
                  for r, p, y in d['rpy_deg']])
    thrust_imu = np.interp(d['t_imu'], d['t_ref'], d['thrust'])
    t_ref, p_ref, on_ref = d['t_ref'], d['p_ref'], d['thrust'] > 1.0
    # the clean part of the flight: from liftoff until the first jump in the
    # onboard estimate, or until the motors stop
    t_on = t_ref[on_ref]
    step = np.diff(p_ref, axis=0)
    speed = np.linalg.norm(step[:, :2], axis=1) / np.diff(t_ref)
    jump = on_ref[1:] & ((speed > 5.0) | (np.abs(step[:, 2]) > 0.08))
    valid_until = t_on[-1] - 1.0
    if jump.any():
        valid_until = min(valid_until, t_ref[1:][jump][0] - 0.5)
    return dict(t=d['t_imu'], acc=d['acc'], R=R, on=thrust_imu > 1.0,
                t_ref=t_ref, p_ref=p_ref, thrust=d['thrust'], name=path,
                t_liftoff=float(t_on[0]), valid_until=float(valid_until))


def onboard_velocity(f):
    from scipy.signal import savgol_filter
    dt = float(np.median(np.diff(f['t_ref'])))
    v = np.stack([savgol_filter(f['p_ref'][:, j], 11, 2, deriv=1, delta=dt) for j in range(3)], 1)
    return np.stack([np.interp(f['t'], f['t_ref'], v[:, j]) for j in range(3)], 1)


# ── drag ────────────────────────────────────────────────────────────────────
def fit_drag(f, start):
    """kd, sigma from the aided flight: body xy accel vs onboard body velocity."""
    v = onboard_velocity(f)
    m = f['on'] & (f['t'] > start)
    vb = np.einsum('nji,nj->ni', f['R'][m], v[m])[:, :2]       # R^T v
    a = f['acc'][m][:, :2]
    a = a - a.mean(0)
    kd = -float(np.sum(a * vb) / np.sum(vb * vb))
    sigma = float(np.std(a + kd * vb))
    return kd, sigma


# ── replay ──────────────────────────────────────────────────────────────────
def replay(f, cfg, aid=True, snapshot_times=(), nll_from=None, nll_to=np.inf):
    """Aided replay. Returns (summed NLL of the position innovations while
    flying in [nll_from, nll_to], their count, {snapshot time: (kf copy, IMU index)})."""
    t, acc, R = f['t'], f['acc'], f['R']
    calib = t < t[0] + 1.5
    kf = k.InertialKF(cfg)
    kf.reset(f['p_ref'][0], k.calibrate_bias(acc[calib], R[calib]), t[np.nonzero(calib)[0][-1]])
    j_ref = int(np.searchsorted(f['t_ref'], kf.t))
    snaps = sorted(snapshot_times)
    out, nll, n = {}, 0.0, 0
    H = np.zeros((3, k.NX))
    H[:, k.P] = np.eye(3)
    Rm = np.eye(3) * REF_SIGMA ** 2
    for i in range(np.nonzero(calib)[0][-1] + 1, len(t)):
        kf.predict(acc[i], R[i], t[i])
        if not f['on'][i] and i % 5 == 0:
            kf.update_zero_velocity()
        while j_ref < len(f['t_ref']) and f['t_ref'][j_ref] <= t[i]:
            if aid:
                nu = f['p_ref'][j_ref] - kf.x[k.P]
                S = kf.P[k.P, k.P] + Rm
                if f['on'][i] and nll_from is not None and nll_from < t[i] < nll_to:
                    nll += 0.5 * (nu @ np.linalg.solve(S, nu) + np.linalg.slogdet(S)[1])
                    n += 1
                kf._update(f['p_ref'][j_ref], H, Rm, gate=False)
            j_ref += 1
        while snaps and t[i] >= snaps[0]:
            out[snaps.pop(0)] = (copy.deepcopy(kf), i)
    return nll, n, out


def coast(f, kf, i0, T, drag=False):
    """IMU only from index i0 for T s, as CrazyflieKF does after the handover.
    Returns (KF displacement, final 1-sigma)."""
    kf = copy.deepcopy(kf)
    p0 = kf.x[k.P].copy()
    t, acc, R = f['t'], f['acc'], f['R']
    i, buf = i0 + 1, []
    while i < len(t) and t[i] <= t[i0] + T:
        kf.predict(acc[i], R[i], t[i])
        if drag and kf.cfg.drag_k > 0:
            buf.append(acc[i])
            if len(buf) >= kf.cfg.drag_every:
                kf.update_drag(np.mean(buf, axis=0), R[i])
                buf = []
        i += 1
    return kf.x[k.P] - p0, kf.uncertainty()


def coast_test(f, cfg, Ts=(2.0, 4.0), drag=False, label=""):
    t_lo, t_hi = f['t_liftoff'] + 4.0, f['valid_until'] - max(Ts)
    starts = list(np.arange(t_lo, t_hi, 0.5))
    if not starts:
        print("flight too short for the coasting test")
        return
    _, _, snaps = replay(f, cfg, snapshot_times=starts)
    p_on = lambda tt: np.array([np.interp(tt, f['t_ref'], f['p_ref'][:, j]) for j in range(3)])
    for T in Ts:
        errs, inside = [], []
        for t0, (kf, i0) in snaps.items():
            d_kf, sig = coast(f, kf, i0, T, drag)
            d_on = p_on(t0 + T) - p_on(t0)
            e = d_kf - d_on
            errs.append(e)
            inside.append(np.all(np.abs(e) <= 3 * sig))
        e = np.array(errs)
        exy = np.linalg.norm(e[:, :2], axis=1)
        print(f"  {label:22s} {T:.0f} s coast, n={len(e):3d}: |xy| median {np.median(exy) * 100:5.1f} cm, "
              f"95% {np.percentile(exy, 95) * 100:5.1f} cm; |z| median "
              f"{np.median(np.abs(e[:, 2])) * 100:4.1f} cm; inside 3-sigma {np.mean(inside) * 100:3.0f}%")


def rw_check(fs, taus=None):
    """Allan deviation of the in-flight horizontal acceleration error, pooled
    over flights; fit N^2/tau + C + K^2 tau/3. Returns K (-> bias_rw)."""
    from scipy.optimize import least_squares
    from scipy.signal import savgol_filter
    if taus is None:
        taus = np.array([0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0])
    sums = np.zeros((len(taus), 2))
    counts = np.zeros(len(taus))
    print("[rw-check] flights and the stretch of each that is used:")
    for f in fs:
        m = (f['t'] > f['t_liftoff'] + 2.0) & (f['t'] < f['valid_until'])
        if m.sum() < 300:
            print(f"  {f['name']:32s} too short, skipped")
            continue
        t = f['t'][m]
        calib = f['t'] < f['t'][0] + 1.5
        bias = k.calibrate_bias(f['acc'][calib], f['R'][calib])
        aw = np.einsum('nij,nj->ni', f['R'][m], f['acc'][m] - bias)[:, :2]   # gravity is all z
        dt_ref = float(np.median(np.diff(f['t_ref'])))
        v_on = np.stack([np.interp(t, f['t_ref'], savgol_filter(f['p_ref'][:, j], 11, 2, deriv=1,
                                                                    delta=dt_ref)) for j in range(2)], 1)
        dt = np.diff(t)[:, None]
        v_imu = np.concatenate(([[0.0, 0.0]], np.cumsum((aw[1:] + aw[:-1]) / 2 * dt, axis=0)))
        theta = v_imu - v_on                       # integrated acceleration error, per axis
        # resample on a uniform grid for the cluster arithmetic
        tg = np.arange(t[0], t[-1], 0.01)
        th = np.stack([np.interp(tg, t, theta[:, j]) for j in range(2)], 1)
        print(f"  {f['name']:32s} {tg[0]:5.1f}-{tg[-1]:5.1f} s ({tg[-1] - tg[0]:4.1f} s)")
        for i, tau in enumerate(taus):
            mm = int(round(tau / 0.01))
            if 3 * mm >= len(th):
                continue
            dd = th[2 * mm:] - 2 * th[mm:-mm] + th[:-2 * mm]
            sums[i] += np.sum(dd ** 2, axis=0) / (2 * tau ** 2)
            counts[i] += len(dd)
    ok = counts > 0
    avar = (sums[ok] / counts[ok][:, None]).mean(axis=1)    # x and y pooled
    t_ok = taus[ok]
    adev = np.sqrt(avar)

    def resid(lx):
        N, C, K = np.exp(lx)
        return np.log(np.sqrt(N ** 2 / t_ok + C ** 2 + K ** 2 * t_ok / 3)) - np.log(adev)
    lx = least_squares(resid, np.log([adev[0] * math.sqrt(t_ok[0]), adev.min() / 2,
                                      adev[-1] * math.sqrt(3 / t_ok[-1])])).x
    N, C, K = np.exp(lx)
    print("\n  tau [s]   Allan dev [m/s^2]   local slope   (white -1/2, offset 0, random walk +1/2)")
    for i, (tt, a) in enumerate(zip(t_ok, adev)):
        slope = "" if i == 0 else f"{math.log(a / adev[i - 1]) / math.log(tt / t_ok[i - 1]):+.2f}"
        print(f"  {tt:6.1f}   {a:10.4f}          {slope}")
    print(f"\n  fit: white N = {N:.4f} m/s/sqrt(s), floor C = {C:.4f} m/s^2, "
          f"random walk K = {K:.4f} m/s^2/sqrt(s)")
    print(f"  -> bias_rw ~ {K:.4f}: over a 6 s IMU-only move the bias changes by "
          f"~{K * math.sqrt(6):.3f} m/s^2 (1-sigma)")
    return K


def fit(fs, cfg0, iters=150, names=('acc_vrw', 'tilt_sigma', 'tilt_tau', 'bias_rw')):
    """Joint maximum likelihood over the flights in fs (mean NLL per innovation)."""
    from scipy.optimize import minimize
    names = list(names)

    def obj(lx):
        c = replace(cfg0, **{n: float(math.exp(v)) for n, v in zip(names, lx)})
        total, count = 0.0, 0
        try:
            for f in fs:
                nll, n, _ = replay(f, c, nll_from=f['t_liftoff'] + 3.0, nll_to=f['valid_until'])
                total, count = total + nll, count + n
        except np.linalg.LinAlgError:
            return 1e9
        return total / max(count, 1)

    x0 = np.log([max(getattr(cfg0, n), 1e-4) for n in names])
    print(f"[fit] start NLL {obj(x0):.3f}")
    res = minimize(obj, x0, method='Nelder-Mead',
                   options=dict(maxiter=iters, xatol=0.05, fatol=1e-3))
    best = {n: float(math.exp(v)) for n, v in zip(names, res.x)}
    print(f"[fit] NLL {res.fun:.3f} after {res.nfev} replays")
    for n in names:
        print(f"  {n:11s} {getattr(cfg0, n):10.4g} -> {best[n]:10.4g}")
    return replace(cfg0, **best)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('flights', nargs='+')
    ap.add_argument('--noise', help='starting noise model (imu_noise.json)')
    ap.add_argument('--fit', action='store_true')
    ap.add_argument('--coast', action='store_true')
    ap.add_argument('--drag', action='store_true')
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--fit-params', default='acc_vrw,tilt_sigma,tilt_tau,bias_rw',
                    help='comma-separated KFConfig fields to fit')
    ap.add_argument('--rw-check', action='store_true')
    ap.add_argument('--out', default='imu_noise_flight.json')
    args = ap.parse_args()

    fs = [load(p) for p in args.flights]
    cfg0 = k.KFConfig.from_json(args.noise) if args.noise else k.KFConfig()
    for f in fs:
        print(f"[replay] {f['name']}: {len(f['t'])} IMU samples, using "
              f"{f['valid_until'] - f['t_liftoff'] - 3.0:.1f} s of flight")

    if args.rw_check:
        rw_check(fs)

    cfg = cfg0
    if args.fit:
        cfg = fit(fs, cfg0, args.iters, args.fit_params.split(','))

    if args.drag:
        drags = [fit_drag(f, f['t_liftoff'] + 3.0) for f in fs]
        kd, sigma = float(np.median([d[0] for d in drags])), float(np.median([d[1] for d in drags]))
        cfg = replace(cfg, drag_k=kd, drag_sigma=sigma * DRAG_SIGMA_MULT)
        print(f"[drag] k = {kd:.3f} 1/s, residual sigma {sigma:.3f} m/s^2 "
              f"-> drag_sigma {cfg.drag_sigma:.3f} at {100 // cfg.drag_every} Hz")

    if args.fit or args.drag:
        with open(args.out, 'w') as fh:
            json.dump({"kf": {n: getattr(cfg, n) for n in
                              ('acc_vrw', 'gm_sigma', 'gm_tau', 'bias_rw', 'tilt_sigma',
                               'tilt_tau', 'drag_k', 'drag_sigma', 'drag_every')},
                       "source": f"kf_replay.py on {', '.join(args.flights)}"}, fh, indent=2)
        print(f"[replay] written to {args.out}")

    if args.coast:
        print("[coast] IMU-only from points in the aided flight, vs onboard displacement:")
        name = "fitted" if args.fit else "given"
        for f in fs:
            print(f"  {f['name']}")
            if args.fit:
                coast_test(f, cfg0, label="before fit")
            coast_test(f, cfg, label=name)
            if cfg.drag_k > 0:
                coast_test(f, cfg, drag=True, label=name + " + drag")
